"""ML Signal v4 — Binary classification with rich features.

Simpler than ordinal: just predict win vs loss.
Uses 27 features, sample weighting by |return|, probability calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from screener.backtester.models import Trade
from screener.ml_signal_v4 import V4FeatureExtractor


class MissingMLDependencyError(RuntimeError):
    pass


def _require_ml() -> None:
    if not _HAS_ML:
        raise MissingMLDependencyError(
            "xgboost and scikit-learn required. Install: uv pip install xgboost scikit-learn"
        )


@dataclass
class V4BinaryModel:
    """Binary win/loss model with calibration."""

    model: Any | None = None
    calibrator: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(V4FeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> V4BinaryModel:
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = V4FeatureExtractor()
        print("Pre-computing features for all symbols...")
        features_cache = {}
        for sym, bars in bars_by_symbol.items():
            if bars is None or bars.empty:
                continue
            bench = None
            if benchmark_bars is not None:
                if isinstance(benchmark_bars, dict):
                    for mkey, bdf in benchmark_bars.items():
                        if isinstance(mkey, str) and (sym.startswith(mkey) or mkey in sym):
                            bench = bdf
                            break
                elif isinstance(benchmark_bars, pd.DataFrame):
                    bench = benchmark_bars
            features_cache[sym] = extractor.extract(bars, benchmark_bars=bench)

        X_rows = []
        y = []
        weights = []
        dates = []

        for trade in trades:
            features = features_cache.get(trade.ticker)
            if features is None or features.empty:
                continue
            signal_ts = pd.Timestamp(trade.signal_date)
            mask = features.index <= signal_ts
            if not mask.any():
                continue
            row = features.loc[mask].iloc[[-1]].copy()
            if row.isna().all().all():
                continue
            X_rows.append(row)
            y.append(1 if trade.return_pct > 0 else 0)
            weights.append(max(abs(trade.return_pct), 0.001))
            dates.append(trade.signal_date)

        if not X_rows:
            raise ValueError("Could not extract features for any trade.")

        X = pd.concat(X_rows, ignore_index=True)
        X = X[self.feature_names].fillna(0.0)
        y_arr = np.array(y)
        weights_arr = np.array(weights)
        dates_arr = pd.to_datetime(dates)

        pos_rate = y_arr.mean()
        print(f"Training on {len(y_arr)} trades | Win rate: {pos_rate:.1%} | {len(self.feature_names)} features")

        if len(set(y_arr)) < 2:
            raise ValueError("All trades have the same label.")

        # Time-series split
        train_mask = dates_arr <= "2023-12-31"
        val_mask = (dates_arr >= "2024-01-01") & (dates_arr <= "2024-06-30")
        test_mask = dates_arr >= "2024-07-01"

        if test_mask.sum() < 20:
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(y_arr))
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_arr)
            train_mask = np.zeros(len(y_arr), dtype=bool)
            train_mask[train_idx] = True
            test_mask = np.zeros(len(y_arr), dtype=bool)
            test_mask[test_idx] = True
            val_mask = np.zeros(len(y_arr), dtype=bool)

        X_train = X[train_mask]
        y_train = y_arr[train_mask]
        w_train = weights_arr[train_mask]
        X_val = X[val_mask]
        y_val = y_arr[val_mask]
        X_test = X[test_mask]
        y_test = y_arr[test_mask]

        print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=2.0,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=30,
            scale_pos_weight=scale_pos_weight,
        )

        eval_set = [(X_val, y_val)] if len(y_val) > 10 else []
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calibration
        self.calibrator = None
        if len(y_val) > 50:
            val_proba = model.predict_proba(X_val)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_proba, y_val)
            self.calibrator = iso

        # Test metrics
        y_proba = model.predict_proba(X_test)[:, 1]
        if self.calibrator is not None:
            y_proba = self.calibrator.predict(y_proba)
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan")
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        self.model = model
        self.metrics = {
            "auc": float(auc),
            "accuracy": float(acc),
            "brier_score": float(brier),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "win_rate": float(pos_rate),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 400,
        }
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model not trained.")
        X = features_df[self.feature_names].fillna(0.0)
        proba = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            proba = self.calibrator.predict(proba)
        return proba

    def save(self, path: str | Path) -> None:
        _require_ml()
        import joblib
        payload = {
            "model": self.model,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> V4BinaryModel:
        _require_ml()
        import joblib
        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.calibrator = payload.get("calibrator")
        instance.feature_names = payload.get("feature_names", list(V4FeatureExtractor.FEATURE_COLUMNS))
        instance.metrics = payload.get("metrics")
        return instance

    def feature_importance(self) -> pd.DataFrame:
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df
