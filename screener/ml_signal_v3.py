"""ML Signal Confidence v3 — simplified, regime-robust.

Fixes overfitting from v2:
- Feature selection: only top 8 most robust features
- Strong L2 regularization (reg_lambda=3.0)
- Early stopping with validation set
- No interaction/rank features (too noisy for small data)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from screener.backtester.models import Trade


class MissingMLDependencyError(RuntimeError):
    """Raised when ML dependencies are not installed."""


def _require_ml() -> None:
    if not _HAS_ML:
        raise MissingMLDependencyError(
            "xgboost and scikit-learn are required for ML confidence. "
            "Install with: uv pip install xgboost scikit-learn"
        )


class SimpleFeatureExtractor:
    """Simplified feature set — only robust, interpretable features."""

    FEATURE_COLUMNS = [
        "rvol_5d",           # Relative volume (volume confirmation)
        "returns_20d",       # Intermediate momentum
        "returns_5d",        # Short-term momentum
        "close_vs_ema20",    # Distance from short-term trend
        "ema20_vs_ema50",    # Trend alignment
        "ATR_14_pct",        # Volatility regime
        "benchmark_return_20d",  # Market context
        "volatility_percentile_90d",  # Vol regime
    ]

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def extract(
        self,
        bars: pd.DataFrame,
        benchmark_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return features aligned to ``bars`` index."""
        if bars is None or bars.empty:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        df = bars.copy().sort_index()
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]

        vol_ma5 = volume.rolling(5, min_periods=1).mean()
        rvol_5d = volume / vol_ma5.replace(0, np.nan)

        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)

        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        close_vs_ema20 = close / ema20.replace(0, np.nan) - 1.0
        ema20_vs_ema50 = ema20 / ema50.replace(0, np.nan) - 1.0

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14, min_periods=1).mean()
        ATR_14_pct = atr_14 / close.replace(0, np.nan)

        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20, min_periods=1).std()

        def _percentile_in_window(x: pd.Series) -> float:
            if len(x) == 0 or x.isna().all():
                return 0.0
            return float(x.rank(pct=True).iloc[-1])

        volatility_percentile_90d = vol_20.rolling(90, min_periods=1).apply(
            _percentile_in_window, raw=False
        )

        benchmark_return_20d = pd.Series(np.nan, index=df.index, dtype=float)
        if benchmark_bars is not None and not benchmark_bars.empty:
            bench = benchmark_bars.copy().sort_index()
            if "close" in bench.columns:
                bench_close = pd.to_numeric(bench["close"], errors="coerce")
                benchmark_return_20d = bench_close.pct_change(20).reindex(df.index)

        features = pd.DataFrame(
            {
                "rvol_5d": rvol_5d,
                "returns_20d": returns_20d,
                "returns_5d": returns_5d,
                "close_vs_ema20": close_vs_ema20,
                "ema20_vs_ema50": ema20_vs_ema50,
                "ATR_14_pct": ATR_14_pct,
                "benchmark_return_20d": benchmark_return_20d,
                "volatility_percentile_90d": volatility_percentile_90d,
            },
            index=df.index,
        )
        return features.fillna(0.0)


@dataclass
class SimpleSignalModel:
    """Simplified XGBoost with strong regularization."""

    model: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(SimpleFeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> SimpleSignalModel:
        """Train with heavy regularization to avoid overfitting."""
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = SimpleFeatureExtractor()
        X_rows = []
        y = []

        for trade in trades:
            bars = bars_by_symbol.get(trade.ticker)
            if bars is None or bars.empty:
                continue
            features = extractor.extract(bars, benchmark_bars=benchmark_bars)
            if features.empty:
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

        if not X_rows:
            raise ValueError("Could not extract features for any trade.")

        X = pd.concat(X_rows, ignore_index=True)
        X = X[self.feature_names].fillna(0.0)
        y_arr = np.array(y)

        pos_rate = y_arr.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        print(f"Training on {len(y_arr)} trades, {len(self.feature_names)} features")
        print(f"Positive rate: {pos_rate:.1%}, scale_pos_weight: {scale_pos_weight:.2f}")

        if len(set(y_arr)) < 2:
            raise ValueError("All trades have the same label.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
        )

        # Heavy regularization to prevent overfitting
        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,              # Shallow trees (less complex)
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,       # Require more samples per leaf
            reg_alpha=0.5,            # L1 regularization
            reg_lambda=3.0,           # L2 regularization (strong)
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        self.model = model
        self.metrics = {
            "auc": float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else float("nan"),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "positive_rate": float(y_arr.mean()),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 200,
        }
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        _require_ml()
        import joblib
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> SimpleSignalModel:
        _require_ml()
        import joblib
        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.feature_names = payload.get("feature_names", list(SimpleFeatureExtractor.FEATURE_COLUMNS))
        instance.metrics = payload.get("metrics")
        return instance

    def feature_importance(self) -> pd.DataFrame:
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        importances = self.model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df
