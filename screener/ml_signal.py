"""ML Signal Confidence Layer for breakout predictions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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


class BreakoutFeatureExtractor:
    """Extract quantitative features from OHLCV bars for ML classification."""

    FEATURE_COLUMNS = [
        "rvol_5d",
        "rvol_20d",
        "volume_trend",
        "volume_momentum",
        "returns_5d",
        "returns_20d",
        "returns_55d",
        "close_vs_ema20",
        "close_vs_ema50",
        "ema20_vs_ema50",
        "ema50_vs_ema200",
        "ATR_14_pct",
        "volatility_percentile_90d",
        "benchmark_return_20d",
        "breadth_ratio",
    ]

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def extract(
        self,
        bars: pd.DataFrame,
        benchmark_bars: pd.DataFrame | None = None,
        all_bars: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of features aligned to ``bars`` index."""
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

        # Volume profile
        vol_ma5 = volume.rolling(5, min_periods=1).mean()
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        vol_ma10 = volume.rolling(10, min_periods=1).mean()
        rvol_5d = volume / vol_ma5.replace(0, np.nan)
        rvol_20d = volume / vol_ma20.replace(0, np.nan)
        volume_trend = vol_ma10.pct_change(5)
        volume_momentum = vol_ma5 / vol_ma20.replace(0, np.nan)

        # Price momentum
        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)
        returns_55d = close.pct_change(55)
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        ema200 = self._ema(close, 200)
        close_vs_ema20 = close / ema20.replace(0, np.nan) - 1.0
        close_vs_ema50 = close / ema50.replace(0, np.nan) - 1.0

        # Trend alignment
        ema20_vs_ema50 = ema20 / ema50.replace(0, np.nan) - 1.0
        ema50_vs_ema200 = ema50 / ema200.replace(0, np.nan) - 1.0

        # Volatility regime
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

        # Market context
        benchmark_return_20d = pd.Series(np.nan, index=df.index, dtype=float)
        if benchmark_bars is not None and not benchmark_bars.empty:
            bench = benchmark_bars.copy().sort_index()
            if "close" in bench.columns:
                bench_close = pd.to_numeric(bench["close"], errors="coerce")
                benchmark_return_20d = bench_close.pct_change(20).reindex(df.index)

        breadth_ratio = pd.Series(np.nan, index=df.index, dtype=float)
        if all_bars is not None and len(all_bars) > 0:
            changes: list[pd.Series] = []
            for sym, sym_bars in all_bars.items():
                if sym_bars is None or sym_bars.empty:
                    continue
                sb = sym_bars.copy().sort_index()
                if "close" not in sb.columns:
                    continue
                sb_close = pd.to_numeric(sb["close"], errors="coerce")
                change = sb_close.diff().reindex(df.index)
                changes.append(change)
            if changes:
                changes_df = pd.concat(changes, axis=1)
                advances = (changes_df > 0).sum(axis=1)
                declines = (changes_df < 0).sum(axis=1)
                total = advances + declines
                breadth_ratio = advances / total.replace(0, np.nan)

        features = pd.DataFrame(
            {
                "rvol_5d": rvol_5d,
                "rvol_20d": rvol_20d,
                "volume_trend": volume_trend,
                "volume_momentum": volume_momentum,
                "returns_5d": returns_5d,
                "returns_20d": returns_20d,
                "returns_55d": returns_55d,
                "close_vs_ema20": close_vs_ema20,
                "close_vs_ema50": close_vs_ema50,
                "ema20_vs_ema50": ema20_vs_ema50,
                "ema50_vs_ema200": ema50_vs_ema200,
                "ATR_14_pct": ATR_14_pct,
                "volatility_percentile_90d": volatility_percentile_90d,
                "benchmark_return_20d": benchmark_return_20d,
                "breadth_ratio": breadth_ratio,
            },
            index=df.index,
        )
        return features.fillna(0.0)


@dataclass
class SignalConfidenceModel:
    """XGBoost classifier wrapper for signal confidence."""

    model: Any | None = None
    feature_names: list[str] = field(
        default_factory=lambda: list(BreakoutFeatureExtractor.FEATURE_COLUMNS)
    )
    metrics: dict[str, float] | None = None

    def train(
        self, trades: list[Trade], bars_by_symbol: dict[str, pd.DataFrame]
    ) -> SignalConfidenceModel:
        """Train on historical trades.  Win = return_pct > 0."""
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = BreakoutFeatureExtractor()
        X_rows: list[pd.DataFrame] = []
        y: list[int] = []

        for trade in trades:
            bars = bars_by_symbol.get(trade.ticker)
            if bars is None or bars.empty:
                continue
            features = extractor.extract(bars)
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
        X = X[self.feature_names]
        X = X.fillna(0.0)
        y_arr = np.array(y)

        if len(set(y_arr)) < 2:
            raise ValueError("All trades have the same label; cannot train classifier.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
        )

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        self.model = clf
        self.metrics = {
            "auc": float(roc_auc_score(y_test, y_proba))
            if len(set(y_test)) > 1
            else float("nan"),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "positive_rate": float(y_arr.mean()),
        }
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return probability of win (0-1) for each row."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Persist model to disk."""
        _require_ml()
        import joblib

        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> SignalConfidenceModel:
        """Load model from disk."""
        _require_ml()
        import joblib

        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.feature_names = payload.get(
            "feature_names", list(BreakoutFeatureExtractor.FEATURE_COLUMNS)
        )
        instance.metrics = payload.get("metrics")
        return instance

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        importances = self.model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df


class EnsembleConfidence:
    """Average predictions from multiple SignalConfidenceModels."""

    def __init__(self, models: list[SignalConfidenceModel]) -> None:
        self.models = models

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return ensemble-averaged probabilities."""
        if not self.models:
            raise RuntimeError("No models in ensemble.")
        probs = [m.predict(features_df) for m in self.models]
        return np.mean(probs, axis=0)
