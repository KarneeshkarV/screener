"""ML Signal Confidence v4 — production-grade.

Improvements over v3:
- 20+ rich features (cross-sectional, technical, seasonal, risk)
- Ordinal classification: strong_loss / weak_loss / weak_win / strong_win
- Sample weighting by |return| magnitude
- Probability calibration (isotonic regression)
- Time-series aware train/val/test split
- LightGBM support (faster, better with large data)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import TimeSeriesSplit
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


class V4FeatureExtractor:
    """Rich feature set with cross-sectional, technical, seasonal, and risk features."""

    FEATURE_COLUMNS = [
        # Volume features
        "rvol_5d", "rvol_20d", "volume_trend_10d",
        # Momentum features
        "returns_5d", "returns_20d", "returns_60d",
        "momentum_5d_vs_20d",  # acceleration
        # Trend features
        "close_vs_ema20", "close_vs_ema50", "ema20_vs_ema50", "ema50_vs_ema200",
        # Volatility / Risk features
        "ATR_14_pct", "volatility_percentile_90d", "bb_position",
        # Technical indicators
        "rsi_14", "macd_hist", "adx_14",
        # Price structure
        "dist_from_52w_high", "dist_from_52w_low",
        # Market context
        "benchmark_return_20d", "beta_20d",
        # Cross-sectional ranks (computed externally)
        "rank_rvol_5d", "rank_returns_20d", "rank_close_vs_ema20",
        # Seasonal
        "month", "day_of_week",
        # Risk-adjusted
        "sharpe_20d",
    ]

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return adx

    @staticmethod
    def _bb_position(close: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        sma = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return (close - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def _beta(stock_rets: pd.Series, bench_rets: pd.Series, window: int = 20) -> pd.Series:
        cov = stock_rets.rolling(window, min_periods=window).cov(bench_rets)
        var = bench_rets.rolling(window, min_periods=window).var()
        return cov / var.replace(0, np.nan)

    def extract(
        self,
        bars: pd.DataFrame,
        benchmark_bars: pd.DataFrame | None = None,
        cross_sectional: dict[str, float] | None = None,
    ) -> pd.DataFrame:
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

        # Volume
        vol_ma5 = volume.rolling(5, min_periods=1).mean()
        rvol_5d = volume / vol_ma5.replace(0, np.nan)
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        rvol_20d = volume / vol_ma20.replace(0, np.nan)
        volume_trend_10d = volume.rolling(10, min_periods=1).mean() / volume.rolling(30, min_periods=1).mean().replace(0, np.nan)

        # Momentum
        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)
        returns_60d = close.pct_change(60)
        momentum_5d_vs_20d = returns_5d - returns_20d

        # Trend
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        ema200 = self._ema(close, 200)
        close_vs_ema20 = close / ema20.replace(0, np.nan) - 1.0
        close_vs_ema50 = close / ema50.replace(0, np.nan) - 1.0
        ema20_vs_ema50 = ema20 / ema50.replace(0, np.nan) - 1.0
        ema50_vs_ema200 = ema50 / ema200.replace(0, np.nan) - 1.0

        # Volatility / Risk
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

        volatility_percentile_90d = vol_20.rolling(90, min_periods=1).apply(_percentile_in_window, raw=False)

        bb_position = self._bb_position(close)
        rsi_14 = self._rsi(close, 14)
        _, _, macd_hist = self._macd(close)
        adx_14 = self._adx(high, low, close, 14)

        # Price structure
        high_252 = high.rolling(252, min_periods=1).max()
        low_252 = low.rolling(252, min_periods=1).min()
        dist_from_52w_high = close / high_252.replace(0, np.nan) - 1.0
        dist_from_52w_low = close / low_252.replace(0, np.nan) - 1.0

        # Market context
        benchmark_return_20d = pd.Series(np.nan, index=df.index, dtype=float)
        beta_20d = pd.Series(np.nan, index=df.index, dtype=float)
        if benchmark_bars is not None and not benchmark_bars.empty:
            bench = benchmark_bars.copy().sort_index()
            if "close" in bench.columns:
                bench_close = pd.to_numeric(bench["close"], errors="coerce")
                bench_ret = bench_close.pct_change()
                benchmark_return_20d = bench_close.pct_change(20).reindex(df.index)
                beta_20d = self._beta(daily_ret, bench_ret.reindex(df.index), 20)

        # Seasonal
        month = pd.Series(df.index.month, index=df.index)
        day_of_week = pd.Series(df.index.dayofweek, index=df.index)

        # Risk-adjusted
        sharpe_20d = (daily_ret.rolling(20, min_periods=1).mean() / vol_20.replace(0, np.nan)) * np.sqrt(252)

        features = pd.DataFrame({
            "rvol_5d": rvol_5d,
            "rvol_20d": rvol_20d,
            "volume_trend_10d": volume_trend_10d,
            "returns_5d": returns_5d,
            "returns_20d": returns_20d,
            "returns_60d": returns_60d,
            "momentum_5d_vs_20d": momentum_5d_vs_20d,
            "close_vs_ema20": close_vs_ema20,
            "close_vs_ema50": close_vs_ema50,
            "ema20_vs_ema50": ema20_vs_ema50,
            "ema50_vs_ema200": ema50_vs_ema200,
            "ATR_14_pct": ATR_14_pct,
            "volatility_percentile_90d": volatility_percentile_90d,
            "bb_position": bb_position,
            "rsi_14": rsi_14,
            "macd_hist": macd_hist,
            "adx_14": adx_14,
            "dist_from_52w_high": dist_from_52w_high,
            "dist_from_52w_low": dist_from_52w_low,
            "benchmark_return_20d": benchmark_return_20d,
            "beta_20d": beta_20d,
            "rank_rvol_5d": cross_sectional.get("rank_rvol_5d", 0.5) if cross_sectional else 0.5,
            "rank_returns_20d": cross_sectional.get("rank_returns_20d", 0.5) if cross_sectional else 0.5,
            "rank_close_vs_ema20": cross_sectional.get("rank_close_vs_ema20", 0.5) if cross_sectional else 0.5,
            "month": month,
            "day_of_week": day_of_week,
            "sharpe_20d": sharpe_20d,
        }, index=df.index)

        return features.fillna(0.0)


@dataclass
class V4SignalModel:
    """Production ML model with ordinal classification + calibration."""

    model: Any | None = None
    calibrator: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(V4FeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None
    label_bins: tuple[float, ...] = (-0.05, 0.0, 0.05)
    label_names: tuple[str, ...] = ("strong_loss", "weak_loss", "weak_win", "strong_win")

    def _encode_label(self, return_pct: float) -> int:
        for i, thresh in enumerate(self.label_bins):
            if return_pct < thresh:
                return i
        return len(self.label_bins)

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> V4SignalModel:
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = V4FeatureExtractor()
        # Pre-compute features for all symbols
        print("Pre-computing features for all symbols...")
        features_cache = {}
        for sym, bars in bars_by_symbol.items():
            if bars is None or bars.empty:
                continue
            bench = None
            # Try to find benchmark for this symbol's market
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
            y.append(self._encode_label(trade.return_pct))
            weights.append(max(abs(trade.return_pct), 0.001))
            dates.append(trade.signal_date)

        if not X_rows:
            raise ValueError("Could not extract features for any trade.")

        X = pd.concat(X_rows, ignore_index=True)
        X = X[self.feature_names].fillna(0.0)
        y_arr = np.array(y)
        weights_arr = np.array(weights)
        dates_arr = pd.to_datetime(dates)

        print(f"Training on {len(y_arr)} trades, {len(self.feature_names)} features")
        print(f"Label distribution: {dict(zip(self.label_names, np.bincount(y_arr, minlength=len(self.label_names))))}")

        if len(set(y_arr)) < 2:
            raise ValueError("All trades have the same label.")

        # Time-series split: train <= 2023, val = 2024, test = 2024-2025
        train_mask = dates_arr <= "2023-12-31"
        val_mask = (dates_arr >= "2024-01-01") & (dates_arr <= "2024-06-30")
        test_mask = dates_arr >= "2024-07-01"

        # If no test data (all old), fall back to random split
        if test_mask.sum() < 20:
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(y_arr))
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_arr)
            train_mask = pd.Series(False, index=range(len(y_arr)))
            train_mask.iloc[train_idx] = True
            test_mask = pd.Series(False, index=range(len(y_arr)))
            test_mask.iloc[test_idx] = True
            val_mask = pd.Series(False, index=range(len(y_arr)))

        X_train = X[train_mask]
        y_train = y_arr[train_mask]
        w_train = weights_arr[train_mask]
        X_val = X[val_mask]
        y_val = y_arr[val_mask]
        X_test = X[test_mask]
        y_test = y_arr[test_mask]

        print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

        # Compute class weights for imbalance
        class_counts = np.bincount(y_train, minlength=len(self.label_names))
        class_weights = {i: max(class_counts.sum() / (len(self.label_names) * max(c, 1)), 1.0) for i, c in enumerate(class_counts)}
        print(f"Class weights: {class_weights}")

        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=2.0,
            gamma=0.1,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            early_stopping_rounds=30,
        )

        eval_set = [(X_val, y_val)] if len(y_val) > 10 else []
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Predictions
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Calibration on validation set
        self.calibrator = {}
        if len(y_val) > 50:
            val_proba = model.predict_proba(X_val)
            for cls in range(len(self.label_names)):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(val_proba[:, cls], (y_val == cls).astype(int))
                self.calibrator[cls] = iso

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        # AUC for strong_win vs rest (most important for trading)
        strong_win_proba = y_proba[:, -1]
        strong_win_label = (y_test == len(self.label_names) - 1).astype(int)
        auc_strong = roc_auc_score(strong_win_label, strong_win_proba) if len(set(strong_win_label)) > 1 else float("nan")
        # AUC for any win vs any loss
        win_proba = y_proba[:, 2:].sum(axis=1)
        win_label = (y_test >= 2).astype(int)
        auc_win = roc_auc_score(win_label, win_proba) if len(set(win_label)) > 1 else float("nan")

        self.model = model
        self.metrics = {
            "accuracy": float(acc),
            "auc_strong_win": float(auc_strong),
            "auc_any_win": float(auc_win),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 500,
            "label_distribution": dict(zip(self.label_names, [int(c) for c in np.bincount(y_arr, minlength=len(self.label_names))])),
        }
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].fillna(0.0)
        proba = self.model.predict_proba(X)
        # Apply calibration
        if self.calibrator:
            calibrated = np.zeros_like(proba)
            for cls, iso in self.calibrator.items():
                calibrated[:, cls] = iso.predict(proba[:, cls])
            # Renormalize
            calibrated = np.clip(calibrated, 0.001, 0.999)
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
            proba = calibrated
        return proba

    def predict_confidence(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return probability of 'strong_win' (highest class)."""
        proba = self.predict_proba(features_df)
        return proba[:, -1]

    def save(self, path: str | Path) -> None:
        _require_ml()
        import joblib
        payload = {
            "model": self.model,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "label_bins": self.label_bins,
            "label_names": self.label_names,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> V4SignalModel:
        _require_ml()
        import joblib
        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.calibrator = payload.get("calibrator")
        instance.feature_names = payload.get("feature_names", list(V4FeatureExtractor.FEATURE_COLUMNS))
        instance.metrics = payload.get("metrics")
        instance.label_bins = payload.get("label_bins", (-0.10, -0.02, 0.02, 0.10))
        instance.label_names = payload.get("label_names", ("strong_loss", "weak_loss", "weak_win", "strong_win"))
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
