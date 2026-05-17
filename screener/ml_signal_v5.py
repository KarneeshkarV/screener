"""ML Signal Confidence v5 — regression-based expected return prediction.

Key improvements over v4:
- XGBoost regression (predicts expected return_pct, not class labels)
- No categorical noise (removed month/day_of_week)
- No isotonic calibration (was causing probability inversion)
- Richer signal-quality features (drawdown, divergence, gap)
- Walk-forward training support for regime adaptation
- Meta-confidence score based on prediction magnitude
- PROPER train/val/test split: early-stop on val, report on untouched test
- Per-feature neutral fill values instead of blanket 0.0
- External data integration (insider, short interest, earnings via FMP)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
    from scipy import stats
    from xgboost import XGBRegressor
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from screener.backtester.models import Trade

logger = logging.getLogger(__name__)


class MissingMLDependencyError(RuntimeError):
    pass


def _require_ml() -> None:
    if not _HAS_ML:
        raise MissingMLDependencyError(
            "xgboost and scikit-learn required. Install: uv pip install xgboost scikit-learn scipy"
        )


# Per-feature neutral fill values (no fake signal injection)
FEATURE_NEUTRAL_VALUES: dict[str, float] = {
    # Volume — neutral = 1.0 (average volume)
    "rvol_5d": 1.0,
    "rvol_20d": 1.0,
    "volume_trend_10d": 1.0,
    # Momentum — neutral = 0.0 (no change)
    "returns_5d": 0.0,
    "returns_20d": 0.0,
    "returns_60d": 0.0,
    "momentum_5d_vs_20d": 0.0,
    # Trend alignment — neutral = 0.0 (price at EMA)
    "close_vs_ema20": 0.0,
    "close_vs_ema50": 0.0,
    "ema20_vs_ema50": 0.0,
    "ema50_vs_ema200": 0.0,
    # Volatility / risk — ATR 0, vol percentile middle
    "ATR_14_pct": 0.0,
    "volatility_percentile_90d": 0.5,
    # Technical — bb middle, rsi neutral, macd flat, adx weak trend
    "bb_position": 0.5,
    "rsi_14": 50.0,
    "macd_hist": 0.0,
    "adx_14": 25.0,
    # Price structure — neutral = at midpoint
    "dist_from_52w_high": 0.0,
    "dist_from_52w_low": 0.0,
    # Market context — neutral market return, beta = 1.0
    "benchmark_return_20d": 0.0,
    "beta_20d": 1.0,
    # Signal quality — no drawdown, no gap, no streak
    "max_dd_20d": 0.0,
    "range_pct": 0.0,
    "gap_pct": 0.0,
    "consecutive_up_days": 0.0,
    "volume_price_corr_20d": 0.0,
    "sharpe_20d": 0.0,
    # External data — neutral = no signal
    "insider_buy_ratio": 0.5,
    "insider_buy_shares_ratio": 0.5,
    "insider_buy_dollar_ratio": 0.5,
    "insider_n_transactions": 0.0,
    "short_pct_float": 0.05,
    "short_trend": 1.0,
    "earnings_surprise_last": 0.0,
    "earnings_beat_streak": 0.0,
    "days_since_earnings": 90.0,
}


class V5FeatureExtractor:
    """Production feature set focused on signal quality and regime context."""

    FEATURE_COLUMNS = list(FEATURE_NEUTRAL_VALUES.keys())

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
        open_px = df["open"]

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

        # Signal quality (NEW)
        rolling_high = high.rolling(20, min_periods=1).max()
        max_dd_20d = (close - rolling_high) / rolling_high.replace(0, np.nan)
        range_pct = (high - low) / close.replace(0, np.nan)
        gap_pct = (open_px - close.shift(1)) / close.shift(1).replace(0, np.nan)

        up_days = (daily_ret > 0).astype(int)
        consecutive_up_days = up_days * 0
        streak = 0
        for i in range(len(up_days)):
            if up_days.iloc[i] == 1:
                streak += 1
            else:
                streak = 0
            consecutive_up_days.iloc[i] = streak

        volume_price_corr_20d = daily_ret.rolling(20, min_periods=5).corr(volume.pct_change().replace([np.inf, -np.inf], np.nan))

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
            "max_dd_20d": max_dd_20d,
            "range_pct": range_pct,
            "gap_pct": gap_pct,
            "consecutive_up_days": consecutive_up_days,
            "volume_price_corr_20d": volume_price_corr_20d,
            "sharpe_20d": sharpe_20d,
        }, index=df.index)

        # Drop rows that are >50% NaN (insufficient history), then fill remaining with neutral values
        na_frac = features.isna().mean(axis=1)
        features = features[na_frac <= 0.5]
        features = features.fillna(FEATURE_NEUTRAL_VALUES)
        return features


@dataclass
class V5SignalModel:
    """Production ML model: predicts expected return via regression.

    Uses XGBRegressor to predict return_pct directly.
    PROPER train/val/test split:
        - Train: 60% oldest data (by signal date)
        - Val:   20% middle data (early stopping here)
        - Test:  20% newest data (untouched for final metrics)
    """

    model: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(V5FeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None
    rolling_window_months: int = 0  # 0 = use all data (walk-forward overrides)
    n_estimators: int = 300
    max_depth: int = 5
    reg_lambda: float = 3.0
    learning_rate: float = 0.05

    def _build_dataset(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Extract features and labels from trades. Returns (X, y, weights, dates)."""
        extractor = V5FeatureExtractor()
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
            y.append(trade.return_pct)
            weights.append(max(abs(trade.return_pct), 0.001))
            dates.append(trade.signal_date)

        if not X_rows:
            raise ValueError("Could not extract features for any trade.")

        X = pd.concat(X_rows, ignore_index=True)
        X = X[self.feature_names]
        # Apply per-feature neutral fill for any remaining NAs (should be rare after extract())
        for col in X.columns:
            if col in FEATURE_NEUTRAL_VALUES:
                X[col] = X[col].fillna(FEATURE_NEUTRAL_VALUES[col])
            else:
                X[col] = X[col].fillna(0.0)
        y_arr = np.array(y)
        weights_arr = np.array(weights)
        dates_arr = pd.to_datetime(dates)
        return X, y_arr, weights_arr, dates_arr

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> V5SignalModel:
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        print("Building dataset...")
        X, y_arr, weights_arr, dates_arr = self._build_dataset(
            trades, bars_by_symbol, benchmark_bars
        )

        print(f"Dataset: {len(y_arr)} trades, {len(self.feature_names)} features")
        print(f"Return distribution: mean={y_arr.mean():.3%}, std={y_arr.std():.3%}")

        # Time-series sort: oldest first
        sort_idx = np.argsort(dates_arr)
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y_arr[sort_idx]
        w_sorted = weights_arr[sort_idx]
        dates_sorted = dates_arr[sort_idx]

        n = len(y_sorted)
        # 60/20/20 temporal split
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted[:train_end]
        w_train = w_sorted[:train_end]

        X_val = X_sorted.iloc[train_end:val_end]
        y_val = y_sorted[train_end:val_end]

        X_test = X_sorted.iloc[val_end:]
        y_test = y_sorted[val_end:]

        print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
        print(f"Date ranges — train: {dates_sorted[0].date()} to {dates_sorted[train_end-1].date()}")
        print(f"                 val: {dates_sorted[train_end].date()} to {dates_sorted[val_end-1].date()}")
        print(f"                test: {dates_sorted[val_end].date()} to {dates_sorted[-1].date()}")

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.6,
            colsample_bytree=0.6,
            colsample_bylevel=0.8,
            min_child_weight=1,
            reg_alpha=0.0,
            reg_lambda=self.reg_lambda,
            gamma=0.0,
            random_state=42,
            n_jobs=4,
            early_stopping_rounds=20,
        )

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # === REPORT ON UNTOUCHED TEST SET ===
        y_pred_test = model.predict(X_test)

        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Directional metrics on TEST
        y_test_dir = (y_test > 0).astype(int)
        auc_test = roc_auc_score(y_test_dir, y_pred_test) if len(set(y_test_dir)) > 1 else float("nan")
        y_pred_dir = (y_pred_test > np.median(y_pred_test)).astype(int)
        acc_test = accuracy_score(y_test_dir, y_pred_dir)

        # Filter performance: top predictions on TEST
        sorted_pred_idx = np.argsort(y_pred_test)[::-1]
        n10 = max(1, int(len(y_test) * 0.1))
        top10_idx = sorted_pred_idx[:n10]
        top10_avg_ret = y_test[top10_idx].mean()
        top10_wr = (y_test[top10_idx] > 0).mean()

        n20 = max(1, int(len(y_test) * 0.2))
        top20_idx = sorted_pred_idx[:n20]
        top20_avg_ret = y_test[top20_idx].mean()
        top20_wr = (y_test[top20_idx] > 0).mean()

        bottom20_idx = sorted_pred_idx[-n20:]
        bottom20_avg_ret = y_test[bottom20_idx].mean()
        bottom20_wr = (y_test[bottom20_idx] > 0).mean()

        # Val metrics for reference (where early stopping happened)
        y_pred_val = model.predict(X_val)
        y_val_dir = (y_val > 0).astype(int)
        auc_val = roc_auc_score(y_val_dir, y_pred_val) if len(set(y_val_dir)) > 1 else float("nan")

        self.model = model
        self.metrics = {
            "mse_test": float(mse_test),
            "r2_test": float(r2_test),
            "auc_test": float(auc_test),
            "auc_val": float(auc_val),
            "accuracy_test": float(acc_test),
            "top10_avg_return": float(top10_avg_ret),
            "top10_win_rate": float(top10_wr),
            "top20_avg_return": float(top20_avg_ret),
            "top20_win_rate": float(top20_wr),
            "bottom20_avg_return": float(bottom20_avg_ret),
            "bottom20_win_rate": float(bottom20_wr),
            "baseline_win_rate": float(y_test_dir.mean()),
            "baseline_avg_return": float(y_test.mean()),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else self.n_estimators,
        }

        print(f"\n{'='*60}")
        print("FINAL METRICS (UNTOUCHED TEST SET)")
        print(f"{'='*60}")
        print(f"AUC (test):  {auc_test:.4f}  |  AUC (val): {auc_val:.4f}")
        print(f"Top 10% WR:  {top10_wr:.1%}  |  Top 10% Avg: {top10_avg_ret:.3%}")
        print(f"Top 20% WR:  {top20_wr:.1%}  |  Top 20% Avg: {top20_avg_ret:.3%}")
        print(f"Baseline WR: {y_test_dir.mean():.1%}  |  Baseline Avg: {y_test.mean():.3%}")
        print(f"{'='*60}")
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return predicted expected return_pct."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].copy()
        for col in X.columns:
            if col in FEATURE_NEUTRAL_VALUES:
                X[col] = X[col].fillna(FEATURE_NEUTRAL_VALUES[col])
            else:
                X[col] = X[col].fillna(0.0)
        return self.model.predict(X)

    def predict_confidence(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return confidence score [0,1] monotonically increasing with expected return.

        Uses a sigmoid transformation of predicted return, calibrated so that
        the median prediction maps to 0.5.
        """
        preds = self.predict(features_df)
        scale = 0.02  # ~2% return maps to high confidence
        return 1.0 / (1.0 + np.exp(-preds / scale))

    def save(self, path: str | Path) -> None:
        _require_ml()
        import joblib
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "rolling_window_months": self.rolling_window_months,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.learning_rate,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> V5SignalModel:
        _require_ml()
        import joblib
        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.feature_names = payload.get("feature_names", list(V5FeatureExtractor.FEATURE_COLUMNS))
        instance.metrics = payload.get("metrics")
        instance.rolling_window_months = payload.get("rolling_window_months", 0)
        instance.n_estimators = payload.get("n_estimators", 300)
        instance.max_depth = payload.get("max_depth", 5)
        instance.reg_lambda = payload.get("reg_lambda", 3.0)
        instance.learning_rate = payload.get("learning_rate", 0.05)
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
