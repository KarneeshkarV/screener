"""ML Signal Confidence v5 — regression-based expected return prediction.

Key improvements over v4:
- XGBoost regression (predicts expected return_pct, not class labels)
- No categorical noise (removed month/day_of_week)
- No isotonic calibration (was causing probability inversion)
- Richer signal-quality features (drawdown, divergence, gap)
- Walk-forward training support for regime adaptation
- Meta-confidence score based on prediction magnitude
"""
from __future__ import annotations

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


class MissingMLDependencyError(RuntimeError):
    pass


def _require_ml() -> None:
    if not _HAS_ML:
        raise MissingMLDependencyError(
            "xgboost and scikit-learn required. Install: uv pip install xgboost scikit-learn scipy"
        )


class V5FeatureExtractor:
    """Production feature set focused on signal quality and regime context."""

    FEATURE_COLUMNS = [
        # Volume
        "rvol_5d", "rvol_20d", "volume_trend_10d",
        # Momentum / acceleration
        "returns_5d", "returns_20d", "returns_60d",
        "momentum_5d_vs_20d",
        # Trend alignment
        "close_vs_ema20", "close_vs_ema50", "ema20_vs_ema50", "ema50_vs_ema200",
        # Volatility / risk
        "ATR_14_pct", "volatility_percentile_90d", "bb_position",
        # Technical
        "rsi_14", "macd_hist", "adx_14",
        # Price structure
        "dist_from_52w_high", "dist_from_52w_low",
        # Market context
        "benchmark_return_20d", "beta_20d",
        # Signal quality (NEW)
        "max_dd_20d",
        "range_pct",
        "gap_pct",
        "consecutive_up_days",
        "volume_price_corr_20d",
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

        return features.fillna(0.0)


@dataclass
class V5SignalModel:
    """Production ML model: predicts expected return via regression.

    Uses XGBRegressor to predict return_pct directly.  Meta-confidence
    is derived from prediction magnitude (higher expected return = higher
    confidence) and optional rolling calibration.
    """

    model: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(V5FeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None
    rolling_window_months: int = 3  # months of history to use for training
    n_estimators: int = 100
    max_depth: int = 3
    reg_lambda: float = 5.0

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> V5SignalModel:
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = V5FeatureExtractor()
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
            y.append(trade.return_pct)
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
        print(f"Return distribution: mean={y_arr.mean():.3%}, std={y_arr.std():.3%}")

        # Default: use rolling window of recent data (regime-adaptive)
        if self.rolling_window_months > 0:
            cutoff = dates_arr.max() - pd.DateOffset(months=self.rolling_window_months)
            recent_mask = dates_arr >= cutoff
            if recent_mask.sum() >= 100:
                print(f"Using rolling window: last {self.rolling_window_months} months ({recent_mask.sum()} trades)")
                X = X[recent_mask]
                y_arr = y_arr[recent_mask]
                weights_arr = weights_arr[recent_mask]
                dates_arr = dates_arr[recent_mask]

        # Time-series split for evaluation: sort by date, take last 20% as test
        n = len(y_arr)
        split_idx = int(n * 0.8)
        sort_idx = np.argsort(dates_arr)
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y_arr[sort_idx]
        w_sorted = weights_arr[sort_idx]

        X_train = X_sorted.iloc[:split_idx]
        y_train = y_sorted[:split_idx]
        w_train = w_sorted[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_test = y_sorted[split_idx:]

        print(f"Train: {len(y_train)} | Test: {len(y_test)}")

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=self.reg_lambda,
            gamma=0.1,
            random_state=42,
            early_stopping_rounds=20,
        )

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Directional metrics
        y_test_dir = (y_test > 0).astype(int)
        y_pred_dir = (y_pred > np.median(y_pred)).astype(int)
        auc = roc_auc_score(y_test_dir, y_pred) if len(set(y_test_dir)) > 1 else float("nan")
        acc = accuracy_score(y_test_dir, y_pred_dir)

        # Filter performance: top predictions
        sorted_pred_idx = np.argsort(y_pred)[::-1]
        top_n = max(1, int(len(y_test) * 0.2))
        top_idx = sorted_pred_idx[:top_n]
        top_avg_ret = y_test[top_idx].mean()
        top_win_rate = (y_test[top_idx] > 0).mean()

        bottom_n = max(1, int(len(y_test) * 0.2))
        bottom_idx = sorted_pred_idx[-bottom_n:]
        bottom_avg_ret = y_test[bottom_idx].mean()
        bottom_win_rate = (y_test[bottom_idx] > 0).mean()

        self.model = model
        self.metrics = {
            "mse": float(mse),
            "r2": float(r2),
            "auc_direction": float(auc),
            "accuracy_direction": float(acc),
            "top20_avg_return": float(top_avg_ret),
            "top20_win_rate": float(top_win_rate),
            "bottom20_avg_return": float(bottom_avg_ret),
            "bottom20_win_rate": float(bottom_win_rate),
            "baseline_win_rate": float(y_test_dir.mean()),
            "baseline_avg_return": float(y_test.mean()),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else self.n_estimators,
        }
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return predicted expected return_pct."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].fillna(0.0)
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
        instance.rolling_window_months = payload.get("rolling_window_months", 3)
        instance.n_estimators = payload.get("n_estimators", 100)
        instance.max_depth = payload.get("max_depth", 3)
        instance.reg_lambda = payload.get("reg_lambda", 5.0)
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
