"""Market regime detection using Hurst, volatility, trend, and stress."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

VolRegime = Literal["LOW", "NORMAL", "HIGH", "EXTREME", "UNKNOWN"]
TrendRegime = Literal["UPTREND", "DOWNTREND", "CHOP", "UNKNOWN"]


@dataclass(frozen=True)
class Regime:
    hurst: float
    vol_regime: VolRegime
    trend_regime: TrendRegime
    stress: float
    is_tradeable: bool


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


class RegimeDetector:
    """Classify market regime from price and benchmark data."""

    @staticmethod
    def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
        """Estimate Hurst exponent via the Rescaled Range (R/S) method.

        Returns a float where:
            H < 0.5  → mean-reverting
            H ≈ 0.5  → random walk
            H > 0.5  → trending
        """
        prices = pd.to_numeric(prices, errors="coerce").dropna()
        if len(prices) < max(max_lag * 2, 20):
            return 0.5

        returns = np.diff(np.log(prices.to_numpy()))
        if np.std(returns) == 0 or not np.isfinite(np.std(returns)):
            return 0.5

        lags = range(10, min(max_lag, len(returns) // 4) + 1, 10)
        if len(lags) < 2:
            lags = range(2, min(max_lag, len(returns) // 2) + 1, max(1, (len(returns) // 2 - 2) // 10))
        if len(lags) < 2:
            return 0.5

        rs_values: list[float] = []
        lag_values: list[float] = []

        for lag in lags:
            n = len(returns) // lag * lag
            if n < lag * 2:
                continue
            chunks = returns[:n].reshape(n // lag, lag)
            means = chunks.mean(axis=1, keepdims=True)
            dev = np.cumsum(chunks - means, axis=1)
            ranges = dev.max(axis=1) - dev.min(axis=1)
            stds = chunks.std(axis=1, ddof=1)
            stds[stds == 0] = np.nan
            rs = ranges / stds
            rs_mean = float(np.nanmean(rs))
            if np.isfinite(rs_mean) and rs_mean > 0:
                rs_values.append(np.log(rs_mean))
                lag_values.append(np.log(lag))

        if len(lag_values) < 2:
            return 0.5

        coeffs = np.polyfit(lag_values, rs_values, 1)
        return float(coeffs[0])

    @staticmethod
    def volatility_regime(
        returns: pd.Series,
        window: int = 20,
        lookback: int = 252,
    ) -> VolRegime:
        """Classify current volatility relative to its historical distribution."""
        returns = pd.to_numeric(returns, errors="coerce").dropna()
        if len(returns) < window + 1:
            return "UNKNOWN"

        realized = (
            returns.rolling(window=window, min_periods=window)
            .std()
            .dropna()
        )
        if len(realized) < 2:
            return "UNKNOWN"

        current = float(realized.iloc[-1])
        hist = realized.iloc[-lookback:].dropna()
        if len(hist) < 10:
            return "UNKNOWN"

        p = float(hist.rank(pct=True).iloc[-1])
        if p > 0.90:
            return "EXTREME"
        if p > 0.70:
            return "HIGH"
        if p < 0.30:
            return "LOW"
        return "NORMAL"

    @staticmethod
    def trend_regime(
        prices: pd.Series,
        fast: int = 50,
        slow: int = 200,
    ) -> TrendRegime:
        """Classify trend based on price vs fast and slow EMAs."""
        prices = pd.to_numeric(prices, errors="coerce").dropna()
        min_bars = max(fast, slow) + 5
        if len(prices) < min_bars:
            return "UNKNOWN"

        ema_fast = _ema(prices, fast)
        ema_slow = _ema(prices, slow)
        price = float(prices.iloc[-1])
        fast_val = float(ema_fast.iloc[-1])
        slow_val = float(ema_slow.iloc[-1])

        if price > fast_val > slow_val:
            return "UPTREND"
        if price < fast_val < slow_val:
            return "DOWNTREND"
        return "CHOP"

    @staticmethod
    def stress_index(
        benchmark_bars: pd.DataFrame,
        vix: Optional[pd.Series] = None,
    ) -> float:
        """Combine volatility regime, drawdown, and breadth into a 0–1 score."""
        if benchmark_bars is None or benchmark_bars.empty:
            return 0.0

        df = benchmark_bars.copy()
        needed = {"close"}
        if not needed.issubset(df.columns):
            return 0.0

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 30:
            return 0.0

        returns = close.pct_change().dropna()
        vol_regime = RegimeDetector.volatility_regime(returns, window=20, lookback=252)
        vol_score: float = {"LOW": 0.0, "NORMAL": 0.3, "HIGH": 0.7, "EXTREME": 1.0, "UNKNOWN": 0.5}.get(vol_regime, 0.5)

        peak = close.cummax()
        drawdown = (peak - close) / peak
        dd_score = float(drawdown.iloc[-1])
        if not np.isfinite(dd_score):
            dd_score = 0.0

        trend = RegimeDetector.trend_regime(close, fast=50, slow=200)
        breadth_score: float = {"UPTREND": 0.0, "CHOP": 0.5, "DOWNTREND": 1.0, "UNKNOWN": 0.5}.get(trend, 0.5)

        if vix is not None and not vix.empty:
            vix_val = float(pd.to_numeric(vix, errors="coerce").dropna().iloc[-1])
            if np.isfinite(vix_val):
                vix_score = min(vix_val / 40.0, 1.0)
                return float(np.mean([vol_score, dd_score, breadth_score, vix_score]))

        return float(np.mean([vol_score, dd_score, breadth_score]))

    @classmethod
    def classify(
        cls,
        bars: pd.DataFrame,
        benchmark: pd.DataFrame,
    ) -> Regime:
        """Return a composite regime classification for the given bars."""
        if bars is None or bars.empty or benchmark is None or benchmark.empty:
            return Regime(
                hurst=0.5,
                vol_regime="UNKNOWN",
                trend_regime="UNKNOWN",
                stress=0.0,
                is_tradeable=False,
            )

        needed = {"close"}
        if not needed.issubset(bars.columns):
            return Regime(
                hurst=0.5,
                vol_regime="UNKNOWN",
                trend_regime="UNKNOWN",
                stress=0.0,
                is_tradeable=False,
            )

        close = pd.to_numeric(bars["close"], errors="coerce").dropna()
        if len(close) < 30:
            return Regime(
                hurst=0.5,
                vol_regime="UNKNOWN",
                trend_regime="UNKNOWN",
                stress=0.0,
                is_tradeable=False,
            )

        hurst = cls.hurst_exponent(close)
        returns = close.pct_change().dropna()
        vol_regime = cls.volatility_regime(returns)
        trend_regime = cls.trend_regime(close)
        stress = cls.stress_index(benchmark)

        is_tradeable = (
            stress < 0.7
            and vol_regime != "EXTREME"
            and trend_regime != "DOWNTREND"
        )

        return Regime(
            hurst=round(hurst, 4),
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            stress=round(stress, 4),
            is_tradeable=is_tradeable,
        )

    @classmethod
    def classify_series(
        cls,
        bars: pd.DataFrame,
        benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a daily regime history DataFrame."""
        if bars is None or bars.empty or benchmark is None or benchmark.empty:
            return pd.DataFrame()

        needed = {"close"}
        if not needed.issubset(bars.columns) or not needed.issubset(benchmark.columns):
            return pd.DataFrame()

        close = pd.to_numeric(bars["close"], errors="coerce").dropna()
        bench_close = pd.to_numeric(benchmark["close"], errors="coerce").dropna()
        if len(close) < 30 or len(bench_close) < 30:
            return pd.DataFrame()

        idx = close.index.intersection(bench_close.index)
        if len(idx) < 30:
            return pd.DataFrame()

        records: list[dict] = []
        for i in range(30, len(idx) + 1):
            day = idx[i - 1]
            sub = bars.loc[bars.index <= day]
            bench_sub = benchmark.loc[benchmark.index <= day]
            regime = cls.classify(sub, bench_sub)
            records.append(
                {
                    "date": day,
                    "hurst": regime.hurst,
                    "vol_regime": regime.vol_regime,
                    "trend_regime": regime.trend_regime,
                    "stress": regime.stress,
                    "is_tradeable": regime.is_tradeable,
                }
            )

        return pd.DataFrame(records).set_index("date")


def stress_position_multiplier(stress: float) -> float:
    """Return a position-size multiplier inversely related to market stress."""
    if stress >= 0.9:
        return 0.25
    if stress >= 0.7:
        return 0.5
    if stress >= 0.5:
        return 0.75
    return 1.0
