"""Category 1: trend direction and strength.

Everything here answers "which way, and how hard". Nothing here judges whether
the trend is clean; that is :mod:`.quality`.

Slopes are measured on log price so they are per-day fractional drift and
comparable across price levels. Where a raw distance is more natural (channel
position, distance from a high) the output is already a ratio and needs no
further normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.research.features.base import (
    FeatureCtx,
    feature,
    rolling_ols_slope,
    safe_ratio,
)


@feature(
    "ema_slope",
    category="trend",
    doc="Per-day fractional slope of an EMA, measured over its own span.",
    params={"span": 50, "lookback": 20},
    grid=(
        {"span": 20, "lookback": 10},
        {"span": 100, "lookback": 20},
        {"span": 200, "lookback": 40},
    ),
    min_lookback=220,
)
def ema_slope(ctx: FeatureCtx, *, span: int, lookback: int) -> pd.Series:
    # Normalized by the EMA level, so this is a growth rate per day rather than
    # a rupee-per-day slope that would scale with price.
    ema = ctx.close.ewm(span=span, adjust=False, min_periods=span).mean()
    return safe_ratio(ema - ema.shift(lookback), ema.shift(lookback) * float(lookback))


@feature(
    "logprice_slope",
    category="trend",
    doc="Trailing OLS slope of log price: average daily drift over the window.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def logprice_slope(ctx: FeatureCtx, *, window: int) -> pd.Series:
    slope, _, _ = rolling_ols_slope(ctx.log_close, window)
    return slope


@feature(
    "logprice_slope_t",
    category="trend",
    doc="t-statistic of the log-price regression slope: drift per unit of noise.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def logprice_slope_t(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # The t-stat is the direction and the quality in one number, which makes it
    # the natural head-to-head against efficiency ratio and R-squared.
    _, t_stat, _ = rolling_ols_slope(ctx.log_close, window, with_stats=True)
    return t_stat


@feature(
    "trend_r2",
    category="trend",
    doc="R-squared of the trailing log-price regression: how linear the move is.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def trend_r2(ctx: FeatureCtx, *, window: int) -> pd.Series:
    _, _, r_squared = rolling_ols_slope(ctx.log_close, window, with_stats=True)
    return r_squared


@feature(
    "momentum",
    category="trend",
    doc="Total return over a lookback, optionally skipping the last `skip` bars.",
    params={"window": 120, "skip": 0},
    grid=(
        {"window": 20, "skip": 0},
        {"window": 60, "skip": 0},
        {"window": 250, "skip": 0},
        {"window": 250, "skip": 21},
    ),
    min_lookback=275,
)
def momentum(ctx: FeatureCtx, *, window: int, skip: int) -> pd.Series:
    # skip=21 is the classic 12-1 construction: drop the most recent month to
    # step around short-term reversal.
    close = ctx.close
    recent = close.shift(skip)
    past = close.shift(window)
    return safe_ratio(recent, past) - 1.0


@feature(
    "momentum_consistency",
    category="trend",
    doc="Fraction of 20/60/120/250-day momenta that are positive, in [0, 1].",
    params={"windows": (20, 60, 120, 250)},
    grid=({"windows": (60, 120, 250)}, {"windows": (10, 20, 60, 120, 250)}),
    min_lookback=250,
)
def momentum_consistency(ctx: FeatureCtx, *, windows: tuple[int, ...]) -> pd.Series:
    # Multi-horizon agreement. A name trending on every horizon scores 1.0; one
    # whose short and long horizons disagree scores near 0.5.
    close = ctx.close
    scores = [
        (safe_ratio(close, close.shift(w)) - 1.0 > 0.0)
        .astype(float)
        .where(close.shift(w).notna())
        for w in windows
    ]
    return pd.concat(scores, axis=1).mean(axis=1)


@feature(
    "fast_slow_momentum",
    category="trend",
    doc="Fast-horizon return minus slow-horizon return, per day of horizon.",
    params={"fast": 20, "slow": 120},
    grid=({"fast": 10, "slow": 60}, {"fast": 60, "slow": 250}),
    min_lookback=250,
)
def fast_slow_momentum(ctx: FeatureCtx, *, fast: int, slow: int) -> pd.Series:
    # Per-day so the two horizons are on the same scale; positive means the
    # recent leg is running hotter than the long leg.
    close = ctx.close
    fast_leg = (safe_ratio(close, close.shift(fast)) - 1.0) / float(fast)
    slow_leg = (safe_ratio(close, close.shift(slow)) - 1.0) / float(slow)
    return fast_leg - slow_leg


@feature(
    "ema_distance_vol",
    category="trend",
    doc="Distance of close above an EMA, in units of trailing daily volatility.",
    params={"span": 50, "vol_window": 60},
    grid=(
        {"span": 20, "vol_window": 60},
        {"span": 200, "vol_window": 60},
        {"span": 50, "vol_window": 120},
    ),
    min_lookback=220,
)
def ema_distance_vol(ctx: FeatureCtx, *, span: int, vol_window: int) -> pd.Series:
    # Volatility-normalized extension. A 5% gap means something very different
    # on a 1%-a-day name than on a 4%-a-day name, and this collapses the two.
    ema = ctx.close.ewm(span=span, adjust=False, min_periods=span).mean()
    gap = safe_ratio(ctx.close - ema, ema)
    vol = ctx.log_returns.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    return (gap / vol).where(vol > 0.0)


@feature(
    "channel_position",
    category="trend",
    doc="Where the close sits in its Donchian channel: 0 at the low, 1 at the high.",
    params={"window": 120},
    grid=({"window": 20}, {"window": 60}, {"window": 250}),
    min_lookback=250,
)
def channel_position(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # The channel includes the current bar, which is known at t. A flat channel
    # (high == low) yields NaN rather than a spurious 0 or 1.
    high = ctx.bars["high"].astype(float).rolling(window, min_periods=window).max()
    low = ctx.bars["low"].astype(float).rolling(window, min_periods=window).min()
    span = high - low
    return ((ctx.close - low) / span).where(span > 0.0)


@feature(
    "distance_from_high",
    category="trend",
    doc="Close divided by the trailing period high, in [0, 1]. 1.0 is a new high.",
    params={"window": 250},
    grid=({"window": 60}, {"window": 120}, {"window": 500}),
    min_lookback=500,
)
def distance_from_high(ctx: FeatureCtx, *, window: int) -> pd.Series:
    high = ctx.close.rolling(window, min_periods=window).max()
    return safe_ratio(ctx.close, high)


@feature(
    "new_high_recency",
    category="trend",
    doc="Bars since the trailing-window high, scaled to [0, 1]. 1.0 = high is today.",
    params={"window": 250},
    grid=({"window": 60}, {"window": 120}),
    min_lookback=250,
)
def new_high_recency(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # argmax over a trailing window: how fresh the leadership is, which
    # separates "at its high" from "was at its high eleven months ago".
    def recency(values: np.ndarray) -> float:
        return float(np.argmax(values)) / float(len(values) - 1)

    return ctx.close.rolling(window, min_periods=window).apply(recency, raw=True)
