"""Category 4: trend acceleration and deterioration.

The case these exist for: the long trend is still up, but the short leg has
started rolling over. A level feature cannot see that; a change in a level can.
"""

from __future__ import annotations

import pandas as pd

from screener.research.features.base import (
    FeatureCtx,
    feature,
    rolling_ols_slope,
    safe_ratio,
)
from screener.research.features.trend import ema_slope


@feature(
    "slope_change",
    category="acceleration",
    doc="Change in the log-price regression slope over the last `lag` bars.",
    params={"window": 60, "lag": 20},
    grid=(
        {"window": 20, "lag": 10},
        {"window": 120, "lag": 20},
        {"window": 120, "lag": 60},
    ),
    min_lookback=200,
)
def slope_change(ctx: FeatureCtx, *, window: int, lag: int) -> pd.Series:
    slope, _, _ = rolling_ols_slope(ctx.log_close, window)
    return slope - slope.shift(lag)


@feature(
    "ema_slope_change",
    category="acceleration",
    doc="Change in normalized EMA slope: is the moving average steepening?",
    params={"span": 50, "lookback": 20, "lag": 20},
    grid=(
        {"span": 20, "lookback": 10, "lag": 10},
        {"span": 200, "lookback": 40, "lag": 40},
    ),
    min_lookback=280,
)
def ema_slope_change(
    ctx: FeatureCtx, *, span: int, lookback: int, lag: int
) -> pd.Series:
    slope = ema_slope(ctx, span=span, lookback=lookback)
    return slope - slope.shift(lag)


@feature(
    "price_velocity",
    category="acceleration",
    doc="Average daily log return over the window: first derivative of log price.",
    params={"window": 20},
    grid=({"window": 10}, {"window": 60}, {"window": 120}),
    min_lookback=125,
)
def price_velocity(ctx: FeatureCtx, *, window: int) -> pd.Series:
    return ctx.log_close.diff(window) / float(window)


@feature(
    "price_acceleration",
    category="acceleration",
    doc="Change in price velocity: second derivative of log price.",
    params={"window": 20},
    grid=({"window": 10}, {"window": 60}, {"window": 120}),
    min_lookback=250,
)
def price_acceleration(ctx: FeatureCtx, *, window: int) -> pd.Series:
    velocity = ctx.log_close.diff(window) / float(window)
    return velocity - velocity.shift(window)


@feature(
    "slope_spread",
    category="acceleration",
    doc="Fast regression slope minus slow regression slope, both on log price.",
    params={"fast": 20, "slow": 120},
    grid=({"fast": 10, "slow": 60}, {"fast": 60, "slow": 250}),
    min_lookback=260,
)
def slope_spread(ctx: FeatureCtx, *, fast: int, slow: int) -> pd.Series:
    fast_slope, _, _ = rolling_ols_slope(ctx.log_close, fast)
    slow_slope, _, _ = rolling_ols_slope(ctx.log_close, slow)
    return fast_slope - slow_slope


@feature(
    "momentum_acceleration",
    category="acceleration",
    doc="Change in `window`-day return over the last `window` days.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}),
    min_lookback=250,
)
def momentum_acceleration(ctx: FeatureCtx, *, window: int) -> pd.Series:
    close = ctx.close
    current = safe_ratio(close, close.shift(window)) - 1.0
    return current - current.shift(window)


@feature(
    "trend_deterioration",
    category="acceleration",
    doc="1 when the long trend is still up but the short slope has turned down.",
    params={"fast": 20, "slow": 200},
    grid=({"fast": 10, "slow": 120}, {"fast": 60, "slow": 250}),
    min_lookback=260,
    higher_is_stronger=False,
)
def trend_deterioration(ctx: FeatureCtx, *, fast: int, slow: int) -> pd.Series:
    # The explicit warning flag the brief asks for, kept as a 0/1 gate rather
    # than a score so it can be read straight off a screen.
    fast_slope, _, _ = rolling_ols_slope(ctx.log_close, fast)
    slow_slope, _, _ = rolling_ols_slope(ctx.log_close, slow)
    flag = (slow_slope > 0.0) & (fast_slope < 0.0)
    return flag.astype(float).where(fast_slope.notna() & slow_slope.notna())
