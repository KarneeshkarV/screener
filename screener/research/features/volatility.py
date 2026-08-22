"""Category 3: volatility and risk, expressed relatively wherever possible.

A fixed ATR threshold is not portable across names or across years, so every
feature here is either a ratio, a percentile against the stock's own history, or
a normalized fraction of price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.frames import wilder_atr
from screener.research.features.base import (
    FeatureCtx,
    feature,
    rolling_percentile,
    safe_ratio,
)

_ANNUAL = 252.0


@feature(
    "realized_vol",
    category="volatility",
    doc="Annualized trailing standard deviation of daily log returns.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
    higher_is_stronger=False,
)
def realized_vol(ctx: FeatureCtx, *, window: int) -> pd.Series:
    daily = ctx.log_returns.rolling(window, min_periods=window).std(ddof=1)
    return pd.Series(daily * np.sqrt(_ANNUAL), index=daily.index)


@feature(
    "vol_ratio",
    category="volatility",
    doc="Short-horizon vol over long-horizon vol. >1 means vol is expanding.",
    params={"fast": 20, "slow": 120},
    grid=(
        {"fast": 10, "slow": 60},
        {"fast": 20, "slow": 250},
        {"fast": 60, "slow": 250},
    ),
    min_lookback=260,
    higher_is_stronger=False,
)
def vol_ratio(ctx: FeatureCtx, *, fast: int, slow: int) -> pd.Series:
    returns = ctx.log_returns
    fast_vol = returns.rolling(fast, min_periods=fast).std(ddof=1)
    slow_vol = returns.rolling(slow, min_periods=slow).std(ddof=1)
    return safe_ratio(fast_vol, slow_vol)


@feature(
    "vol_percentile",
    category="volatility",
    doc="Where current realized vol sits in the stock's own trailing history.",
    params={"window": 60, "history": 500},
    grid=({"window": 20, "history": 250}, {"window": 60, "history": 250}),
    min_lookback=560,
    higher_is_stronger=False,
)
def vol_percentile(ctx: FeatureCtx, *, window: int, history: int) -> pd.Series:
    # Self-referential by design: "calm for this name" beats "calm in absolute
    # terms" when the universe spans very different volatility regimes.
    vol = ctx.log_returns.rolling(window, min_periods=window).std(ddof=1)
    return rolling_percentile(vol, history)


@feature(
    "vol_of_vol",
    category="volatility",
    doc="Volatility of the realized-vol series itself, relative to its own level.",
    params={"window": 20, "outer": 120},
    grid=({"window": 20, "outer": 250}, {"window": 60, "outer": 250}),
    min_lookback=280,
    higher_is_stronger=False,
)
def vol_of_vol(ctx: FeatureCtx, *, window: int, outer: int) -> pd.Series:
    # Scaled by the mean so this is a coefficient of variation, not a number
    # that grows simply because the name is volatile.
    vol = ctx.log_returns.rolling(window, min_periods=window).std(ddof=1)
    spread = vol.rolling(outer, min_periods=outer).std(ddof=1)
    level = vol.rolling(outer, min_periods=outer).mean()
    return safe_ratio(spread, level)


@feature(
    "atr_pct",
    category="volatility",
    doc="Wilder ATR as a fraction of close: average daily range in percent.",
    params={"window": 14},
    grid=({"window": 20}, {"window": 60}),
    min_lookback=70,
    higher_is_stronger=False,
)
def atr_pct(ctx: FeatureCtx, *, window: int) -> pd.Series:
    atr = wilder_atr(
        ctx.bars["high"].astype(float),
        ctx.bars["low"].astype(float),
        ctx.close,
        window,
        min_periods=window,
    )
    return safe_ratio(atr, ctx.close)


@feature(
    "downside_vol",
    category="volatility",
    doc="Annualized standard deviation of negative daily returns only.",
    params={"window": 60},
    grid=({"window": 120}, {"window": 250}),
    min_lookback=250,
    higher_is_stronger=False,
)
def downside_vol(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # Semi-deviation about zero: upside dispersion is not a risk a long book
    # wants to be penalized for.
    returns = ctx.log_returns
    losses = returns.where(returns < 0.0, 0.0).where(returns.notna())
    mean_square = (losses**2).rolling(window, min_periods=window).mean()
    return pd.Series(np.sqrt(mean_square) * np.sqrt(_ANNUAL), index=mean_square.index)


@feature(
    "recent_drawdown",
    category="volatility",
    doc="Current drawdown from the trailing-window peak close, as a negative fraction.",
    params={"window": 120},
    grid=({"window": 60}, {"window": 250}),
    min_lookback=250,
    higher_is_stronger=False,
)
def recent_drawdown(ctx: FeatureCtx, *, window: int) -> pd.Series:
    peak = ctx.close.rolling(window, min_periods=window).max()
    return safe_ratio(ctx.close, peak) - 1.0


@feature(
    "tail_frequency",
    category="volatility",
    doc="Share of days in the window whose move exceeded k trailing sigmas.",
    params={"window": 120, "k": 3.0, "vol_window": 60},
    grid=(
        {"window": 250, "k": 3.0, "vol_window": 60},
        {"window": 120, "k": 2.0, "vol_window": 60},
    ),
    min_lookback=320,
    higher_is_stronger=False,
)
def tail_frequency(
    ctx: FeatureCtx, *, window: int, k: float, vol_window: int
) -> pd.Series:
    # Gap and shock frequency. The sigma is itself trailing and lagged one bar,
    # so a day is never judged against a threshold that its own move helped set.
    returns = ctx.log_returns
    sigma = returns.rolling(vol_window, min_periods=vol_window).std(ddof=1).shift(1)
    extreme = (returns.abs() > k * sigma).astype(float).where(sigma.notna())
    return extreme.rolling(window, min_periods=window).mean()
