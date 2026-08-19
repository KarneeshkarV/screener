"""Chart/price/volume ranking recipes.

``ema`` preserves the historical ``scanner._add_setup_score`` weights so the
default screen path stays numerically equivalent.
"""

from __future__ import annotations

import pandas as pd

from screener.scoring import scorer
from screener.scoring.components import (
    above_flag,
    liquidity_from_dollar_volume,
    log_percentile,
    momentum_change,
    numeric,
    overextension_penalty,
    percentile,
    proximity_to_high,
    rsi_quality,
    rvol_surge,
    trend_stack_strength,
)

_MOMENTUM_12_1_COLUMNS = ("Perf.Y", "Perf.1M")
_MARK_MINERVINI_COLUMNS = (
    "SMA50",
    "SMA150",
    "SMA200",
    "price_52_week_high",
    "price_52_week_low",
)
_EMA_COLUMNS = ("EMA5", "EMA20", "EMA100", "EMA200", "RSI")
_BREAKOUT_COLUMNS = (
    "price_52_week_high",
    "relative_volume_10d_calc",
    "EMA20",
    "EMA200",
    "RSI",
)
_INTRADAY_MOM_COLUMNS = (
    "relative_volume_10d_calc",
    "EMA20",
    "EMA200",
    "RSI",
)
_INTRADAY_BREAKOUT_COLUMNS = (
    "price_52_week_high",
    "relative_volume_10d_calc",
    "EMA5",
    "EMA20",
    "RSI",
)


def _score_ema_setup(df: pd.DataFrame) -> pd.Series:
    """Trend stack + liquidity + RSI sweet spot − overextension.

    Weights (legacy setup_score):
      25*liquidity + 30*trend + 15*momentum + 15*mcap
      + 10*rsi_quality + 5*price_quality − 15*overextension
    """
    close = numeric(df, "close")
    ema5 = numeric(df, "EMA5")
    ema20 = numeric(df, "EMA20")
    ema100 = numeric(df, "EMA100")
    ema200 = numeric(df, "EMA200")
    change = numeric(df, "change")
    rsi = numeric(df, "RSI")
    volume = numeric(df, "volume")

    liquidity = liquidity_from_dollar_volume(volume, close)
    market_cap = log_percentile(numeric(df, "market_cap_basic"))
    trend_strength = trend_stack_strength(close, ema5, ema20, ema100, ema200)
    momentum = momentum_change(change)
    rsi_q = rsi_quality(rsi)
    price_quality = percentile(close.clip(lower=0, upper=200))
    penalty = overextension_penalty(close, ema20)

    return (
        25 * liquidity
        + 30 * trend_strength
        + 15 * momentum
        + 15 * market_cap
        + 10 * rsi_q
        + 5 * price_quality
        - 15 * penalty
    ).round(2)


@scorer(
    "momentum_12_1",
    columns=_MOMENTUM_12_1_COLUMNS,
    description="Higher 12-1 momentum (1y return net of last-month return)",
)
def score_momentum_12_1(df: pd.DataFrame) -> pd.Series:
    perf_y = numeric(df, "Perf.Y")
    perf_m = numeric(df, "Perf.1M")
    mom_12_1 = ((1.0 + perf_y) / (1.0 + perf_m) - 1.0).fillna(-1.0)
    return (100 * percentile(mom_12_1)).round(2)


@scorer(
    "mark_minervini",
    columns=_MARK_MINERVINI_COLUMNS,
    description="Minervini template: trend stack + proximity to 52w high + liquidity",
)
def score_mark_minervini(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    high = numeric(df, "price_52_week_high")
    low = numeric(df, "price_52_week_low")
    sma50 = numeric(df, "SMA50")
    sma150 = numeric(df, "SMA150")
    sma200 = numeric(df, "SMA200")
    volume = numeric(df, "volume")

    stack = (
        above_flag(close, sma50)
        + above_flag(sma50, sma150)
        + above_flag(sma150, sma200)
    ) / 3.0
    near_high = proximity_to_high(close, high)
    above_low = above_flag(close, low * 1.3)
    liquidity = liquidity_from_dollar_volume(volume, close)

    return (35 * stack + 25 * near_high + 15 * above_low + 25 * liquidity).round(2)


@scorer(
    "ema",
    columns=_EMA_COLUMNS,
    description="Trend stack + liquidity + RSI sweet spot − overextension",
)
def score_ema(df: pd.DataFrame) -> pd.Series:
    return _score_ema_setup(df)


def _score_above_ema(df: pd.DataFrame, ema_column: str) -> pd.Series:
    """Rank names already above one moving average by how healthy that stance is.

    The criterion has filtered on the moving average, so the flag itself carries
    no ranking information: every match is above it. What separates the matches
    is how much room the move has left. The recipe therefore rewards liquidity,
    a positive EMA20/EMA200 stack and recent change, and penalizes price that has
    stretched far above its 20-day average - a stock 30% above the line is above
    it for a worse reason than one 3% above.
    """
    close = numeric(df, "close")
    ema20 = numeric(df, "EMA20")
    ema200 = numeric(df, "EMA200")
    volume = numeric(df, "volume")

    liquidity = liquidity_from_dollar_volume(volume, close)
    long_trend = above_flag(ema20, ema200)
    headroom = percentile(-((close - numeric(df, ema_column)) / close).abs())
    momentum = momentum_change(numeric(df, "change"))
    penalty = overextension_penalty(close, ema20)

    return (
        35 * liquidity + 25 * long_trend + 20 * headroom + 20 * momentum - 15 * penalty
    ).round(2)


@scorer(
    "above_20ema",
    columns=("EMA20", "EMA200"),
    description="Short-term trend stance: liquidity + stack + headroom − overextension",
)
def score_above_20ema(df: pd.DataFrame) -> pd.Series:
    return _score_above_ema(df, "EMA20")


@scorer(
    "above_200ema",
    columns=("EMA20", "EMA200"),
    description="Long-term trend stance: liquidity + stack + headroom − overextension",
)
def score_above_200ema(df: pd.DataFrame) -> pd.Series:
    return _score_above_ema(df, "EMA200")


def _score_breakout_family(df: pd.DataFrame) -> pd.Series:
    """Proximity to 52w high + volume confirmation + mild trend support."""
    close = numeric(df, "close")
    high = numeric(df, "price_52_week_high")
    rvol = numeric(df, "relative_volume_10d_calc")
    change = numeric(df, "change")
    volume = numeric(df, "volume")
    ema20 = numeric(df, "EMA20")
    ema200 = numeric(df, "EMA200")
    rsi = numeric(df, "RSI")

    near_high = proximity_to_high(close, high)
    # Prefer cross-sectional rank of RVOL when present; fall back to change energy.
    vol_surge = rvol_surge(rvol, change)
    liquidity = liquidity_from_dollar_volume(volume, close)
    trend = above_flag(ema20, ema200)
    rsi_q = rsi_quality(rsi, center=65.0, half_width=35.0)
    # Mild penalty if already far above EMA20 (chasing exhaust).
    penalty = overextension_penalty(close, ema20, start=0.15, span=0.30)

    return (
        35 * near_high
        + 25 * vol_surge
        + 15 * liquidity
        + 10 * trend
        + 10 * rsi_q
        + 5 * momentum_change(change)
        - 10 * penalty
    ).round(2)


@scorer(
    "breakout",
    columns=_BREAKOUT_COLUMNS,
    description="Near 52w high + relative volume + trend support",
)
def score_breakout(df: pd.DataFrame) -> pd.Series:
    return _score_breakout_family(df)


@scorer(
    "near_52_high",
    columns=_BREAKOUT_COLUMNS,
    description="Under resistance near 52w high with volume confirmation",
)
def score_near_52_high(df: pd.DataFrame) -> pd.Series:
    return _score_breakout_family(df)


@scorer(
    "intraday_breakout",
    columns=_INTRADAY_BREAKOUT_COLUMNS,
    description="Intraday thrust at highs on volume surge",
)
def score_intraday_breakout(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    high = numeric(df, "price_52_week_high")
    rvol = numeric(df, "relative_volume_10d_calc")
    change = numeric(df, "change")
    volume = numeric(df, "volume")
    ema5 = numeric(df, "EMA5")
    ema20 = numeric(df, "EMA20")

    near_high = proximity_to_high(close, high)
    vol_surge = rvol_surge(rvol, change)
    liquidity = liquidity_from_dollar_volume(volume, close)
    short_trend = above_flag(ema5, ema20)
    day_move = momentum_change(change)

    return (
        30 * near_high
        + 30 * vol_surge
        + 15 * day_move
        + 15 * liquidity
        + 10 * short_trend
    ).round(2)


@scorer(
    "intraday_momentum",
    columns=_INTRADAY_MOM_COLUMNS,
    description="Liquid day-movers: RVOL + change + RSI band + trend",
)
def score_intraday_momentum(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    rvol = numeric(df, "relative_volume_10d_calc")
    change = numeric(df, "change")
    volume = numeric(df, "volume")
    ema20 = numeric(df, "EMA20")
    ema200 = numeric(df, "EMA200")
    rsi = numeric(df, "RSI")

    vol_surge = rvol_surge(rvol, change)
    day_move = momentum_change(change)
    # Intraday sweet spot is a bit hotter than the swing RSI center.
    rsi_q = rsi_quality(rsi, center=65.0, half_width=25.0)
    liquidity = liquidity_from_dollar_volume(volume, close)
    trend = above_flag(close, ema20) * 0.5 + above_flag(ema20, ema200) * 0.5

    return (
        30 * vol_surge + 25 * day_move + 20 * liquidity + 15 * rsi_q + 10 * trend
    ).round(2)


@scorer(
    "ema_breakout",
    columns=tuple(dict.fromkeys(_EMA_COLUMNS + _BREAKOUT_COLUMNS)),
    description="Equal blend of EMA trend setup and 52w breakout quality",
)
def score_ema_breakout(df: pd.DataFrame) -> pd.Series:
    blended = (_score_ema_setup(df) + _score_breakout_family(df)) / 2.0
    return blended.round(2)


__all__ = [
    "score_breakout",
    "score_ema",
    "score_ema_breakout",
    "score_intraday_breakout",
    "score_intraday_momentum",
    "score_mark_minervini",
    "score_momentum_12_1",
    "score_near_52_high",
]
