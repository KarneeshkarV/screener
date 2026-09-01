"""Chart/price/volume ranking recipes.

``ema`` preserves the historical ``scanner._add_setup_score`` weights so the
default screen path stays numerically equivalent.

Everything registered with ``@scorer`` here is ``data_source="snapshot"``:
each recipe reads TradingView's precomputed per-row fields - the vendor
``RSI``, ``relative_volume_10d_calc``, ``market_cap_basic``, the EMA/SMA
columns and the 52-week high/low. Those arrive as one as-of-today value with
no history behind them, so ranking a *past* day by them would use numbers
nobody had on that day. That is why they are screen-only and
``ensure_backtestable_scorer`` rejects them in the backtest path.

``momentum_12_1`` is the exception: it is ``data_source="bars"`` and delegates
to the shared price-only recipe in ``screener.factors``.
"""

from __future__ import annotations

import pandas as pd

from screener.scoring import SNAPSHOT_SOURCE, register_bar_scorer, scorer
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


# ``momentum_12_1`` is bar-derived, not a snapshot recipe. It used to be
# ``100 * percentile((1 + Perf.Y) / (1 + Perf.1M) - 1)``, a different number
# from the ``momentum_12_1`` backtest strategy of the same name, so a backtest
# said nothing about what the screen would pick. Both now read the one recipe
# in ``screener.factors.recipes``.
register_bar_scorer(
    "momentum_12_1",
    "momentum_12_1",
    description="Jegadeesh-Titman 12-1 momentum from bars (same recipe as the "
    "momentum_12_1 backtest strategy)",
)

register_bar_scorer(
    "momentum_12_1_ema10",
    "momentum_12_1",
    description="12-1 momentum ranking with close-above-EMA10 eligibility "
    "from the momentum_12_1_ema10 strategy",
)

register_bar_scorer(
    "ha_momentum",
    "ha_momentum",
    description="12-1 momentum ranked only while Heikin-Ashi confirms an "
    "active uptrend (same recipe as the ha_momentum backtest strategy)",
)


@scorer(
    "mark_minervini",
    columns=_MARK_MINERVINI_COLUMNS,
    description="Minervini template: trend stack + proximity to 52w high + liquidity",
    data_source=SNAPSHOT_SOURCE,
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
    data_source=SNAPSHOT_SOURCE,
)
def score_ema(df: pd.DataFrame) -> pd.Series:
    return _score_ema_setup(df)


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
    data_source=SNAPSHOT_SOURCE,
)
def score_breakout(df: pd.DataFrame) -> pd.Series:
    return _score_breakout_family(df)


@scorer(
    "above_avg_volume",
    columns=("relative_volume_10d_calc",),
    description="Volume surge over the 10d average, weighted by liquidity",
    data_source=SNAPSHOT_SOURCE,
)
def score_above_avg_volume(df: pd.DataFrame) -> pd.Series:
    """Rank a volume-only cut by how big the surge is and how liquid the name.

    The criterion is a prefilter first: it fronts the ``breakout`` strategy's
    ``volume > sma(volume, 10)`` leg. Selected on its own it still needs a
    ranking, and the only thing it asserts about a name is the surge, so that
    is what it ranks on.
    """
    rvol = numeric(df, "relative_volume_10d_calc")
    change = numeric(df, "change")
    volume = numeric(df, "volume")
    close = numeric(df, "close")

    return (
        70 * rvol_surge(rvol, change)
        + 20 * liquidity_from_dollar_volume(volume, close)
        + 10 * momentum_change(change)
    ).round(2)


@scorer(
    "near_52_high",
    columns=_BREAKOUT_COLUMNS,
    description="Under resistance near 52w high with volume confirmation",
    data_source=SNAPSHOT_SOURCE,
)
def score_near_52_high(df: pd.DataFrame) -> pd.Series:
    return _score_breakout_family(df)


@scorer(
    "intraday_breakout",
    columns=_INTRADAY_BREAKOUT_COLUMNS,
    description="Intraday thrust at highs on volume surge",
    data_source=SNAPSHOT_SOURCE,
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
    data_source=SNAPSHOT_SOURCE,
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
    data_source=SNAPSHOT_SOURCE,
)
def score_ema_breakout(df: pd.DataFrame) -> pd.Series:
    blended = (_score_ema_setup(df) + _score_breakout_family(df)) / 2.0
    return blended.round(2)


__all__ = [
    "score_above_avg_volume",
    "score_breakout",
    "score_ema",
    "score_ema_breakout",
    "score_intraday_breakout",
    "score_intraday_momentum",
    "score_mark_minervini",
    "score_near_52_high",
]
