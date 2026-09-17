"""The price-only score recipes themselves, defined exactly once.

Each function here is causal and reads nothing but the bar series on
:class:`screener.factors.BarFeatures`. Registration makes the recipe available
to both adapters, so ``momentum_12_1`` in the screen and ``momentum_12_1`` in
the backtest are the *same* number rather than two formulas sharing a name.

``RSI``, ``relative_volume_10d_calc`` and ``Perf.Y`` were TradingView snapshot
columns; they are computed from bars here instead, in the *same units the
vendor reports* (RSI 0-100, RVOL as a ratio around 1.0, Perf.Y in percent), so
the thresholds the snapshot scorers already use keep their meaning when those
scorers switch over.

Deliberately *not* here: ``market_cap_basic``. It is not derivable from bars,
and its point-in-time form is ``market_cap``, which reaches the backtester
through :func:`screener.backtester.fundamentals.merge_fundamentals_into_bars`
and is read off :attr:`screener.factors.BarFeatures.fundamentals` rather than
recomputed. A raw snapshot column carries only today's value, so replaying one
through history would be lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors import BarFeatures, price_score
from screener.indicators.plugins.heikin_ashi import heikin_ashi_ohlc
from screener.indicators.plugins.rsi import rsi as _wilder_rsi

# Trading-day windows. 252 ~ 12 months, 21 ~ 1 month (the skipped reversal leg).
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21

#: Consecutive bullish Heikin-Ashi candles required to confirm a trend.
HA_STREAK_MIN = 3

#: Wilder's default RSI period, matching TradingView's ``RSI`` column.
RSI_PERIOD = 14
#: TradingView's ``relative_volume_10d_calc`` averages ten sessions.
RVOL_WINDOW = 10
#: TradingView's ``Perf.Y`` is a trailing one-year return.
PERF_Y_LOOKBACK = 252

#: Smallest close a momentum leg may sit on. Below one cent a quote is not a
#: price: US OTC and grey-market lines are carried at a flat $0.0001 stub for
#: months at a time, volume zero, and that stub becomes the denominator of a
#: 12-1 ratio ``close[t-21] / close[t-252] - 1``. One observed name divided a
#: $4.26 close by a $0.0001 stub and scored +4,259,800%, the top of a 13,326
#: name field. A cent is also the minimum tick on both markets this screener
#: covers, so no real quote is excluded by the floor.
MIN_TRADEABLE_PRICE = 0.01

#: Fraction of the sessions spanned by a momentum window that must have traded
#: before the window's endpoints count as prices. A series the vendor carries
#: forward through months of no trading has closes at every bar, so bar count
#: alone cannot tell it from a liquid name; the volume column can. Half is
#: deliberately permissive - it keeps thinly traded but real names and cuts
#: only series that are mostly not a market.
MIN_TRADED_FRACTION = 0.5


def tradeable_momentum_window(
    close: pd.Series,
    volume: pd.Series | None,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.Series:
    """Boolean mask: bars whose 12-1 window rests on real, traded prices.

    A momentum ratio is only a return when both of its legs are prices a
    market actually set. Two ways that fails, and neither shows up as missing
    data - the frame is full-length and every bar carries a close:

    * **A stub quote.** A venue carries a dormant line at a flat sub-cent
      value. As the ratio's denominator it manufactures a five-figure return.
      :data:`MIN_TRADEABLE_PRICE` rejects both legs below one tick.
    * **A series that is mostly not trading.** The vendor repeats the last
      close through months of no trades, so the endpoints are stale marks
      rather than prices. :data:`MIN_TRADED_FRACTION` of the sessions the
      window spans must carry non-zero volume.

    ``volume`` may be ``None`` - a close-only frame cannot answer the second
    question, so only the price floor applies there. That is stated rather
    than assumed away: the gate is as strong as the columns allow.
    """
    prices = pd.to_numeric(close, errors="coerce").astype(float)
    floor = MIN_TRADEABLE_PRICE
    eligible = (prices.shift(skip) >= floor) & (prices.shift(lookback) >= floor)
    if volume is None:
        return eligible
    traded = (pd.to_numeric(volume, errors="coerce").astype(float) > 0).astype(float)
    # Sessions strictly inside the window, i.e. the ones the return is made
    # of. ``shift(skip)`` puts the near leg at bar ``t``; the window then
    # reaches back ``lookback - skip`` sessions to the far leg.
    span = max(int(lookback) - int(skip), 1)
    fraction = traded.shift(skip).rolling(span).mean()
    return eligible & (fraction >= MIN_TRADED_FRACTION)


def momentum_12_1(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.Series:
    """Jegadeesh-Titman 12-1 momentum, as of bar ``t``.

    ``close[t - skip] / close[t - lookback] - 1``: the cumulative return from
    ~12 months ago to ~1 month ago. Skipping the last 21 sessions avoids the
    short-term reversal that contaminates raw 12-month momentum.

    Stays NaN until ``lookback`` prior closes exist, and wherever
    :func:`tradeable_momentum_window` says the window does not rest on real
    prices. Both NaNs are the eligibility signal for both adapters - neither
    is ever filled with 0.

    ``volume`` is optional only so a close-only caller still works; pass it
    whenever the frame has it, because it is what rules out a series the
    vendor carried forward rather than a market traded.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    raw = values.shift(skip) / values.shift(lookback) - 1.0
    return raw.where(
        tradeable_momentum_window(values, volume, lookback=lookback, skip=skip)
    )


@price_score(
    "momentum_12_1",
    required_lookback=MOMENTUM_LOOKBACK,
    description="Jegadeesh-Titman 12-1 momentum: close[t-21]/close[t-252] - 1",
    aux_column="mom_12_1",
    # Jegadeesh-Titman buys winners, so a non-positive 12-1 return is not a
    # candidate at all. Declaring the floor with the recipe is what lets the
    # screen and the backtest gate on one rule instead of two: the screen
    # filters on it after bar scoring, and the backtest's ``ENTRY_PURE``
    # expression is rendered from it.
    eligible_above=0.0,
)
def score_momentum_12_1(features: BarFeatures) -> pd.Series:
    return momentum_12_1(features.close, features.volume)


def ha_momentum(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    min_streak: int = HA_STREAK_MIN,
) -> pd.Series:
    """12-1 momentum, confirmed by an active Heikin-Ashi uptrend.

    Same ``mom_12_1`` value as :func:`momentum_12_1`, tradeability gate
    included, except a name only
    carries a score while its Heikin-Ashi candles are on a bullish streak of
    at least ``min_streak`` bars (``ha_close > ha_open``, no wicks below the
    body: ``ha_open == ha_low``). Everywhere else the value is NaN, which
    both adapters already treat as "not a candidate" - Heikin-Ashi's smoothing
    is the trend-confirmation leg, 12-1 momentum stays the ranking leg.
    """
    op = np.asarray(pd.to_numeric(open_, errors="coerce"), dtype=np.float64)
    hi = np.asarray(pd.to_numeric(high, errors="coerce"), dtype=np.float64)
    lo = np.asarray(pd.to_numeric(low, errors="coerce"), dtype=np.float64)
    cl = np.asarray(pd.to_numeric(close, errors="coerce"), dtype=np.float64)

    ha_open, _ha_high, ha_low, ha_close = heikin_ashi_ohlc(op, hi, lo, cl)
    bullish = pd.Series((ha_close > ha_open) & (ha_open == ha_low), index=close.index)
    # Length of the bullish run ending at each bar. Grouping by
    # ``(~bullish).cumsum()`` puts the terminating bearish bar first in its own
    # group, so ``cumcount() + 1`` would score the first bullish bar after any
    # bearish bar as 2. Subtracting the running total carried at the last
    # bearish bar counts only the bullish bars since it.
    running = bullish.cumsum()
    streak = running - running.where(~bullish).ffill().fillna(0)
    confirmed = streak >= min_streak

    mom = momentum_12_1(close, volume, lookback=lookback, skip=skip)
    return mom.where(confirmed)


@price_score(
    "ha_momentum",
    required_lookback=MOMENTUM_LOOKBACK,
    description=(
        "12-1 momentum, ranked only while Heikin-Ashi confirms an active "
        f"uptrend ({HA_STREAK_MIN}+ consecutive bullish HA candles)"
    ),
    aux_column="ha_mom_12_1",
    # Same floor as plain momentum_12_1: only long the winners.
    eligible_above=0.0,
)
def score_ha_momentum(features: BarFeatures) -> pd.Series:
    if features.open is None or features.high is None or features.low is None:
        return pd.Series(np.nan, index=features.close.index, dtype=float)
    return ha_momentum(
        features.open,
        features.high,
        features.low,
        features.close,
        features.volume,
    )


def rsi_14(close: pd.Series, *, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI on the 0-100 scale, the unit TradingView's ``RSI`` reports.

    Delegates to the one RSI implementation in
    :mod:`screener.indicators.plugins.rsi` rather than restating the formula,
    and keeps its NaN warm-up: the first ``period - 1`` bars have no value.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    if values.empty:
        return values
    return pd.Series(
        _wilder_rsi(values.to_numpy(dtype=float), period),
        index=values.index,
        dtype=float,
    )


def relative_volume_10d(volume: pd.Series, *, window: int = RVOL_WINDOW) -> pd.Series:
    """Today's volume over the trailing ``window``-session average volume.

    A ratio around 1.0, matching ``relative_volume_10d_calc``. The window
    includes the current bar, exactly as the vendor's does, which stays causal
    because today's own volume is known at today's close.
    """
    values = pd.to_numeric(volume, errors="coerce").astype(float)
    average = values.rolling(window).mean()
    return values / average.where(average > 0)


def perf_y(close: pd.Series, *, lookback: int = PERF_Y_LOOKBACK) -> pd.Series:
    """Trailing one-year return in *percent*, the unit ``Perf.Y`` reports."""
    values = pd.to_numeric(close, errors="coerce").astype(float)
    return (values / values.shift(lookback) - 1.0) * 100.0


@price_score(
    "rsi_14",
    required_lookback=RSI_PERIOD,
    description="Wilder RSI(14) on bars, 0-100, replacing the TradingView RSI column",
    aux_column="rsi_14",
    # No floor: RSI is an input to a shaped score (``rsi_quality`` peaks at 60
    # and falls off either side), not a level that makes a name tradeable on
    # its own. Only NaN - too little history - is ineligible.
)
def score_rsi_14(features: BarFeatures) -> pd.Series:
    return rsi_14(features.close)


@price_score(
    "relative_volume_10d",
    required_lookback=RVOL_WINDOW,
    description="Volume over its trailing 10-session average, as a ratio",
    aux_column="rvol_10d",
)
def score_relative_volume_10d(features: BarFeatures) -> pd.Series:
    if features.volume is None:
        return pd.Series(np.nan, index=features.close.index, dtype=float)
    return relative_volume_10d(features.volume)


@price_score(
    "perf_y",
    required_lookback=PERF_Y_LOOKBACK,
    description="Trailing 252-session return in percent, replacing Perf.Y",
    aux_column="perf_y",
)
def score_perf_y(features: BarFeatures) -> pd.Series:
    return perf_y(features.close)


__all__ = [
    "HA_STREAK_MIN",
    "MIN_TRADEABLE_PRICE",
    "MIN_TRADED_FRACTION",
    "MOMENTUM_LOOKBACK",
    "MOMENTUM_SKIP",
    "PERF_Y_LOOKBACK",
    "RSI_PERIOD",
    "RVOL_WINDOW",
    "ha_momentum",
    "momentum_12_1",
    "perf_y",
    "relative_volume_10d",
    "rsi_14",
    "score_ha_momentum",
    "score_momentum_12_1",
    "score_perf_y",
    "score_relative_volume_10d",
    "score_rsi_14",
    "tradeable_momentum_window",
]
