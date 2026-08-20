"""Reversal, 52-week-high and price-anomaly strategies (round 2).

Improves on the round-1 ``long_term_reversal`` / ``short_term_reversal`` /
``fifty_two_week_high`` plugins with the parent-anomaly literature's most
important robustness findings, and adds one cross-sectional calendar anomaly
not covered anywhere else in the repo:

- ``lt_reversal_path``  — De Bondt & Thaler (1985), "Does the Stock Market
                          Overreact?", Journal of Finance 40(3), 793-805.
                          Long-term reversal: 3-5 year losers outperform over
                          the following 36 months (the overreaction
                          hypothesis). Round 1 ranked by raw 3-year return
                          alone; this version requires the name to trade
                          *below its longer-term path* (``close < 2y SMA``) on
                          top of the negative 3-year return, so the portfolio
                          only buys stocks that are both structurally beaten
                          down and still under pressure — the classic DBT
                          loser portfolio — instead of 3-year losers that have
                          already mean-reverted.
- ``str_reversal_trend`` — Jegadeesh (1990), "Evidence of Predictable Behavior
                          of Security Returns", Journal of Finance 45(3),
                          881-898. One-month reversal: last month's losers
                          bounce next month. Round 1 bought every 21-day
                          loser (falling knives); this version gates on a
                          200-day uptrend plus a crash filter (skip >25%
                          one-month drops) and a volume floor, so it buys
                          *pullbacks inside uptrends* — the investable form of
                          the anomaly.
- ``gw52_proximity``     — George & Hwang (2004), "The 52-Week High and
                          Momentum Investing", Journal of Finance 59(5),
                          2145-2176. Anchoring: price proximity to the
                          52-week high predicts returns *better* than raw
                          past returns, and survives controlling for
                          Jegadeesh-Titman momentum. Round-1
                          ``fifty_two_week_high`` trades the fresh-high
                          breakout; this strategy instead ranks the whole
                          universe by ``close / highest(close, 252)`` and
                          longs the names closest to their high — the paper's
                          long-only decile portfolio, cross-sectionally.
- ``hs_same_month``      — Heston & Sadka (2008), "Seasonality in the
                          Cross-Section of Stock Returns", Journal of
                          Financial Economics 87(2), 418-445. Same-calendar-
                          month momentum: the return a stock earned in month
                          M last year predicts its return in month M this
                          year (r_{t-12} seasonal, distinct from momentum and
                          the January effect). We rank by the prior-year
                          same-month return, long-only.

All four are pure price-side factors: no fundamentals required, every market
works, and each ``rank_score`` is causal (bar ``t`` uses only bars ``<= t``).
Grinblatt & Moskowitz (2004, "Predicting Stock Price Movements from Past
Returns", JFE 71(3)) — momentum profits concentrate in *consistent* winners
near their 52-week high — motivates the trend/consistency overlays above and
is the natural next variant if the proximity factor earns its keep.

Hold suggestions (the backtester's ``--hold``, trading days): long-term
reversal 126 (DBT's multi-year horizon, repo value/quality convention),
short-term reversal 21 (Jegadeesh's monthly cadence), 52-week proximity 126
(G&H rebalance), same-month seasonality 21 (the effect pays off inside the
same calendar month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Windows (trading days) ────────────────────────────────────────────────
_MONTH = 21  # one trading month
_YEAR = 252  # one trading year
_PATH_WINDOW = 504  # two trading years (longer-term path)
_LT_WINDOW = 756  # three trading years (De Bondt-Thaler loser window)

# ── Shared gates ──────────────────────────────────────────────────────────
_TREND_SMA = 200  # long-term uptrend filter (falling-knife guard)
_VOLUME_SMA = 63  # quarter-average volume (liquidity floor)
_CRASH_FLOOR = -0.25  # skip one-month drops worse than -25% (distressed names)
_PROXIMITY_MIN = 0.7  # George-Hwang: within 30% of the 52-week high

# ── 1. Long-term reversal below the longer-term path (De Bondt & Thaler) ──
LT_REVERSAL_ENTRY = "ret_756 < 0 and close < path_504"


def _prepare_lt_reversal(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the 3-year return and 2-year SMA path; rank by underperformance.

    ``rank_score = -ret_756`` so the descending ranker fills its ``--top``
    slots with the most beaten-down names (biggest 3-year losers first). The
    entry expression additionally requires the close to sit below the 2-year
    average path, i.e. the name is still under its longer-term trend.
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["ret_756"] = close / close.shift(_LT_WINDOW) - 1.0
        frame["path_504"] = close.rolling(_PATH_WINDOW, min_periods=_PATH_WINDOW).mean()
        frame["rank_score"] = -frame["ret_756"]
        out[tv] = frame
    return out


def _lookback_lt() -> int:
    # 3-year return leg needs 756 prior closes; covers path_504 (504) too.
    return _LT_WINDOW + 1


# ── 2. Short-term reversal inside an uptrend (Jegadeesh + trend gate) ─────
ST_REVERSAL_ENTRY = (
    f"ret_21 < 0 and ret_21 > {_CRASH_FLOOR} "
    f"and close > sma(close, {_TREND_SMA}) "
    f"and volume > sma(volume, {_VOLUME_SMA}) * 0.5"
)


def _prepare_st_reversal(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the 1-month return; rank by short-term underperformance.

    ``rank_score = -ret_21`` picks the deepest 21-day pullbacks; the entry
    gates (uptrend, crash filter, volume floor) keep them investable.
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["ret_21"] = close / close.shift(_MONTH) - 1.0
        frame["rank_score"] = -frame["ret_21"]
        out[tv] = frame
    return out


def _lookback_st() -> int:
    # 200-day trend SMA dominates the 21-day return and 63-day volume SMA.
    return _TREND_SMA + 1


# ── 3. 52-week-high proximity (George & Hwang) ────────────────────────────
GW52_ENTRY = f"ratio_52w > {_PROXIMITY_MIN}"


def _prepare_proximity(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the 52-week-high proximity ratio; rank by it (high = good).

    ``ratio_52w = close / highest(close, 252)`` is in (0, 1]; a name at its
    yearly peak scores 1.0. The descending ranker fills ``--top`` with the
    names closest to their 52-week high — the anchoring portfolio of George
    & Hwang (2004).
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        high_252 = close.rolling(_YEAR, min_periods=_YEAR).max()
        frame["ratio_52w"] = close / high_252
        frame["rank_score"] = frame["ratio_52w"]
        out[tv] = frame
    return out


def _lookback_52w() -> int:
    # ``highest(close, 252)`` needs a full year of prior closes.
    return _YEAR


# ── 4. Same-calendar-month seasonality (Heston & Sadka) ───────────────────
HS_ENTRY = "same_month_ret > 0"


def _prior_year_same_month_return(close: pd.Series) -> pd.Series:
    """Return *during* the same calendar month one year ago (lag-12 seasonal).

    For a bar in month M of year Y: ``last_close(M, Y-1) / last_close(M-1,
    Y-1) - 1`` — the return the stock earned in month M last year. The value
    is constant within each month and changes only at month boundaries; it is
    fully causal because month M of Y-1 is complete before month M of Y
    starts. Names without ~13 months of history get NaN (ineligible).
    """
    index = close.index
    if not isinstance(index, pd.DatetimeIndex):
        # Backtester bars always carry a DatetimeIndex; anything else (e.g. a
        # bare Series in a unit test) gets an all-NaN signal -> ineligible.
        return pd.Series(np.nan, index=index)
    month = index.to_period("M")
    monthly = close.groupby(month).last()
    monthly_ret = monthly / monthly.shift(1) - 1.0
    same = monthly_ret.shift(12)  # same calendar month, one year earlier
    values = same.loc[month.to_numpy()].to_numpy()
    return pd.Series(values, index=close.index)


def _prepare_same_month(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the prior-year same-month return; rank by it (high = good).

    ``rank_score = same_month_ret``: among names that rose in month M last
    year (the ``same_month_ret > 0`` gate), the descending ranker picks the
    strongest same-month winners — Heston & Sadka's seasonal portfolio.
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["same_month_ret"] = _prior_year_same_month_return(close)
        frame["rank_score"] = frame["same_month_ret"]
        out[tv] = frame
    return out


def _lookback_same_month() -> int:
    # 12-month lag + one month-end close for the monthly return leg:
    # ~13 calendar months of history, ~262 trading days; 275 for margin.
    return 275


register_expression_strategy(
    "lt_reversal_path",
    entry=LT_REVERSAL_ENTRY,
    exit=None,
    prepare_bars=_prepare_lt_reversal,
    required_lookback=_lookback_lt,
)

register_expression_strategy(
    "str_reversal_trend",
    entry=ST_REVERSAL_ENTRY,
    exit=None,
    prepare_bars=_prepare_st_reversal,
    required_lookback=_lookback_st,
)

register_expression_strategy(
    "gw52_proximity",
    entry=GW52_ENTRY,
    exit=None,
    prepare_bars=_prepare_proximity,
    required_lookback=_lookback_52w,
)

register_expression_strategy(
    "hs_same_month",
    entry=HS_ENTRY,
    exit=None,
    prepare_bars=_prepare_same_month,
    required_lookback=_lookback_same_month,
)
