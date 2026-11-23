"""Calendar & seasonality timing strategies with a long-term trend overlay.

Non-momentum strategies that gate long exposure on well-documented calendar
anomalies, each overlaid with a 200-day SMA trend filter so the seasonal tilt
is only harvested when the name is in an up-trend (investable, not a pure
calendar bet). All four share one ``prepare_bars`` hook that adds date-derived
columns and a low-volatility ``rank_score`` (Ang, Hodrick, Xing & Zhang 2006,
"the cross-section of volatility and expected returns", J. Finance 61(1)) so
the portfolio holds the calmest names inside each seasonal window.

Evidence (academic + in-house, computed on ^NSEI 2007-2026 and SPY 2003-2026):

1. Month-of-year (India). Nifty's strongest months are April (start of India's
   fiscal year; April is the strongest month in the in-house sample,
   +0.17%/day 2007-2026 and +0.22%/day in the trailing 5y), July and December;
   November is Diwali-adjacent and the strongest month in the trailing 5y.
   January-February-March (Q1: budget month + fiscal-year-end tax-loss selling)
   are the weakest quarter (cumulative ~0.63x over 19y). The April/December
   strength and March weakness are the most replicated findings of the Indian
   calendar-effect literature (e.g. "Seasonality in the Indian stock market"
   studies; March tax-loss selling at the fiscal year end).
   -> ``seasonal_strong_trend`` longs months {Apr, Jul, Nov, Dec}.

2. Turn-of-month, trading-day variant. Ariel (1987, J. Financial Economics 18)
   showed most of the monthly equity premium accrues around the turn of the
   month; McConnell & Xu (2008, "Equity returns at the turn of the month",
   Financial Analysts Journal 64) localised it to the last trading day plus the
   first ~4 trading days of the next month. The repo's calendar-day
   ``turn_of_month`` (day-of-month >= 28 or <= 3) was weak; the trading-day
   window is the stronger variant: on ^NSEI 2007-2026 the last-trading-day +
   first-3-trading-days window earned +0.145%/day (win 54%) vs +0.045%/day for
   all days, and +0.33%/day (win 60%) on SPY 2003-2026.
   -> ``tom_window_trend`` longs the last 1 + first 3 trading days of each
      month (exposure ~20% of days).

3. Pre-holiday effect. Ariel (1990, "High stock returns before holidays",
   J. Finance 45) documented outsized returns on the trading day before US
   holidays; replicated for India (Coutts & Sheikh; and the BSE holiday
   literature). In-house: on ^NSEI 2007-2026 the day before a >=4-calendar-day
   market closure (i.e. a holiday block) averaged +0.356%/day with a 69% win
   rate and the first day back +0.305%/day — the strongest calendar effect in
   the Indian sample. (Same-day US effect is weaker: +0.128% pre-holiday.)
   Because the backtester fills at the next open, the entry signal fires the
   day BEFORE the pre-holiday day so the fill lands on the pre-holiday open;
   the exit fires on the post-holiday day, capturing both days' returns.
   -> ``pre_holiday_trend``.

4. Halloween effect (Nov-Apr). Bouman & Jacobsen (2002, "The Halloween
   indicator, 'Sell in May and go away': another puzzle", American Economic
   Review 92) found Nov-Apr beats May-Oct in 36 of 37 markets; Jacobsen &
   Visaltanachoti (2009) confirmed it for US sectors. In-house on SPY
   2003-2026 Nov-Apr compounded 4.31x vs 3.02x for May-Oct, and in the trailing
   5y the Nov-Apr+trend window captured 1.54x vs 1.23x gate-only. (The effect
   does NOT hold for India in-house: Nov-Apr was flat 2007-2026, so this
   strategy is targeted at the US.)
   -> ``nov_apr_trend``.

Causality notes: ``month``/``tdom``/``dteom``/``pre_holiday_buy``/``post_holiday``
are derived from the bar index (trading-calendar information, known ex-ante —
never from prices). ``vol_252`` is strictly trailing. ``pre_holiday_buy`` uses
the next bar's date, which is calendar information available at bar ``t``.

Entry/exit shape for every strategy:
    entry = <seasonal gate> and close > sma(close, 200)
    exit  = not(<seasonal gate>) or close < sma(close, 200)
with ``--hold`` as a time backstop. Suggested rotation: hold ~63 trading days
(quarterly) for the month-gated strategies, ~21 for the turn-of-month window,
~10 for the holiday strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_TREND_SMA = 200
_VOL_WINDOW = 252
_HOLIDAY_GAP_DAYS = 4  # a >=4-calendar-day trading gap implies a holiday block

# India strong months: April (fiscal-year start), July, November (Diwali
# season), December (year-end).
_STRONG_MONTHS = (4, 7, 11, 12)
# Halloween window: November through April (US evidence).
_NOV_APR_MONTHS = (11, 12, 1, 2, 3, 4)

ENTRY_SEASONAL = "strong_month == 1 and close > sma(close, {sma})"
EXIT_SEASONAL = "strong_month == 0 or close < sma(close, {sma})"

ENTRY_TOM = "tom_window == 1 and close > sma(close, {sma})"
EXIT_TOM = "tom_window == 0 or close < sma(close, {sma})"

ENTRY_PRE_HOLIDAY = "pre_holiday_buy == 1 and close > sma(close, {sma})"
EXIT_PRE_HOLIDAY = "post_holiday == 1 or close < sma(close, {sma})"

ENTRY_NOV_APR = "nov_apr == 1 and close > sma(close, {sma})"
EXIT_NOV_APR = "nov_apr == 0 or close < sma(close, {sma})"


def _trading_day_counters(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tdom, dteom)`` — trading day of month (1-based) and trading
    days left in the month (0 = last trading day) — as int arrays.

    Computed from the bar index only; both are calendar facts, not prices.
    """
    ym = idx.year * 100 + idx.month
    ym_codes, _ = pd.factorize(ym, sort=True)
    counts = np.bincount(ym_codes)
    month_len = counts[ym_codes]
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(len(idx)) - starts  # 0-based position within the month
    tdom = pos + 1
    dteom = month_len - tdom
    return tdom, dteom


def _holiday_columns(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pre_holiday_buy, post_holiday)`` 0/1 float arrays.

    ``pre_holiday_buy[t] == 1`` when bar ``t+1`` is a pre-holiday day, i.e. the
    gap in calendar days from bar ``t+1`` to the following trading bar is
    >= ``_HOLIDAY_GAP_DAYS`` (a weekend is only 3 days; 4+ implies a holiday).
    Firing the entry one bar early makes the next-open fill land on the
    pre-holiday day itself. ``post_holiday[t] == 1`` when bar ``t`` follows a
    >=4-day gap (first trading day back). Only the index dates are used, so
    both are knowable at bar ``t``.
    """
    dts = idx.to_series()
    next_gap = -dts.iloc[::-1].diff().dt.days.to_numpy()[::-1]  # days to next bar
    prev_gap = dts.diff().dt.days.to_numpy()  # days since previous bar
    # pre_holiday[t] = next_gap[t] >= 4; we need pre_holiday[t+1] at bar t.
    pre_holiday_buy = np.zeros(len(idx), dtype=float)
    pre_holiday_buy[:-1] = (next_gap[1:] >= _HOLIDAY_GAP_DAYS).astype(float)
    post_holiday = (prev_gap >= _HOLIDAY_GAP_DAYS).astype(float)
    post_holiday[0] = 0.0
    return pre_holiday_buy, post_holiday


def _prepare_calendar(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Add calendar gates, holiday flags and a low-vol rank to every frame.

    The same prepared bars serve all four strategies; each entry/exit
    expression references a different gate column.
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)

        month = idx.month.to_numpy()
        tdom, dteom = _trading_day_counters(idx)
        pre_holiday_buy, post_holiday = _holiday_columns(idx)

        frame["month"] = month.astype(float)
        frame["tdom"] = tdom.astype(float)
        frame["dteom"] = dteom.astype(float)
        frame["pre_holiday_buy"] = pre_holiday_buy
        frame["post_holiday"] = post_holiday
        frame["strong_month"] = np.isin(month, _STRONG_MONTHS).astype(float)
        frame["tom_window"] = ((dteom == 0) | (tdom <= 3)).astype(float)
        frame["nov_apr"] = np.isin(month, _NOV_APR_MONTHS).astype(float)

        # Low-volatility rank (Ang et al. 2006): calmest names first inside the
        # seasonal window. Strictly trailing; needs _VOL_WINDOW returns.
        returns = frame["close"].astype(float).pct_change()
        vol = returns.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW).std()
        frame["vol_252"] = vol
        frame["rank_score"] = -vol
        out[tv] = frame
    return out


def _lookback() -> int:
    # pct_change consumes one bar, then rolling std needs _VOL_WINDOW returns.
    return _VOL_WINDOW + 1


register_expression_strategy(
    "seasonal_strong_trend",
    entry=ENTRY_SEASONAL.format(sma=_TREND_SMA),
    exit=EXIT_SEASONAL.format(sma=_TREND_SMA),
    prepare_bars=_prepare_calendar,
    required_lookback=_lookback,
)

register_expression_strategy(
    "tom_window_trend",
    entry=ENTRY_TOM.format(sma=_TREND_SMA),
    exit=EXIT_TOM.format(sma=_TREND_SMA),
    prepare_bars=_prepare_calendar,
    required_lookback=_lookback,
)

register_expression_strategy(
    "pre_holiday_trend",
    entry=ENTRY_PRE_HOLIDAY.format(sma=_TREND_SMA),
    exit=EXIT_PRE_HOLIDAY.format(sma=_TREND_SMA),
    prepare_bars=_prepare_calendar,
    required_lookback=_lookback,
)

register_expression_strategy(
    "nov_apr_trend",
    entry=ENTRY_NOV_APR.format(sma=_TREND_SMA),
    exit=EXIT_NOV_APR.format(sma=_TREND_SMA),
    prepare_bars=_prepare_calendar,
    required_lookback=_lookback,
)
