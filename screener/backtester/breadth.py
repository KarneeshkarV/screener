"""Historical market breadth computed from the panel's own bars.

The live ``market-condition`` command reads breadth from a TradingView snapshot,
which has no history. A backtest needs the same measurement on every past date,
so this module derives it from the bars the simulation already loaded: for each
date, the share of the universe whose close sits above its own 20-day and
200-day EMA.

Strictly point-in-time. Each date's EMA uses only closes up to and including
that date, and a ticker contributes to a date only if it has a bar on that date
and enough prior history for the EMA to be defined — so no name is counted
before it started trading, and none is carried forward after it stopped.

The denominator is the tickers actually measurable that day, not the nominal
universe size. Dates whose coverage falls under :data:`MIN_COVERAGE_TICKERS` are
labeled ``unknown`` rather than reported as a percentage of a handful of names.
"""

from __future__ import annotations

import pandas as pd

from screener.regime import classify_breadth_series

FAST_EMA_SPAN = 20
SLOW_EMA_SPAN = 200

# Below this many measurable tickers the share is too noisy to act on. Warmup
# dates at the very start of a fetch window are the usual cause.
MIN_COVERAGE_TICKERS = 30


def _above_ema_counts(
    bars_by_tv: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    span: int,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(above, measurable)`` per-date counts for one EMA span."""
    above = pd.Series(0, index=index, dtype="int64")
    measurable = pd.Series(0, index=index, dtype="int64")

    for bars in bars_by_tv.values():
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        close = pd.to_numeric(bars["close"], errors="coerce").astype(float)
        if close.index.has_duplicates:
            close = close[~close.index.duplicated(keep="last")]
        # min_periods=span keeps the EMA undefined until the span is genuinely
        # covered; pandas would otherwise emit a value from the first bar.
        ema = close.ewm(span=span, min_periods=span, adjust=False).mean()

        # Reindex without filling: a date the ticker did not trade is not an
        # observation, and forward-filling would let a delisted name keep
        # voting on breadth long after its last bar.
        close_on = close.reindex(index)
        ema_on = ema.reindex(index)

        valid = close_on.notna() & ema_on.notna()
        measurable = measurable.add(valid.astype("int64"), fill_value=0)
        above = above.add((valid & (close_on > ema_on)).astype("int64"), fill_value=0)

    return above, measurable


def breadth_percentages(
    bars_by_tv: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(pct_above_20ema, pct_above_200ema)`` as 0-100 series.

    Dates with fewer than :data:`MIN_COVERAGE_TICKERS` measurable names are NaN.
    """
    index = pd.DatetimeIndex(index).sort_values()

    percentages: list[pd.Series] = []
    for span in (FAST_EMA_SPAN, SLOW_EMA_SPAN):
        above, measurable = _above_ema_counts(bars_by_tv, index, span)
        covered = measurable >= MIN_COVERAGE_TICKERS
        pct = pd.Series(float("nan"), index=index, dtype=float)
        pct[covered] = above[covered] / measurable[covered] * 100.0
        percentages.append(pct)

    return percentages[0], percentages[1]


def breadth_regime_series(
    bars_by_tv: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Label every date in ``index`` with its breadth regime.

    Labels come from :data:`screener.regime.BREADTH_LABELS`; dates without
    enough coverage are ``unknown``.
    """
    pct_20, pct_200 = breadth_percentages(bars_by_tv, index)
    return classify_breadth_series(pct_20, pct_200)


__all__ = [
    "FAST_EMA_SPAN",
    "MIN_COVERAGE_TICKERS",
    "SLOW_EMA_SPAN",
    "breadth_percentages",
    "breadth_regime_series",
]
