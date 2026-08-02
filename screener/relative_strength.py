"""The three distinct measures this package calls "relative strength".

Three screens each compute something named RS, and the three are *not* the
same measure. They used to live one per screen, each named some variant of
"relative strength", which made them look interchangeable when they are not:

* :func:`relative_strength_ratio` - the RS-breakout scan's 55-bar ratio of the
  stock's return to the benchmark's, as a percent. Per bar, benchmark
  relative, one symbol at a time.
* :func:`relative_strength_rank` - Minervini's 252-bar cross-sectional
  percentile, 0-100. Per bar, *universe* relative, so it needs every symbol at
  once and has no meaning for a single ticker.
* :func:`relative_strength_spread` - the conviction card's 63-bar excess
  return over the benchmark, in percentage points. One scalar at the last bar.

They are deliberately kept separate: a ratio, a percentile and a spread answer
different questions over different windows. The defect being fixed here is
that they shared a word, not that there were three of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd


RS_RATIO_WINDOW = 55
RS_RANK_WINDOW = 252
RS_SPREAD_WINDOW = 63


def relative_strength_ratio(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    window: int = RS_RATIO_WINDOW,
) -> pd.Series:
    """Percent by which the stock's ``window``-bar return ratio beats the benchmark.

    Aligned on the intersection of the two indices, so a stock that did not
    trade on a benchmark session simply has no reading for that bar.
    """
    aligned = pd.concat(
        [stock_close.astype(float), benchmark_close.astype(float)],
        axis=1,
        join="inner",
    ).dropna()
    aligned.columns = ["stock", "benchmark"]
    stock_ret = aligned["stock"] / aligned["stock"].shift(window)
    bench_ret = aligned["benchmark"] / aligned["benchmark"].shift(window)
    rs = ((stock_ret / bench_ret) - 1.0) * 100.0
    rs.name = f"rs_{window}"
    return rs


def relative_strength_rank(
    closes_by_symbol: Mapping[str, pd.Series],
    *,
    window: int = RS_RANK_WINDOW,
) -> pd.DataFrame:
    """Cross-sectional percentile, 0-100, of each symbol's ``window``-bar return.

    Returns a ``bar x symbol`` frame over the union of the inputs' indices.
    Symbols with no usable close series are dropped; an empty mapping yields an
    empty frame.
    """
    returns: dict[str, pd.Series] = {}
    for symbol, close in closes_by_symbol.items():
        if close is None or close.empty:
            continue
        values = close.astype(float)
        returns[symbol] = values / values.shift(window) - 1.0
    if not returns:
        return pd.DataFrame()
    return pd.DataFrame(returns).rank(axis=1, pct=True) * 100.0


def relative_strength_spread(
    stock_close: pd.Series,
    benchmark_close: pd.Series | None,
    *,
    window: int = RS_SPREAD_WINDOW,
) -> float | None:
    """Excess return over the benchmark across ``window`` bars, in points.

    ``None`` when there is no benchmark or not enough overlapping history,
    which callers must treat as "unknown" rather than as zero excess return.
    """
    if benchmark_close is None or benchmark_close.empty:
        return None
    aligned = pd.concat(
        [stock_close.astype(float), benchmark_close.astype(float)],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) <= window:
        return None
    stock_ret = (aligned.iloc[-1, 0] / aligned.iloc[-1 - window, 0] - 1.0) * 100.0
    bench_ret = (aligned.iloc[-1, 1] / aligned.iloc[-1 - window, 1] - 1.0) * 100.0
    return float(cast(Any, stock_ret - bench_ret))
