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

All three divide one close by another, so all three take the tradeability gate
in :mod:`screener.tradeable`: an untraded sub-cent stub as the denominator
turns a dormant shell into the strongest name in the universe. Each returns
NaN (or ``None``, for the scalar spread) where the window is not a market,
which every caller already reads as "no reading for this bar" rather than as a
zero. ``volume`` is optional on each, and without it only the price floor
applies - pass it whenever the frame has it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from screener.tradeable import tradeable_window

RS_RATIO_WINDOW = 55
RS_RANK_WINDOW = 252
RS_SPREAD_WINDOW = 63


def relative_strength_ratio(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    stock_volume: pd.Series | None = None,
    window: int = RS_RATIO_WINDOW,
) -> pd.Series:
    """Percent by which the stock's ``window``-bar return ratio beats the benchmark.

    Aligned on the intersection of the two indices, so a stock that did not
    trade on a benchmark session simply has no reading for that bar. NaN where
    the stock's own leg is not a tradeable window; the benchmark is an index
    and needs no gate.
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
    volume = None if stock_volume is None else stock_volume.reindex(aligned.index)
    rs = rs.where(tradeable_window(aligned["stock"], volume, lookback=window))
    rs.name = f"rs_{window}"
    return rs


def relative_strength_rank(
    closes_by_symbol: Mapping[str, pd.Series],
    *,
    volumes_by_symbol: Mapping[str, pd.Series] | None = None,
    window: int = RS_RANK_WINDOW,
) -> pd.DataFrame:
    """Cross-sectional percentile, 0-100, of each symbol's ``window``-bar return.

    Returns a ``bar x symbol`` frame over the union of the inputs' indices.
    Symbols with no usable close series are dropped; an empty mapping yields an
    empty frame.

    This is the measure the gate matters most for. A percentile is
    *competitive*: an untradeable name does not merely carry a wrong number of
    its own, it takes a slot at the top and pushes every real name down a rank.
    Observed on cached US bars, the twelve highest ``rs_rank`` values in a
    2,261-name field were all stub-priced shells, led by a 0.006-to-5.35 line
    reporting +89,067%. Ungated returns are dropped to NaN *before* ranking, so
    they neither score nor crowd; ``pandas`` excludes NaN from a rank, which
    means the surviving names are ranked against each other alone.
    """
    returns: dict[str, pd.Series] = {}
    for symbol, close in closes_by_symbol.items():
        if close is None or close.empty:
            continue
        values = close.astype(float)
        raw = values / values.shift(window) - 1.0
        volume = None if volumes_by_symbol is None else volumes_by_symbol.get(symbol)
        if volume is not None:
            volume = volume.reindex(values.index)
        returns[symbol] = raw.where(tradeable_window(values, volume, lookback=window))
    if not returns:
        return pd.DataFrame()
    return pd.DataFrame(returns).rank(axis=1, pct=True) * 100.0


def relative_strength_spread(
    stock_close: pd.Series,
    benchmark_close: pd.Series | None,
    *,
    stock_volume: pd.Series | None = None,
    window: int = RS_SPREAD_WINDOW,
) -> float | None:
    """Excess return over the benchmark across ``window`` bars, in points.

    ``None`` when there is no benchmark, not enough overlapping history, or the
    stock's own window is not a tradeable one. Callers must treat all three as
    "unknown" rather than as zero excess return - :mod:`screener.conviction`
    already renormalizes its pillar score when this is ``None``, which is the
    correct handling for a stub-priced name too.
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
    volume = None if stock_volume is None else stock_volume.reindex(aligned.index)
    gate = tradeable_window(aligned.iloc[:, 0], volume, lookback=window)
    if not bool(gate.iloc[-1]):
        return None
    stock_ret = (aligned.iloc[-1, 0] / aligned.iloc[-1 - window, 0] - 1.0) * 100.0
    bench_ret = (aligned.iloc[-1, 1] / aligned.iloc[-1 - window, 1] - 1.0) * 100.0
    return float(cast(Any, stock_ret - bench_ret))
