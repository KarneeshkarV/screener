"""Event study of post-addition price drift for S&P 500 index additions.

Anchors each event to the effective "date added" from the current S&P 500
constituents table (see ``screener.universes.load_sp500_membership``) and
measures cumulative returns from ``PRE_EVENT_TRADING_DAYS`` trading days
before the addition to each horizon after it, minus the benchmark's return
over the same window (excess return).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from screener.backtester.data import PriceFetcher


BENCHMARK_SYMBOL = "SPY"
PRE_EVENT_TRADING_DAYS = 5
HORIZONS: tuple[int, ...] = (5, 20, 60)

LIMITATION_NOTE = (
    "Limitation: events are anchored to the effective 'date added' from the "
    "current S&P 500 constituents table, not the (earlier) announcement date, "
    "and removed ex-members are absent, so results carry survivorship bias."
)


@dataclass(frozen=True)
class InclusionEvent:
    """Per-event excess returns keyed by horizon (trading days after addition)."""

    symbol: str
    date_added: date
    excess: dict[int, float]


@dataclass(frozen=True)
class HorizonSummary:
    horizon: int
    mean: float
    median: float
    hit_rate: float


@dataclass(frozen=True)
class InclusionStudy:
    events: list[InclusionEvent]
    skipped: int
    horizons: tuple[int, ...]
    summaries: list[HorizonSummary]


def _event_excess(
    symbol_close: pd.Series,
    benchmark_close: pd.Series,
    added: date,
    pre_days: int,
    horizons: tuple[int, ...],
) -> dict[int, float] | None:
    """Excess return per horizon, or ``None`` when price data is insufficient.

    Trading days are counted on the intersection of the symbol's and the
    benchmark's calendars so both legs span the exact same sessions.
    """
    joint = symbol_close.index.intersection(benchmark_close.index)
    if joint.empty:
        return None
    sym = symbol_close.loc[joint]
    bench = benchmark_close.loc[joint]
    event_pos = int(joint.searchsorted(pd.Timestamp(added)))
    base_pos = event_pos - pre_days
    if base_pos < 0 or event_pos + max(horizons) >= len(joint):
        return None
    base_sym = float(sym.iloc[base_pos])
    base_bench = float(bench.iloc[base_pos])
    if base_sym <= 0 or base_bench <= 0:
        return None
    excess: dict[int, float] = {}
    for horizon in horizons:
        sym_ret = float(sym.iloc[event_pos + horizon]) / base_sym - 1.0
        bench_ret = float(bench.iloc[event_pos + horizon]) / base_bench - 1.0
        excess[horizon] = sym_ret - bench_ret
    return excess


def run_inclusion_study(
    membership: dict[str, date | None],
    fetcher: PriceFetcher,
    *,
    years: int = 5,
    as_of: date | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    pre_days: int = PRE_EVENT_TRADING_DAYS,
    benchmark: str = BENCHMARK_SYMBOL,
) -> InclusionStudy:
    """Run the post-addition drift event study over the trailing ``years``.

    Events without enough price data to cover the full window (``pre_days``
    trading days before the addition through ``max(horizons)`` after it) are
    skipped and counted in ``InclusionStudy.skipped``.
    """
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=round(365.25 * years))
    additions = sorted(
        (added, symbol)
        for symbol, added in membership.items()
        if added is not None and cutoff <= added <= as_of
    )
    if not additions:
        return InclusionStudy(events=[], skipped=0, horizons=horizons, summaries=[])

    # Calendar buffer wide enough to cover pre_days trading days before the
    # earliest addition (holidays included).
    start = additions[0][0] - timedelta(days=pre_days * 3 + 15)
    frames = fetcher.fetch(
        [symbol for _, symbol in additions] + [benchmark], start, as_of
    )
    bench_frame = frames.get(benchmark)
    if bench_frame is None or bench_frame.empty:
        raise ValueError(f"no price data for benchmark {benchmark}")
    bench_close = bench_frame["close"].astype(float)

    events: list[InclusionEvent] = []
    skipped = 0
    for added, symbol in additions:
        frame = frames.get(symbol)
        if frame is None or frame.empty or "close" not in frame.columns:
            skipped += 1
            continue
        excess = _event_excess(
            frame["close"].astype(float), bench_close, added, pre_days, horizons
        )
        if excess is None:
            skipped += 1
            continue
        events.append(InclusionEvent(symbol=symbol, date_added=added, excess=excess))

    summaries: list[HorizonSummary] = []
    for horizon in horizons:
        values = pd.Series([event.excess[horizon] for event in events], dtype=float)
        if values.empty:
            continue
        summaries.append(
            HorizonSummary(
                horizon=horizon,
                mean=float(values.mean()),
                median=float(values.median()),
                hit_rate=float((values > 0).mean()),
            )
        )
    return InclusionStudy(
        events=events, skipped=skipped, horizons=horizons, summaries=summaries
    )
