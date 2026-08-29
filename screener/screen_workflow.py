"""Screen workflow Module behind the Click Adapter.

The workflow calls its collaborators (``scan``, history persistence, report
rendering, earnings enrichment) directly as module-level names. Tests exercise
edge cases by monkeypatching those names — the seams that actually touch the
network, the history DB, or the filesystem — rather than threading a bag of
injected callables through every call site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from screener.cache import parse_ttl
from screener.criteria import resolve_criteria
from screener.screen_candidates import (
    ScreenStrategy,
    UnscreenableStrategyError,
    prefilter_filters,
    resolve_screen_strategy,
    resolve_universe_tickers,
    screen_candidates,
    screen_label,
)
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.history import diff, previous_run, save_run
from screener.scanner import scan
from screener.scoring import (
    DEFAULT_PRICE_ADJUSTMENT,
    OUTPUT_SCORE_COLUMN,
    PriceAdjustment,
    resolve_scorer,
)


LOG = logging.getLogger(__name__)


def temp_report_path(prefix: str) -> Path:
    """Lazy wrapper so the CSV path skips reporting helpers."""
    from screener.reporting import temp_report_path as _impl

    return _impl(prefix)


def render_screen_report(*args: Any, **kwargs: Any) -> Path:
    """Lazy wrapper: plotly lives behind screen_report, only imported on use."""
    from screener.commands.screen_report import render_screen_report as _impl

    return _impl(*args, **kwargs)


class ScreenMode(str, Enum):
    CSV = "csv"
    RESULTS = "results"


@dataclass(frozen=True)
class ScreenRequest:
    market: str
    criteria_names: tuple[str, ...]
    limit: int
    order_by: str
    output_csv: bool
    detail: bool
    refresh: bool
    cache_ttl: str
    report_path: Path | None
    open_report: bool = False
    earnings: bool = False
    # Drop final result rows whose next earnings date is within N calendar days.
    # ``None`` disables the filter. Unknown earnings dates are always kept.
    earnings_buffer: int | None = None
    # Price adjustment for bar-derived ranking scores. Same spelling as the
    # backtester's ``--price-adjustment``. Snapshot scorers ignore it.
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT
    # Named universe or universe file for the exact path (D9). ``None`` keeps
    # the TradingView prefilter, which is the default. Only a criterion that
    # aliases a strategy has a bar rule to run, so this is refused otherwise.
    universe: str | None = None


@dataclass(frozen=True)
class ScreenOutcome:
    mode: ScreenMode
    market: str
    label: str
    total: int
    df: pd.DataFrame
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    first_run: bool = False
    report_path: Path | None = None


def _run_bar_screen(
    request: ScreenRequest,
    strategy: ScreenStrategy,
) -> tuple[int, pd.DataFrame]:
    """Screen by ``strategy``'s entry rule instead of by TradingView filters.

    Two modes, differing only in where the field comes from. With
    ``--universe`` the names come from ``screener.universes`` and no vendor
    field is consulted at all. Without it the TradingView prefilter narrows the
    field first, which is only sound because a prefilter may not drop a name
    the bar rule would have kept - the property
    ``tests/correctness`` pins.
    """
    warnings: list[str] = []
    as_of = date.today()

    if request.universe:
        tickers = resolve_universe_tickers(request.universe, request.market)
        scanned = None
        total = len(tickers)
    else:
        total, scanned = scan(
            market=request.market,
            filters=prefilter_filters(strategy),
            limit=request.limit,
            order_by="volume",
            detail=request.detail,
            cache_ttl=parse_ttl(request.cache_ttl, default=900),
            refresh=request.refresh,
            scorer=None,
            price_adjustment=request.price_adjustment,
        )
        tickers = [str(t) for t in scanned.get("ticker", pd.Series(dtype=str))]

    df = screen_candidates(
        strategy,
        market=request.market,
        tickers=tickers,
        as_of=as_of,
        scanned=scanned,
        limit=request.limit,
        refresh=request.refresh,
        price_adjustment=request.price_adjustment,
        warnings=warnings,
    )
    for warning in warnings:
        LOG.warning("%s", warning)
    return total, df


def run_screen_workflow(request: ScreenRequest) -> ScreenOutcome:
    """Run the full non-Click screen lifecycle and return its outcome."""
    strategy = resolve_screen_strategy(request.criteria_names)
    if request.universe and strategy is None:
        raise UnscreenableStrategyError(
            f"--universe needs a criterion that names a strategy, because only "
            f"a strategy carries a bar rule to evaluate; {request.criteria_names} "
            "names TradingView filters only, which have nothing to run against "
            "a local universe."
        )
    if strategy is not None:
        label = screen_label(
            request.criteria_names, strategy=strategy, universe=request.universe
        )
        return _finish_screen(request, label, *_run_bar_screen(request, strategy))

    selection = resolve_criteria(request.criteria_names)
    # Only the ``setup_score`` ranking consumes a scorer, and resolving one can
    # refuse a criteria combination whose scores are incomparable. Skip the
    # resolution when the run sorts by a TradingView column, so a refusal fires
    # only for a run that would actually rank by the refused recipe.
    scorer = (
        resolve_scorer(request.criteria_names, strict=False)
        if request.order_by == OUTPUT_SCORE_COLUMN
        else None
    )

    total, df = scan(
        market=request.market,
        filters=selection.filters,
        limit=request.limit,
        order_by=request.order_by,
        detail=request.detail,
        cache_ttl=parse_ttl(request.cache_ttl, default=900),
        refresh=request.refresh,
        scorer=scorer,
        price_adjustment=request.price_adjustment,
    )

    return _finish_screen(request, selection.label, total, df)


def _finish_screen(
    request: ScreenRequest,
    label: str,
    total: int,
    df: pd.DataFrame,
) -> ScreenOutcome:
    """Everything after the candidate set is decided: enrich, persist, report.

    Shared by both paths on purpose. The two paths differ in how a name is
    selected and in nothing else, so the earnings filter, the history diff and
    the report must not be written twice.
    """
    # Earnings enrichment is opt-in and runs only on final result rows.
    if request.earnings or request.earnings_buffer is not None:
        df = enrich_days_to_earnings(df, request.market)
    if request.earnings_buffer is not None:
        df = filter_earnings_buffer(df, request.earnings_buffer)

    if request.output_csv:
        return ScreenOutcome(
            mode=ScreenMode.CSV,
            market=request.market,
            label=label,
            total=total,
            df=df,
        )

    run_id = save_run(request.market, label, total, df)
    prev = previous_run(request.market, label, run_id)
    if prev is None:
        added: list[str] = []
        removed: list[str] = []
        first_run = True
    else:
        added, removed = diff(df, prev)
        first_run = False

    # Non-CSV always writes an HTML report (temp path when --report omitted).
    # render_screen_report / temp_report_path are lazy wrappers so the CSV path
    # (above) never imports plotly.
    generated_report = request.report_path
    if generated_report is None:
        generated_report = temp_report_path("screen")

    render_screen_report(
        df,
        total,
        request.market,
        label,
        generated_report,
        added=added,
        removed=removed,
        first_run=first_run,
        detail=request.detail,
        refresh=request.refresh,
        cache_ttl=request.cache_ttl,
        order_by=request.order_by,
    )

    return ScreenOutcome(
        mode=ScreenMode.RESULTS,
        market=request.market,
        label=label,
        total=total,
        df=df,
        added=tuple(added),
        removed=tuple(removed),
        first_run=first_run,
        report_path=generated_report,
    )


__all__ = [
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "run_screen_workflow",
]
