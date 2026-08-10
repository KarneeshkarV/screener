"""Screen workflow Module behind the Click Adapter.

The workflow calls its collaborators (``scan``, history persistence, report
rendering, earnings enrichment) directly as module-level names. Tests exercise
edge cases by monkeypatching those names — the seams that actually touch the
network, the history DB, or the filesystem — rather than threading a bag of
injected callables through every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from screener.cache import parse_ttl
from screener.commands.screen_report import render_screen_report
from screener.criteria import resolve_criteria
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.history import diff, previous_run, save_run
from screener.reporting import temp_report_path
from screener.scanner import scan
from screener.scoring import resolve_scorer


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


def run_screen_workflow(request: ScreenRequest) -> ScreenOutcome:
    """Run the full non-Click screen lifecycle and return its outcome."""
    selection = resolve_criteria(request.criteria_names)
    scorer = resolve_scorer(request.criteria_names, strict=False)

    total, df = scan(
        market=request.market,
        filters=selection.filters,
        limit=request.limit,
        order_by=request.order_by,
        detail=request.detail,
        cache_ttl=parse_ttl(request.cache_ttl, default=900),
        refresh=request.refresh,
        scorer=scorer,
    )

    # Earnings enrichment is opt-in and runs only on final result rows.
    if request.earnings or request.earnings_buffer is not None:
        df = enrich_days_to_earnings(df, request.market)
    if request.earnings_buffer is not None:
        df = filter_earnings_buffer(df, request.earnings_buffer)

    if request.output_csv:
        return ScreenOutcome(
            mode=ScreenMode.CSV,
            market=request.market,
            label=selection.label,
            total=total,
            df=df,
        )

    run_id = save_run(request.market, selection.label, total, df)
    prev = previous_run(request.market, selection.label, run_id)
    if prev is None:
        added: list[str] = []
        removed: list[str] = []
        first_run = True
    else:
        added, removed = diff(df, prev)
        first_run = False

    # HTML report is opt-in: only when --report path and/or --open-report.
    generated_report: Path | None = None
    want_report = request.report_path is not None or request.open_report
    if want_report:
        generated_report = request.report_path
        if generated_report is None:
            generated_report = temp_report_path("screen")

        render_screen_report(
            df,
            total,
            request.market,
            selection.label,
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
        label=selection.label,
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
