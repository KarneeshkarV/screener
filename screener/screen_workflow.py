"""Screen workflow Module behind the Click Adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from screener import history
from screener.cache import parse_ttl
from screener.commands.screen_report import render_screen_report
from screener.criteria import (
    FilterCriteriaSelection,
    resolve_criteria,
)
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.reporting import temp_report_path
from screener.scanner import scan


class ScanFn(Protocol):
    def __call__(
        self,
        *,
        market: str,
        filters: list[Any],
        limit: int,
        order_by: str,
        detail: bool,
        cache_ttl: float | None,
        refresh: bool,
    ) -> tuple[int, pd.DataFrame]: ...


class RenderReportFn(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        total: int,
        market: str,
        criteria_name: str,
        output_file: str | Path,
        *,
        added: Sequence[str] = (),
        removed: Sequence[str] = (),
        first_run: bool = False,
        detail: bool = False,
        refresh: bool = False,
        cache_ttl: str = "",
        order_by: str = "",
    ) -> Path: ...


class EnrichDaysToEarningsFn(Protocol):
    def __call__(self, df: pd.DataFrame, market: str) -> pd.DataFrame: ...


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


@dataclass(frozen=True)
class ScreenWorkflowDeps:
    resolve_criteria: Callable[[Sequence[str]], FilterCriteriaSelection]
    parse_cache_ttl: Callable[[str], float | None]
    scan: ScanFn
    save_run: Callable[[str, str, int, pd.DataFrame], int]
    previous_run: Callable[[str, str, int], pd.DataFrame | None]
    diff: Callable[[pd.DataFrame, pd.DataFrame], tuple[list[str], list[str]]]
    temp_report_path: Callable[[str], Path]
    render_report: RenderReportFn
    enrich_days_to_earnings: EnrichDaysToEarningsFn


def default_screen_workflow_deps() -> ScreenWorkflowDeps:
    """Build default dependencies for the screen workflow Interface."""
    return ScreenWorkflowDeps(
        resolve_criteria=resolve_criteria,
        parse_cache_ttl=lambda value: parse_ttl(value, default=900),
        scan=scan,
        save_run=history.save_run,
        previous_run=history.previous_run,
        diff=history.diff,
        temp_report_path=temp_report_path,
        render_report=render_screen_report,
        enrich_days_to_earnings=enrich_days_to_earnings,
    )


def run_screen_workflow(
    request: ScreenRequest,
    deps: ScreenWorkflowDeps | None = None,
) -> ScreenOutcome:
    """Run the full non-Click screen lifecycle and return its outcome."""
    deps = deps or default_screen_workflow_deps()
    selection = deps.resolve_criteria(request.criteria_names)

    total, df = deps.scan(
        market=request.market,
        filters=selection.filters,
        limit=request.limit,
        order_by=request.order_by,
        detail=request.detail,
        cache_ttl=deps.parse_cache_ttl(request.cache_ttl),
        refresh=request.refresh,
    )

    # Earnings enrichment is opt-in and runs only on final result rows.
    if request.earnings or request.earnings_buffer is not None:
        df = deps.enrich_days_to_earnings(df, request.market)
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

    run_id = deps.save_run(request.market, selection.label, total, df)
    prev = deps.previous_run(request.market, selection.label, run_id)
    if prev is None:
        added: list[str] = []
        removed: list[str] = []
        first_run = True
    else:
        added, removed = deps.diff(df, prev)
        first_run = False

    generated_report = request.report_path
    if generated_report is None:
        generated_report = deps.temp_report_path("screen")

    deps.render_report(
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
    "ScreenWorkflowDeps",
    "default_screen_workflow_deps",
    "run_screen_workflow",
]
