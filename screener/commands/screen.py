"""Click command for the TradingView-based technical screener."""

from __future__ import annotations

from pathlib import Path

import click

from screener.cache import parse_ttl
from screener import history
from screener.criteria import CRITERIA, registry as criteria_registry, resolve_criteria
from screener.display import print_csv, print_results
from screener.markets import market_option
from screener.scanner import scan
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    ScreenWorkflowDeps,
    ScreenWorkflowError,
    run_screen_workflow,
)


def _criteria_registry_patch_point() -> object:
    """Return the criteria registry kept here as a CLI Adapter patch point."""
    return criteria_registry


def _screen_workflow_deps() -> ScreenWorkflowDeps:
    from screener import reporting
    from screener.commands.screen_report import render_screen_report

    return ScreenWorkflowDeps(
        resolve_criteria=resolve_criteria,
        parse_cache_ttl=lambda value: parse_ttl(value, default=900),
        scan=scan,
        save_run=history.save_run,
        previous_run=history.previous_run,
        diff=history.diff,
        temp_report_path=lambda prefix: reporting.temp_report_path(prefix),
        render_report=render_screen_report,
    )


@click.command()
@market_option(
    default="us",
    help="Market to screen.",
)
@click.option(
    "-c",
    "--criteria",
    "criteria_names",
    type=click.Choice(list(CRITERIA.keys())),
    multiple=True,
    default=("ema",),
    help="Screening criteria (repeat to combine, e.g. -c ema -c breakout).",
)
@click.option("-n", "--limit", default=50, help="Number of results.")
@click.option(
    "--sort",
    "order_by",
    default="setup_score",
    help="Sort by column. Use setup_score for local composite ranking.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option(
    "--detail", is_flag=True, help="Show fundamental details (P/E, ROE, etc.)."
)
@click.option("--refresh", is_flag=True, help="Bypass cached TradingView data.")
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="TradingView cache TTL, e.g. 30s, 15m, 1h, off.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a static, self-contained HTML report to this file.",
)
@click.option(
    "--open-report",
    is_flag=True,
    default=False,
    help="Open the generated HTML report in the default browser.",
)
def screen(
    market: str,
    criteria_names: tuple[str, ...],
    limit: int,
    order_by: str,
    output_csv: bool,
    detail: bool,
    refresh: bool,
    cache_ttl: str,
    report_path: Path | None,
    open_report: bool,
) -> None:
    """Screen stocks based on technical criteria."""
    _criteria_registry_patch_point()
    request = ScreenRequest(
        market=market,
        criteria_names=criteria_names,
        limit=int(limit),
        order_by=order_by,
        output_csv=output_csv,
        detail=detail,
        refresh=refresh,
        cache_ttl=cache_ttl,
        report_path=report_path,
        open_report=open_report,
    )
    try:
        outcome = run_screen_workflow(request, _screen_workflow_deps())
    except ScreenWorkflowError as exc:
        raise click.UsageError(str(exc)) from exc

    if outcome.mode is ScreenMode.PIPELINE:
        return

    if outcome.df is None:  # pragma: no cover - defensive contract guard
        return

    if outcome.mode is ScreenMode.CSV:
        print_csv(outcome.df)
        return

    print_results(
        outcome.df,
        outcome.total,
        outcome.market,
        outcome.label,
        added=list(outcome.added),
        removed=list(outcome.removed),
        first_run=outcome.first_run,
    )
    click.echo(f"Report: {outcome.report_path}")
    if open_report and outcome.report_path is not None:
        from screener.reporting import open_report as open_report_file

        open_report_file(outcome.report_path)


__all__ = [
    "criteria_registry",
    "history",
    "print_csv",
    "print_results",
    "scan",
    "screen",
]
