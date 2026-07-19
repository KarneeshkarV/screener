"""Click command for the TradingView-based technical screener."""

from __future__ import annotations

from pathlib import Path

import click

from screener import history
from screener.criteria import CRITERIA
from screener.display import print_csv, print_results
from screener.markets import market_option
from screener.scanner import scan
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    run_screen_workflow,
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
    type=click.Choice(list(CRITERIA)),
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
@click.option(
    "--earnings",
    is_flag=True,
    help="Attach days_to_earnings to final result rows.",
)
@click.option(
    "--earnings-buffer",
    type=int,
    default=None,
    help=(
        "Drop result rows whose next earnings date is within N calendar days. "
        "Rows with unknown earnings dates are kept. This also enables earnings "
        "enrichment."
    ),
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
    earnings: bool,
    earnings_buffer: int | None,
) -> None:
    """Screen stocks based on technical criteria."""
    if earnings_buffer is not None and earnings_buffer < 0:
        raise click.UsageError("--earnings-buffer must be >= 0.")
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
        earnings=earnings,
        earnings_buffer=earnings_buffer,
    )
    outcome = run_screen_workflow(request)

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
    "history",
    "print_csv",
    "print_results",
    "scan",
    "screen",
]
