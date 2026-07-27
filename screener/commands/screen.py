"""Click command for the technical screener (TradingView or local bar store).

``screener screen`` runs the TradingView server-side scan by default; ``--source
local`` evaluates the same criteria over the local bar store (offline, intraday,
limited to stored symbols). ``screener screen live`` re-runs the local scanner
every ``--every`` window during the session and emits only new entrants/exits.
"""

from __future__ import annotations

from pathlib import Path

import click

from screener import history
from screener.criteria import CRITERIA
from screener.display import print_csv, print_results
from screener.local_scanner import LocalScanUnsupported
from screener.markets import market_option
from screener.scanner import scan
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    ScreenSource,
    run_screen_workflow,
)

# Intervals the local scanner serves from the stored 1m archive.
_LOCAL_INTERVALS = ["1m", "5m", "15m", "30m", "1h"]


@click.group(invoke_without_command=True)
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
@click.option(
    "--source",
    type=click.Choice([source.value for source in ScreenSource]),
    default=ScreenSource.TRADINGVIEW.value,
    show_default=True,
    help="Scan source: tradingview (daily, broad) or local (offline bar store).",
)
@click.option(
    "--interval",
    type=click.Choice(_LOCAL_INTERVALS),
    default="5m",
    show_default=True,
    help="Bar interval for --source local (served from the stored 1m archive).",
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
@click.pass_context
def screen(
    ctx: click.Context,
    market: str,
    criteria_names: tuple[str, ...],
    limit: int,
    order_by: str,
    source: str,
    interval: str,
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
    if ctx.invoked_subcommand is not None:
        return
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
        source=ScreenSource(source),
        interval=interval,
    )
    try:
        outcome = run_screen_workflow(request)
    except LocalScanUnsupported as exc:
        raise click.UsageError(
            f"criterion field {exc} is not available in --source local "
            "(fundamentals are TradingView-only)."
        ) from exc

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


@screen.command(name="live")
@market_option(default="us", help="Market to screen.")
@click.option(
    "-c",
    "--criteria",
    "criteria_names",
    type=click.Choice(list(CRITERIA)),
    multiple=True,
    default=("intraday_momentum",),
    help="Screening criteria (repeat to combine).",
)
@click.option(
    "--interval",
    type=click.Choice(_LOCAL_INTERVALS),
    default="5m",
    show_default=True,
    help="Bar interval evaluated each pass (served from the stored 1m archive).",
)
@click.option(
    "--every",
    default="5m",
    show_default=True,
    help="Cadence between passes, e.g. 30s, 5m, 15m.",
)
@click.option("-n", "--limit", default=50, help="Number of results per pass.")
@click.option("--sort", "order_by", default="volume", help="Sort by column each pass.")
@click.option(
    "--refresh-days",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Trailing 1m window refreshed before each pass (bars record --days).",
)
@click.option(
    "--max-passes",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Stop after N passes (0 = run until the session closes).",
)
def screen_live(
    market: str,
    criteria_names: tuple[str, ...],
    interval: str,
    every: str,
    limit: int,
    order_by: str,
    refresh_days: int,
    max_passes: int,
) -> None:
    """Re-evaluate the local scanner during the session, emitting only changes."""
    from screener.cache import parse_ttl
    from screener.screen_live import LiveRequest, run_screen_live

    every_seconds = parse_ttl(every, default=300.0) or 300.0
    request = LiveRequest(
        market=market,
        criteria_names=criteria_names,
        interval=interval,
        limit=int(limit),
        order_by=order_by,
        every_seconds=float(every_seconds),
        max_passes=int(max_passes),
        refresh_days=int(refresh_days),
    )
    try:
        session = run_screen_live(request)
    except LocalScanUnsupported as exc:
        raise click.UsageError(
            f"criterion field {exc} is not available for screen live "
            "(fundamentals are TradingView-only)."
        ) from exc

    if not session.passes:
        click.echo("Market is closed — no passes ran.")
        return
    for index, live_pass in enumerate(session.passes, start=1):
        click.echo(
            f"[{live_pass.run_ts}] pass {index}: {live_pass.total} matches, "
            f"showing {len(live_pass.df)}"
        )
        if live_pass.first_pass:
            click.echo("  baseline pass — no diff.")
            continue
        if not live_pass.added and not live_pass.removed:
            click.echo("  no changes.")
            continue
        if live_pass.added:
            click.echo(f"  + {', '.join(live_pass.added)}")
        if live_pass.removed:
            click.echo(f"  - {', '.join(live_pass.removed)}")


__all__ = [
    "history",
    "print_csv",
    "print_results",
    "scan",
    "screen",
    "screen_live",
]
