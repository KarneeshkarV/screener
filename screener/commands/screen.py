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
from screener.backtester.models import SUPPORTED_INTERVALS
from screener.gate_options import gate_options, gate_overrides
from screener.screen_candidates import (
    DEFAULT_INTERVAL,
    IntervalNotScreenableError,
    UnscreenableStrategyError,
)
from screener.scoring import IncompatibleScorerBlendError, PriceAdjustment


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
@click.option(
    "--universe",
    default=None,
    help=(
        "Screen a named universe (nifty50, nifty500, sensex, sp500) or a "
        "universe file, with no TradingView prefilter. This is the exact path: "
        "the answer depends on local bars only. Slower, because every name is "
        "fetched. Needs a criterion that names a strategy."
    ),
)
@click.option("-n", "--limit", default=50, help="Number of results.")
@click.option(
    "--sort",
    "order_by",
    default="setup_score",
    help=(
        "Sort by column. setup_score ranks by the active criteria's philosophy "
        "score (e.g. trend for ema, cheapness for value); other names are "
        "TradingView columns such as volume."
    ),
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
@gate_options()
@click.option(
    "--interval",
    type=click.Choice(list(SUPPORTED_INTERVALS)),
    default=DEFAULT_INTERVAL,
    show_default=True,
    help=(
        "Bar interval for --universe screens. Names the last completed bar of "
        "the as-of date. Refused for strategies that read fundamentals or use "
        "--earnings-blackout, which are dated to a day."
    ),
)
@click.option(
    "--max-universe",
    type=int,
    default=0,
    help="Cap the field before bars are fetched. Pass 0 to disable.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
    help=(
        "Price adjustment for bar-derived ranking scores. Same flag as the "
        "backtester's --price-adjustment. full=yfinance auto_adjust=True; "
        "splits_only=split-adjust OHLC and credit dividends as cash; none=raw "
        "OHLC. Run a backtest with the same value so the scores match."
    ),
)
def screen(
    market: str,
    criteria_names: tuple[str, ...],
    universe: str | None,
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
    min_price: float | None,
    min_avg_dollar_volume: float | None,
    adv_window: int,
    regime_filter_args: tuple[str, ...],
    sector_neutral: bool,
    earnings_blackout_days: int | None,
    min_score: float | None,
    interval: str,
    max_universe: int,
    price_adjustment: PriceAdjustment,
) -> None:
    """Screen stocks based on technical criteria."""
    if earnings_buffer is not None and earnings_buffer < 0:
        raise click.UsageError("--earnings-buffer must be >= 0.")
    if max_universe < 0:
        raise click.UsageError("--max-universe must be >= 0.")
    request = ScreenRequest(
        market=market,
        criteria_names=criteria_names,
        universe=universe,
        limit=int(limit),
        order_by=order_by,
        output_csv=output_csv,
        detail=detail,
        refresh=refresh,
        cache_ttl=cache_ttl,
        report_path=report_path,
        open_report=open_report,
        # Print the table first, then render. The report takes about 0.4s that
        # the terminal spent waiting on a result it already had.
        defer_report=True,
        earnings=earnings,
        earnings_buffer=earnings_buffer,
        price_adjustment=price_adjustment,
        # The same builder ``backtest-rolling`` uses, so a gate flag typed here
        # resolves to exactly the gate it resolves to there.
        gate_overrides=gate_overrides(
            min_price=min_price,
            min_avg_dollar_volume=min_avg_dollar_volume,
            adv_window=adv_window,
            regime_filter_args=regime_filter_args,
            earnings_blackout_days=earnings_blackout_days,
            sector_neutral=sector_neutral,
            min_score=min_score,
        ),
        interval=interval,
        max_universe=int(max_universe),
    )
    try:
        outcome = run_screen_workflow(request)
    except (
        IncompatibleScorerBlendError,
        IntervalNotScreenableError,
        UnscreenableStrategyError,
    ) as exc:
        raise click.UsageError(str(exc)) from exc

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
    if outcome.render_report is not None:
        outcome.render_report()
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
