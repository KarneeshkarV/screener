"""Rolling backtest CLI and compatibility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from screener import agentio
from screener.backtester.cli_common import (
    backtest_options,
    intraday_options,
    resolve_report_path,
    sizing_options,
    write_tearsheet,
)
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run
from screener.markets import market_option
from screener.regime import TREND_LABELS
from screener.universes import available_universes

__all__ = ["backtest_rolling"]


@click.command(name="backtest-rolling")
@market_option(
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--years",
    type=int,
    default=1,
    show_default=True,
    help="Trailing calendar years when --start is omitted.",
)
@backtest_options("rolling", "hold", "top", "entry", "exit", "strategy")
@click.option(
    "--universe",
    type=str,
    default=None,
    help=(
        "Named universe (built-ins: "
        + ", ".join((*available_universes(), "dynamic"))
        + "). Defaults to the market index."
    ),
)
@click.option(
    "--universe-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="TOML/YAML/JSON definitions for custom static, snapshot, or dynamic universes.",
)
@click.option(
    "--dynamic-base",
    default=None,
    help="Candidate index for --universe dynamic (default: sp500 or nifty500).",
)
@click.option(
    "--universe-size",
    type=int,
    default=100,
    show_default=True,
    help="Number of highest lagged-ADV names in a dynamic universe.",
)
@click.option(
    "--universe-lookback",
    type=int,
    default=60,
    show_default=True,
    help="Trailing bars used for dynamic-universe ADV ranking.",
)
@click.option(
    "--universe-rebalance",
    type=click.Choice(["daily", "weekly", "monthly", "quarterly"]),
    default="monthly",
    show_default=True,
    help="Dynamic-universe membership refresh frequency.",
)
@click.option(
    "--no-universe-cache",
    is_flag=True,
    default=False,
    help="Force live constituent refresh instead of today's cache.",
)
@click.option(
    "--point-in-time",
    is_flag=True,
    default=False,
    help=(
        "Require point-in-time membership. Custom snapshot universes use full "
        "membership windows; sp500 uses its available historical additions."
    ),
)
@backtest_options(
    "rolling",
    "tickers",
    "universe-file",
    "max-universe",
    "stop-loss",
    "take-profit",
    "trailing-stop",
    "slippage-bps",
    "commission-bps",
    "cost-model",
)
@click.option(
    "--spread-proxy",
    is_flag=True,
    default=False,
    help=(
        "Estimate per-fill half-spread via Corwin-Schultz (2012) high-low "
        "estimator and charge it as slippage (on top of --slippage-model)."
    ),
)
@backtest_options(
    "rolling",
    "initial-capital",
    "benchmark",
    "min-price",
    "min-avg-dollar-volume",
    "adv-window",
    "slippage-model",
    "half-spread-bps",
    "vol-impact-k",
    "no-gap-fills",
    "entry-order",
    "entry-limit-bps",
    "partial-exit",
    "price-adjustment",
    "interval",
)
@click.option(
    "--regime-filter",
    "regime_filter_args",
    multiple=True,
    type=click.Choice(list(TREND_LABELS)),
    help=(
        "Only allow entries on days whose benchmark trend regime matches "
        "(repeatable). Warmup days with an unknown regime are suppressed."
    ),
)
@click.option(
    "--sector-neutral",
    is_flag=True,
    default=False,
    help=(
        "Z-score rank_score within each sector group per day before ranking "
        "(factor strategies only; no-op when no rank_score column exists)."
    ),
)
@click.option(
    "--earnings-blackout",
    "earnings_blackout_days",
    type=int,
    default=None,
    help=(
        "Suppress entry signals within N calendar days before (and including) "
        "a known earnings date for each ticker. Tickers with no known earnings "
        "dates remain eligible (a warning is recorded)."
    ),
)
@click.option(
    "--fundamentals-provider",
    type=click.Choice(["fmp", "openscreener", "yfinance"]),
    default=None,
    help="Merge dated fundamentals into rolling backtest bars (US: fmp, India: openscreener or yfinance).",
)
@click.option(
    "--fundamental-field",
    "fundamental_field_args",
    multiple=True,
    help=(
        "Fundamental field to fetch for expressions (repeatable). Defaults to "
        "pe_ttm, pb_ttm, roe_ttm, debt_to_equity, revenue_growth_yoy, "
        "eps_growth_yoy, revenue_up_3q, market_cap."
    ),
)
@click.option(
    "--fundamental-lag-days",
    type=int,
    default=None,
    help="Calendar-day lag applied to fundamental effective dates (defaults: fmp=1, openscreener=60).",
)
@backtest_options("rolling", "csv", "report", "open-report")
@click.option(
    "--dashboard",
    is_flag=True,
    default=False,
    help="Render and serve a local interactive dashboard for this run.",
)
@click.option(
    "--dashboard-port",
    type=int,
    default=8765,
    show_default=True,
    help="Local port used when --dashboard is enabled.",
)
@click.option(
    "--dashboard-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(".screener/dashboards"),
    show_default=True,
    help="Directory for generated dashboard HTML files.",
)
@intraday_options  # type: ignore[untyped-decorator]
@sizing_options  # type: ignore[untyped-decorator]
def backtest_rolling(**params: Any) -> None:
    """Run a true daily rolling backtest over a date window."""
    ctx = click.get_current_context()
    request = BacktestRequest(
        mode="rolling",
        context_obj=ctx.obj,
        **params,
    )
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    result = run_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    generated_report = resolve_report_path(
        params["report_path"], params["output_csv"], "backtest-rolling"
    )
    if generated_report:
        write_tearsheet(
            result,
            generated_report,
            title="Rolling Backtest Tear Sheet",
            extra_notes=[run.universe_note] if run.universe_note else [],
        )
    if params["output_csv"]:
        print_ledger_csv(result)
        return

    console = agentio.get_console()
    console.print(
        f"[dim]Rolling window: {run.start_date.isoformat()} to {run.end_date.isoformat()}[/dim]"
    )
    if run.universe_note:
        console.print(f"[dim]Universe: {run.universe_note}[/dim]")
    print_backtest(result)
    if generated_report:
        console.print(f"[green]Report:[/green] {generated_report}")
        if params["open_report"]:
            from screener.reporting import open_report as open_report_file

            open_report_file(generated_report)
    if params["dashboard"]:
        from screener.backtester.dashboard import render_dashboard, serve_dashboard

        dashboard_path = render_dashboard(result, params["dashboard_dir"])
        console.print(f"[green]Dashboard:[/green] {dashboard_path}")
        console.print(
            f"[green]Serving:[/green] http://127.0.0.1:{params['dashboard_port']}/{dashboard_path.name}"
        )
        console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
        serve_dashboard(dashboard_path.parent, int(params["dashboard_port"]))
