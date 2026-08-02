"""Historical backtest CLI command."""

from __future__ import annotations

from typing import Any

import click

from screener.backtester.cli_common import (
    backtest_options,
    intraday_options,
    resolve_report_path,
    sizing_options,
    write_tearsheet,
)
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run
from screener.markets import as_of_option, market_option


@click.command(name="backtest-historical")
@market_option(
    default="us",
    help="Market to backtest.",
)
@as_of_option(
    param_name="as_of",
    required=False,
    help="Signal evaluation date (YYYY-MM-DD). Required unless --from-run is used.",
)
@click.option(
    "--from-run",
    "from_run",
    default=None,
    help=(
        "Replay a persisted screen run as the backtest universe. Accepts a run id "
        "(see `screener history`) or MARKET:CRITERIA (e.g. india:ema), which picks "
        "the most recent run at least --run-age-days old. Sets --as-of to the run "
        "date and the universe to the stored tickers; --entry defaults to "
        "'close > 0' (buy what the screen picked) and --top to the snapshot size."
    ),
)
@click.option(
    "--run-age-days",
    type=int,
    default=0,
    show_default=True,
    help=(
        "With --from-run MARKET:CRITERIA, require the run to be at least this many "
        "calendar days old (0 = latest). Ignored for numeric run ids."
    ),
)
@backtest_options(
    "historical",
    "hold",
    "top",
    "entry",
    "exit",
    "strategy",
    "stop-loss",
    "take-profit",
    "trailing-stop",
    "slippage-bps",
    "commission-bps",
    "cost-model",
    "initial-capital",
    "benchmark",
    "tickers",
    "universe-file",
    "max-universe",
    "min-price",
    "min-avg-dollar-volume",
    "adv-window",
)
@click.option(
    "--reserve-multiple",
    type=int,
    default=3,
    help="Deepen the selection pool to top*N for reserve rotation on exits.",
)
@click.option(
    "--no-reinvest",
    is_flag=True,
    default=False,
    help="Disable reserve rotation (freed cash stays idle, matches legacy behavior).",
)
@backtest_options(
    "historical",
    "slippage-model",
    "half-spread-bps",
    "vol-impact-k",
    "no-gap-fills",
    "entry-order",
    "entry-limit-bps",
)
@click.option(
    "--allow-reentry",
    is_flag=True,
    default=False,
    help="After a position closes, re-enter the same ticker if the entry signal fires again (up to --max-reentries times).",
)
@click.option(
    "--max-reentries",
    type=int,
    default=0,
    help="Maximum number of re-entries per slot when --allow-reentry is set.",
)
@backtest_options(
    "historical",
    "partial-exit",
    "price-adjustment",
    "interval",
    "csv",
    "report",
    "open-report",
)
@intraday_options  # type: ignore[untyped-decorator]
@sizing_options  # type: ignore[untyped-decorator]
def backtest_historical(**params: Any) -> None:
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
    ctx = click.get_current_context()
    request = BacktestRequest(
        mode="historical",
        context_obj=ctx.obj,
        market_was_explicit=(
            ctx.get_parameter_source("market") != click.core.ParameterSource.DEFAULT
        ),
        top_was_explicit=(
            ctx.get_parameter_source("top") != click.core.ParameterSource.DEFAULT
        ),
        **params,
    )
    run = resolve_backtest_run(request)
    if run.replay_note:
        click.echo(run.replay_note, err=True)
    from screener.backtester import historical as historical_engine

    result = historical_engine.run_backtest(run.config, run.price_fetcher)
    generated_report = resolve_report_path(
        params["report_path"], params["output_csv"], "backtest-historical"
    )
    if generated_report:
        if run.replay_note:
            universe_note = run.replay_note.replace("Replaying ", "replayed ")
        else:
            tickers = run.config.tickers
            universe_note = (
                f"explicit universe: {len(tickers)} tickers via --tickers"
                if tickers
                else f"universe file: {run.config.universe_file}"
            ) + "; survivorship bias: supplied list is not point-in-time"
        write_tearsheet(
            result,
            generated_report,
            title="Historical Backtest Tear Sheet",
            extra_notes=[universe_note],
        )
    if params["output_csv"]:
        print_ledger_csv(result)
        return
    print_backtest(result)
    if generated_report:
        click.echo(f"Report: {generated_report}")
        if params["open_report"]:
            from screener.reporting import open_report as open_report_file

            open_report_file(generated_report)


__all__ = ["backtest_historical"]
