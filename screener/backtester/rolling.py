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
from screener.backtester.display import (
    print_backtest,
    print_ledger_csv,
)
from screener.backtester.rolling_simulation import (
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
    run_rolling_backtest,
)
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run
from screener.markets import market_option
from screener.universes import available_universes

__all__ = ["backtest_rolling", "rolling_run_options"]


def _print_candidates(run: Any) -> None:
    """Print the last ranked candidate set the engine would have entered on.

    The point is comparability with ``screener screen --universe``: both read
    the same candidate matrices, so a name in one and not the other is a real
    disagreement rather than a difference in how the two commands were run.

    The date printed is the last *signal* bar of the window, which is not the
    last bar: the engine only calls a name a candidate when a later bar exists
    to fill the entry on. A screen has no such bar and names the last one, so
    to compare them run the screen at the date printed here.
    """
    from screener.backtester.rolling_simulation import prepare_rolling_backtest
    from screener.backtester.signal_panel import SignalPanel, day_candidates_from_panel

    prepared = prepare_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    console = agentio.get_console()
    for warning in prepared.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    panel = SignalPanel(
        exit_signals=prepared.exit_signals,
        candidate_matrices=prepared.candidate_matrices,
    )
    for day in reversed(prepared.master_dates):
        found = day_candidates_from_panel(panel, day)
        if found.candidates:
            break
    else:
        console.print("[yellow]No candidate fired anywhere in the window.[/yellow]")
        return

    console.print(f"[dim]Signal bar: {day.date().isoformat()}[/dim]")
    console.print(f"[dim]Ranked by: {found.candidates[0].rank_basis}[/dim]")
    for candidate in found.candidates:
        console.print(
            f"{candidate.rank:>4}  {candidate.ticker:<20} "
            f"setup_score={candidate.setup_score:6.2f}  "
            f"close={candidate.as_of_close:,.2f}  "
            f"adv={candidate.as_of_dollar_vol:,.0f}"
        )


# Every option that defines *what* a rolling run is: window, universe,
# signals, costs, sizing. Held as a list rather than a decorator stack so
# ``backtest-monte-carlo`` can run the identical simulation from the
# identical flags. Options that only shape how ``backtest-rolling``
# *reports* (--candidates, --compare-reinvestment, --dashboard) stay on
# that command.
_RUN_OPTION_DECORATORS: tuple[Any, ...] = (
    market_option(
        default="us",
        help="Market to backtest.",
    ),
    click.option(
        "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
    ),
    click.option(
        "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
    ),
    click.option(
        "--years",
        type=int,
        default=1,
        show_default=True,
        help="Trailing calendar years when --start is omitted.",
    ),
    backtest_options("rolling", "hold", "top", "entry", "exit", "strategy"),
    click.option(
        "--universe",
        type=str,
        default=None,
        help=(
            "Named universe (built-ins: "
            + ", ".join((*available_universes(), "dynamic"))
            + "). Defaults to the market index."
        ),
    ),
    click.option(
        "--universe-config",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="TOML/YAML/JSON definitions for custom static, snapshot, or dynamic universes.",
    ),
    click.option(
        "--dynamic-base",
        default=None,
        help="Candidate index for --universe dynamic (default: sp500 or nifty500).",
    ),
    click.option(
        "--universe-size",
        type=int,
        default=100,
        show_default=True,
        help="Number of highest lagged-ADV names in a dynamic universe.",
    ),
    click.option(
        "--universe-lookback",
        type=int,
        default=60,
        show_default=True,
        help="Trailing bars used for dynamic-universe ADV ranking.",
    ),
    click.option(
        "--universe-rebalance",
        type=click.Choice(["daily", "weekly", "monthly", "quarterly"]),
        default="monthly",
        show_default=True,
        help="Dynamic-universe membership refresh frequency.",
    ),
    click.option(
        "--no-universe-cache",
        is_flag=True,
        default=False,
        help="Force live constituent refresh instead of today's cache.",
    ),
    click.option(
        "--point-in-time/--no-point-in-time",
        default=True,
        show_default=True,
        help=(
            "Require point-in-time membership. Custom snapshot universes use full "
            "membership windows; sp500 uses its available historical additions. "
            "On by default, and silently inactive for universes that carry no "
            "membership history (--tickers, --universe-file); pass it explicitly to "
            "make those an error instead."
        ),
    ),
    backtest_options(
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
    ),
    click.option(
        "--spread-proxy",
        is_flag=True,
        default=False,
        help=(
            "Estimate per-fill half-spread via Corwin-Schultz (2012) high-low "
            "estimator and charge it as slippage (on top of --slippage-model)."
        ),
    ),
    backtest_options(
        "rolling",
        "initial-capital",
        "benchmark",
        "min-price",
        "min-avg-dollar-volume",
        "adv-window",
        "min-score",
        "slippage-model",
        "half-spread-bps",
        "vol-impact-k",
        "no-gap-fills",
        "entry-order",
        "entry-limit-bps",
        "partial-exit",
        "price-adjustment",
        "interval",
    ),
    backtest_options("rolling", "regime-filter", "sector-neutral"),
    click.option(
        "--rank-exit",
        "rank_exit",
        type=str,
        default=None,
        help=(
            "Rank-based exit rebalance: 'weekly', 'monthly', or an integer "
            "trading-bar period N. On every Nth bar of the window, any holding "
            "outside the top --rank-universe-size of the prior completed bar's "
            "candidate ranking is closed at this bar's close (exit reason: rank). "
            "'weekly'/'monthly' count trading days and require --interval 1d. "
            "Requires --rank-universe-size >= --top."
        ),
    ),
    click.option(
        "--rank-universe-size",
        type=int,
        default=50,
        show_default=True,
        help="Top-N ranked candidates a holding must stay in under --rank-exit.",
    ),
    backtest_options("rolling", "earnings-blackout", "refresh"),
    click.option(
        "--fundamentals-provider",
        type=click.Choice(["fmp", "openscreener", "yfinance"]),
        default=None,
        help="Merge dated fundamentals into rolling backtest bars (US: fmp, India: openscreener or yfinance).",
    ),
    click.option(
        "--fundamental-field",
        "fundamental_field_args",
        multiple=True,
        help=(
            "Fundamental field to fetch for expressions (repeatable). Defaults to "
            "pe_ttm, pb_ttm, roe_ttm, debt_to_equity, revenue_growth_yoy, "
            "eps_growth_yoy, revenue_up_3q, market_cap."
        ),
    ),
    click.option(
        "--fundamental-lag-days",
        type=int,
        default=None,
        help="Calendar-day lag applied to fundamental effective dates (defaults: fmp=1, openscreener=60).",
    ),
    backtest_options("rolling", "csv", "report", "open-report"),
    intraday_options,
    sizing_options,
)


def rolling_run_options(command: Any) -> Any:
    """Apply every shared rolling-run option, in ``--help`` order."""
    for decorator in reversed(_RUN_OPTION_DECORATORS):
        command = decorator(command)
    return command


@click.command(name="backtest-rolling")
@rolling_run_options
@backtest_options("rolling", "candidates")
@click.option(
    "--compare-reinvestment/--no-compare-reinvestment",
    default=True,
    show_default=True,
    help=(
        "Also run the same window under the other equal-slot sizing rule and "
        "show a side-by-side comparison in the table and the HTML report. "
        "Doubles the simulation work; pass --no-compare-reinvestment to skip it."
    ),
)
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
def backtest_rolling(**params: Any) -> None:
    """Run a true daily rolling backtest over a date window."""
    ctx = click.get_current_context()
    request = BacktestRequest(
        mode="rolling",
        context_obj=ctx.obj,
        adv_window_was_explicit=(
            ctx.get_parameter_source("adv_window")
            == click.core.ParameterSource.COMMANDLINE
        ),
        # Anything but DEFAULT counts as asked-for. A --config file reaches
        # Click as DEFAULT_MAP, and a user who turned point-in-time on there
        # meant it just as much as one who typed the flag; treating that as a
        # default would silently downgrade the run to a biased one.
        point_in_time_was_explicit=(
            ctx.get_parameter_source("point_in_time")
            is not click.core.ParameterSource.DEFAULT
        ),
        **params,
    )
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    if params["candidates"]:
        _print_candidates(run)
        return
    compare_reinvestment = (
        params["compare_reinvestment"]
        and run.config.sizing_rule in {"equal_slot", "reinvested_equal_slot"}
        and not params["output_csv"]
    )
    fixed_result = None
    reinvested_result = None
    if compare_reinvestment:
        prepared = prepare_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )
        fixed_config = run.config.model_copy(update={"sizing_rule": "equal_slot"})
        reinvested_config = run.config.model_copy(
            update={"sizing_rule": "reinvested_equal_slot"}
        )
        fixed_result = run_prepared_rolling_backtest(prepared, fixed_config)
        reinvested_result = run_prepared_rolling_backtest(prepared, reinvested_config)
        result = (
            reinvested_result
            if run.config.sizing_rule == "reinvested_equal_slot"
            else fixed_result
        )
    else:
        result = run_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )
    sizing_comparison = (
        (fixed_result, reinvested_result)
        if fixed_result is not None and reinvested_result is not None
        else None
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
            sizing_comparison=sizing_comparison,
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
    # The trade log is large and always lands in the tear sheet, so keep the
    # terminal to the header plus metrics.
    print_backtest(result, sizing_comparison=sizing_comparison, show_ledger=False)
    if generated_report:
        from screener.reporting import windows_report_path

        console.print(f"[green]Report:[/green] {generated_report}")
        windows_report = windows_report_path(generated_report)
        if windows_report:
            console.print(f"[green]Windows:[/green] {windows_report}")
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
