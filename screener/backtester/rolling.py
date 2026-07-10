"""Rolling backtest CLI and compatibility exports."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import click
from rich.console import Console

from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    referenced_fundamental_fields,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.data import build_price_fetcher
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.models import (
    SUPPORTED_INTERVALS,
    BacktestConfig,
    DataPolicy,
    ExecutionPolicy,
    PortfolioPolicy,
    SignalPolicy,
    UniversePolicy,
)
from screener.backtester.rolling_candidates import (
    _RollingCandidateMatrices,
    _build_rolling_candidate_matrices,
    _candidate_rows_for_day,
)
from screener.backtester.rolling_simulation import (
    _DailyRankingSource,
    _RollingSimulationSetup,
    _assemble_results,
    _prepare_simulation,
    run_rolling_backtest,
)
from screener.markets import get_market, get_price_fetcher, market_option
from screener.regime import TREND_LABELS
from screener.universes import (
    UniverseName,
    load_current_universe,
    load_sp500_membership,
)

__all__ = [
    "_DailyRankingSource",
    "_RollingCandidateMatrices",
    "_RollingSimulationSetup",
    "_assemble_results",
    "_build_rolling_candidate_matrices",
    "_candidate_rows_for_day",
    "_prepare_simulation",
    "backtest_rolling",
    "run_rolling_backtest",
]


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
@click.option("--hold", type=int, default=20, help="Holding period (trading days).")
@click.option("--top", type=int, default=10, help="Concurrent portfolio slots.")
@click.option("--entry", "entry_expr", default=None, help="Pine-like entry expression.")
@click.option("--exit", "exit_expr", default=None, help="Pine-like exit expression.")
@click.option(
    "--strategy", "strategy_name", default=None, help="Named strategy shortcut."
)
@click.option(
    "--universe",
    type=click.Choice(["sp500", "nifty50"]),
    default=None,
    help="Current index universe. Defaults to sp500 for US and nifty50 for India.",
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
        "Reduce survivorship bias: only allow entries after a symbol's index "
        "'date added' (sp500 universe only). Removed ex-members are still "
        "absent because their delisted history is unavailable."
    ),
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
@click.option(
    "--max-universe",
    type=int,
    default=0,
    help="Cap universe size before fetching prices. Pass 0 to disable.",
)
@click.option(
    "--stop-loss", type=float, default=None, help="Stop loss (fraction, e.g. 0.08)."
)
@click.option("--take-profit", type=float, default=None, help="Take profit (fraction).")
@click.option(
    "--trailing-stop", type=float, default=None, help="Trailing stop (fraction)."
)
@click.option(
    "--slippage-bps", type=float, default=0.0, help="Slippage per fill (bps)."
)
@click.option(
    "--commission-bps", type=float, default=0.0, help="Commission per fill (bps)."
)
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark symbol (default: SPY for US, ^NSEI for India).",
)
@click.option(
    "--min-price",
    type=float,
    default=None,
    help="Minimum signal-day close. Pass 0 to disable.",
)
@click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help="Minimum rolling mean dollar volume. Pass 0 to disable.",
)
@click.option(
    "--adv-window",
    type=int,
    default=20,
    help="Lookback bars for average dollar-volume filter.",
)
@click.option(
    "--slippage-model",
    type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
    default="fixed",
)
@click.option("--half-spread-bps", type=float, default=0.0)
@click.option("--vol-impact-k", type=float, default=0.1)
@click.option("--no-gap-fills", is_flag=True, default=False)
@click.option(
    "--entry-order", type=click.Choice(["moo", "moc", "limit"]), default="moo"
)
@click.option("--entry-limit-bps", type=float, default=None)
@click.option(
    "--partial-exit",
    "partial_exit_args",
    multiple=True,
    help="Scale-out tier as PROFIT_FRAC:SHARES_FRAC.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
)
@click.option(
    "--interval",
    type=click.Choice(list(SUPPORTED_INTERVALS)),
    default="1d",
    show_default=True,
    help=(
        "Bar interval. Intraday values (1h/30m/15m/5m/1m) fetch from yfinance "
        "and are subject to its history caps (1m ~30d, 15m/30m ~60d, 1h ~730d)."
    ),
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
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a static, self-contained HTML tear-sheet to this file.",
)
@click.option(
    "--open-report",
    is_flag=True,
    default=False,
    help="Open the generated HTML report in the default browser.",
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
def backtest_rolling(
    market,
    start_arg,
    end_arg,
    years,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    universe,
    no_universe_cache,
    point_in_time,
    tickers,
    universe_file,
    max_universe,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    initial_capital,
    benchmark,
    min_price,
    min_avg_dollar_volume,
    adv_window,
    slippage_model,
    half_spread_bps,
    vol_impact_k,
    no_gap_fills,
    entry_order,
    entry_limit_bps,
    partial_exit_args,
    price_adjustment,
    interval,
    regime_filter_args,
    fundamentals_provider,
    fundamental_field_args,
    fundamental_lag_days,
    output_csv,
    report_path,
    open_report,
    dashboard,
    dashboard_port,
    dashboard_dir,
):
    """Run a true daily rolling backtest over a date window."""
    if output_csv and dashboard:
        raise click.UsageError("--csv and --dashboard cannot be used together.")
    if fundamentals_provider == "fmp" and market != "us":
        raise click.UsageError(
            "--fundamentals-provider fmp currently supports only -m us."
        )
    if fundamentals_provider in {"openscreener", "yfinance"} and market != "india":
        raise click.UsageError(
            f"--fundamentals-provider {fundamentals_provider} currently supports only -m india."
        )

    entry_expr, exit_expr = resolve_strategy_exprs(strategy_name, entry_expr, exit_expr)

    # A strategy/expression may reference dated fundamental fields (e.g.
    # revenue_up_3q). Those columns only exist once a fundamentals provider is
    # merged into the bars, so auto-enable the market's default provider when the
    # expressions need fundamentals and none was requested explicitly. Otherwise
    # the entry evaluator fails with "Unknown identifier".
    needed_fundamentals = referenced_fundamental_fields(entry_expr, exit_expr)
    if needed_fundamentals and fundamentals_provider is None:
        fundamentals_provider = "fmp" if market == "us" else "openscreener"

    slip_model = build_slippage_model(
        slippage_model, slippage_bps, half_spread_bps, vol_impact_k
    )
    partial_exits = parse_partial_exits(partial_exit_args)
    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )

    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    market_meta = get_market(market)
    bench = benchmark or market_meta.benchmark
    resolved_fundamental_lag_days = (
        int(fundamental_lag_days)
        if fundamental_lag_days is not None
        else (60 if fundamentals_provider in {"openscreener", "yfinance"} else 1)
    )
    # When the user pins an explicit field list, make sure the fields the
    # expressions actually reference are fetched too; an empty list falls back to
    # the provider defaults, which already cover every known fundamental field.
    resolved_fundamental_fields = tuple(dict.fromkeys(fundamental_field_args))
    if resolved_fundamental_fields:
        resolved_fundamental_fields = tuple(
            dict.fromkeys((*resolved_fundamental_fields, *sorted(needed_fundamentals)))
        )

    ticker_tuple = None
    universe_note = None
    membership_added: tuple[tuple[str, date], ...] = ()
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())
    elif not universe_file:
        resolved_universe = cast(
            UniverseName,
            universe or market_meta.default_universe,
        )
        loaded = load_current_universe(
            resolved_universe,
            as_of=end_date,
            use_cache=not no_universe_cache,
        )
        ticker_tuple = loaded.symbols
        universe_note = f"{loaded.name}: {len(loaded.symbols)} symbols from {loaded.source}; cache={loaded.cached_path}"
        if point_in_time:
            if resolved_universe != "sp500":
                raise click.UsageError(
                    "--point-in-time currently supports only the sp500 universe."
                )
            added_by_symbol = load_sp500_membership(
                as_of=end_date, use_cache=not no_universe_cache
            )
            membership_added = tuple(
                (symbol, added)
                for symbol, added in added_by_symbol.items()
                if added is not None
            )
            universe_note += (
                f"; point-in-time entries via 'date added' "
                f"({len(membership_added)} dated symbols; removed ex-members not reconstructed)"
            )
        else:
            universe_note += (
                "; survivorship bias: today's members applied to history "
                "(pass --point-in-time to filter by 'date added')"
            )
    if point_in_time and not membership_added:
        raise click.UsageError(
            "--point-in-time requires an index universe; it cannot be used with "
            "--tickers or --universe-file."
        )

    cfg = BacktestConfig(
        market=market,
        as_of=end_date,
        benchmark=bench,
        universe=UniversePolicy(
            tickers=ticker_tuple,
            universe_file=universe_file,
            membership_added=membership_added,
            max_universe=int(max_universe),
        ),
        signals=SignalPolicy(
            strategy_name=strategy_name,
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            regime_filter=tuple(dict.fromkeys(regime_filter_args)),
            fundamentals_provider=fundamentals_provider,
            fundamental_fields=resolved_fundamental_fields,
            fundamental_lag_days=max(resolved_fundamental_lag_days, 0),
        ),
        data=DataPolicy(interval=interval, price_adjustment=price_adjustment),
        execution=ExecutionPolicy(
            hold=int(hold),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            slippage_bps=float(slippage_bps),
            commission_bps=float(commission_bps),
            slippage_model=slip_model,
            gap_fills=not no_gap_fills,
            entry_order_type=entry_order,
            entry_limit_bps=entry_limit_bps,
            partial_exits=partial_exits,
        ),
        portfolio=PortfolioPolicy(
            top=int(top),
            initial_capital=float(initial_capital),
            min_price=resolved_min_price,
            min_avg_dollar_volume=resolved_min_adv,
            avg_dollar_volume_window=int(adv_window),
            reinvest=True,
        ),
    )

    fetcher = get_price_fetcher(
        click.get_current_context().obj,
        builder=build_price_fetcher,
        auto_adjust=price_adjustment == "full",
        interval=interval,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=start_date, end_date=end_date
    )
    generated_report = report_path
    if generated_report is None and not output_csv:
        from screener.reporting import temp_report_path

        generated_report = temp_report_path("backtest-rolling")
    if generated_report:
        from screener.backtester.tearsheet import render_tearsheet

        render_tearsheet(
            result,
            generated_report,
            title="Rolling Backtest Tear Sheet",
            extra_notes=[universe_note] if universe_note else [],
        )
    if output_csv:
        print_ledger_csv(result)
        return

    console = Console()
    console.print(
        f"[dim]Rolling window: {start_date.isoformat()} to {end_date.isoformat()}[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_backtest(result)
    if generated_report:
        console.print(f"[green]Report:[/green] {generated_report}")
        if open_report:
            from screener.reporting import open_report as open_report_file

            open_report_file(generated_report)
    if dashboard:
        from screener.backtester.dashboard import render_dashboard, serve_dashboard

        dashboard_path = render_dashboard(result, dashboard_dir)
        console.print(f"[green]Dashboard:[/green] {dashboard_path}")
        console.print(
            f"[green]Serving:[/green] http://127.0.0.1:{dashboard_port}/{dashboard_path.name}"
        )
        console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
        serve_dashboard(dashboard_path.parent, int(dashboard_port))
