"""Historical backtest CLI command."""

from __future__ import annotations

from datetime import date, datetime

import click

from screener import history
from screener.backtester.cli_common import (
    CommonBacktestParams,
    backtest_options,
    build_backtest_config,
    build_backtest_fetcher,
    build_slippage_model,
    intraday_options,
    parse_partial_exits,
    parse_ticker_list,
    resolve_min_filters,
    resolve_report_path,
    resolve_strategy_exprs,
    sizing_options,
    validate_sizing,
    write_tearsheet,
)
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.markets import as_of_option, get_market, market_option


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
@intraday_options
@sizing_options
def backtest_historical(
    market,
    as_of,
    from_run,
    run_age_days,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    cost_model,
    initial_capital,
    benchmark,
    tickers,
    universe_file,
    max_universe,
    min_price,
    min_avg_dollar_volume,
    adv_window,
    reserve_multiple,
    no_reinvest,
    slippage_model,
    half_spread_bps,
    vol_impact_k,
    no_gap_fills,
    entry_order,
    entry_limit_bps,
    allow_reentry,
    max_reentries,
    partial_exit_args,
    price_adjustment,
    interval,
    output_csv,
    report_path,
    open_report,
    sizing_rule,
    sizing_risk_pct,
    sizing_position_pct,
    sizing_atr_window,
    sizing_atr_multiple,
    sizing_vol_window,
    intraday_only,
):
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
    validate_sizing(sizing_rule, stop_loss)
    ctx = click.get_current_context()
    snapshot = None
    if from_run:
        if tickers or universe_file:
            raise click.UsageError(
                "--from-run supplies the universe; drop --tickers/--universe-file."
            )
        if as_of is not None:
            raise click.UsageError(
                "--from-run derives --as-of from the stored run; drop --as-of."
            )
        try:
            snapshot = history.resolve_replay_run(from_run, min_age_days=run_age_days)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if (
            ctx.get_parameter_source("market") != click.core.ParameterSource.DEFAULT
            and market != snapshot.market
        ):
            raise click.UsageError(
                f"screen run #{snapshot.run_id} is for market "
                f"{snapshot.market!r}, not {market!r}."
            )
        market = snapshot.market
        as_of = snapshot.run_date
        if not strategy_name and not entry_expr:
            # Pure replay: admit every stored ticker at the run date.
            entry_expr = "close > 0"
        if ctx.get_parameter_source("top") == click.core.ParameterSource.DEFAULT:
            top = len(snapshot.tickers)
        click.echo(
            f"Replaying screen run #{snapshot.run_id} "
            f"({snapshot.market}/{snapshot.criteria} @ {snapshot.run_date.isoformat()}, "
            f"{len(snapshot.tickers)} tickers)",
            err=True,
        )
    elif as_of is None:
        raise click.UsageError("--as-of is required unless --from-run is used.")

    entry_expr, exit_expr = resolve_strategy_exprs(strategy_name, entry_expr, exit_expr)
    slip_model = build_slippage_model(
        slippage_model, slippage_bps, half_spread_bps, vol_impact_k
    )
    partial_exits = parse_partial_exits(partial_exit_args)
    bench = benchmark or get_market(market).benchmark
    as_of_date: date = as_of.date() if isinstance(as_of, datetime) else as_of

    ticker_tuple = None
    if snapshot is not None:
        ticker_tuple = tuple(snapshot.tickers)
    elif tickers:
        ticker_tuple = parse_ticker_list(tickers)
    if not ticker_tuple and not universe_file:
        raise click.UsageError(
            "No universe provided: pass --tickers, --universe-file, or --from-run. "
            "The TradingView current-screener fallback was removed because it injects survivorship bias."
        )

    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )
    common = CommonBacktestParams(
        market=market,
        benchmark=bench,
        hold=hold,
        top=top,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        strategy_name=strategy_name,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        cost_model=cost_model,
        slippage_model=slip_model,
        initial_capital=initial_capital,
        tickers=ticker_tuple,
        universe_file=universe_file,
        max_universe=max_universe,
        min_price=resolved_min_price,
        min_avg_dollar_volume=resolved_min_adv,
        adv_window=adv_window,
        gap_fills=not no_gap_fills,
        entry_order_type=entry_order,
        entry_limit_bps=entry_limit_bps,
        partial_exits=partial_exits,
        price_adjustment=price_adjustment,
        interval=interval,
        intraday_only=bool(intraday_only),
        sizing_rule=sizing_rule,
        sizing_risk_pct=sizing_risk_pct,
        sizing_position_pct=sizing_position_pct,
        sizing_atr_window=sizing_atr_window,
        sizing_atr_multiple=sizing_atr_multiple,
        sizing_vol_window=sizing_vol_window,
    )
    cfg = build_backtest_config(
        common,
        as_of=as_of_date,
        reserve_multiple=int(reserve_multiple),
        reinvest=not no_reinvest,
        allow_reentry=bool(allow_reentry),
        max_reentries=int(max_reentries),
    )

    fetcher = build_backtest_fetcher(
        ctx.obj, price_adjustment=price_adjustment, interval=interval
    )
    from screener.backtester import historical as historical_engine

    result = historical_engine.run_backtest(cfg, fetcher)
    generated_report = resolve_report_path(
        report_path, output_csv, "backtest-historical"
    )
    if generated_report:
        if snapshot is not None:
            universe_note = (
                f"replayed screen run #{snapshot.run_id} "
                f"({snapshot.market}/{snapshot.criteria} @ {snapshot.run_date.isoformat()}, "
                f"{len(snapshot.tickers)} saved rows); point-in-time snapshot of what the "
                "screen showed — limited to the top-N persisted at screen time"
            )
        else:
            universe_note = (
                f"explicit universe: {len(ticker_tuple)} tickers via --tickers"
                if ticker_tuple
                else f"universe file: {universe_file}"
            ) + "; survivorship bias: supplied list is not point-in-time"
        write_tearsheet(
            result,
            generated_report,
            title="Historical Backtest Tear Sheet",
            extra_notes=[universe_note],
        )
    if output_csv:
        print_ledger_csv(result)
        return
    print_backtest(result)
    if generated_report:
        click.echo(f"Report: {generated_report}")
        if open_report:
            from screener.reporting import open_report as open_report_file

            open_report_file(generated_report)


__all__ = ["backtest_historical"]
