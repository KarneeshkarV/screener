"""Historical backtest CLI command."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click

from screener import history
from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.data import build_price_fetcher
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.models import SUPPORTED_INTERVALS, BacktestConfig
from screener.markets import as_of_option, get_market, get_price_fetcher, market_option


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
@click.option("--hold", type=int, default=20, help="Holding period (trading days).")
@click.option("--top", type=int, default=10, help="Top N tickers to select.")
@click.option("--entry", "entry_expr", default=None, help="Pine-like entry expression.")
@click.option("--exit", "exit_expr", default=None, help="Pine-like exit expression.")
@click.option(
    "--strategy",
    "strategy_name",
    default=None,
    help="Named strategy shortcut (overrides --entry/--exit if given).",
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
@click.option(
    "--cost-model",
    type=click.Choice(["flat", "india"]),
    default="flat",
    show_default=True,
    help=(
        "Statutory fee model. 'flat' applies --commission-bps on every fill "
        "(legacy). 'india' applies NSE equity delivery fees (STT, stamp duty, "
        "exchange, SEBI, GST, IPFT)."
    ),
)
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark symbol (default: SPY for US, ^NSEI for India).",
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
@click.option(
    "--max-universe",
    type=int,
    default=200,
    help="Cap supplied universe size before fetching prices. Pass 0 to disable.",
)
@click.option(
    "--min-price",
    type=float,
    default=None,
    help="Minimum as-of close to admit a ticker. Default: $1 (US) / ₹10 (India). Pass 0 to disable.",
)
@click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help="Minimum rolling-mean dollar volume (close*volume) over --adv-window. Default: $1,000 (US) / ₹100,000 (India). Pass 0 to disable.",
)
@click.option(
    "--adv-window",
    type=int,
    default=20,
    help="Lookback (bars) for average dollar-volume filter.",
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
@click.option(
    "--slippage-model",
    type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
    default="fixed",
    help="Slippage model. 'fixed' = constant bps (legacy); 'half-spread' adds quoted-spread cost; 'vol-impact' adds Almgren-Chriss sqrt-law impact; 'composite' sums all three.",
)
@click.option(
    "--half-spread-bps",
    type=float,
    default=0.0,
    help="Half-spread charged on every fill (bps). Used by half-spread/composite.",
)
@click.option(
    "--vol-impact-k",
    type=float,
    default=0.1,
    help="Coefficient for sqrt-law market impact (vol-impact/composite).",
)
@click.option(
    "--no-gap-fills",
    is_flag=True,
    default=False,
    help="Disable gap-aware stop/target fills (fills always at reference price).",
)
@click.option(
    "--entry-order",
    type=click.Choice(["moo", "moc", "limit"]),
    default="moo",
    help="Entry order type. moo=next-bar open (default); moc=next-bar close; limit=limit order at close*(1 - entry_limit_bps/1e4).",
)
@click.option(
    "--entry-limit-bps",
    type=float,
    default=None,
    help="Discount below signal-bar close for limit entries (bps).",
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
@click.option(
    "--partial-exit",
    "partial_exit_args",
    multiple=True,
    help="Scale-out tier as 'PROFIT_FRAC:SHARES_FRAC' (e.g. 0.05:0.5 = close half at +5%). Repeat to configure multiple tiers.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
    help="Price-adjustment regime. full=legacy (yfinance auto_adjust=True); splits_only=split-adjust OHLC and credit dividends as cash; none=raw OHLC.",
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
):
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
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
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())
    if not ticker_tuple and not universe_file:
        raise click.UsageError(
            "No universe provided: pass --tickers, --universe-file, or --from-run. "
            "The TradingView current-screener fallback was removed because it injects survivorship bias."
        )

    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )
    cfg = BacktestConfig(
        market=market,
        as_of=as_of_date,
        hold=int(hold),
        top=int(top),
        strategy_name=strategy_name,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        initial_capital=float(initial_capital),
        benchmark=bench,
        tickers=ticker_tuple,
        universe_file=universe_file,
        max_universe=int(max_universe),
        min_price=resolved_min_price,
        min_avg_dollar_volume=resolved_min_adv,
        avg_dollar_volume_window=int(adv_window),
        reserve_multiple=int(reserve_multiple),
        reinvest=not no_reinvest,
        slippage_model=slip_model,
        cost_model=cost_model,
        gap_fills=not no_gap_fills,
        entry_order_type=entry_order,
        entry_limit_bps=entry_limit_bps,
        allow_reentry=bool(allow_reentry),
        max_reentries=int(max_reentries),
        partial_exits=partial_exits,
        price_adjustment=price_adjustment,
        interval=interval,
    )

    fetcher = get_price_fetcher(
        click.get_current_context().obj,
        builder=build_price_fetcher,
        auto_adjust=price_adjustment == "full",
        interval=interval,
    )
    from screener.backtester import historical as historical_engine

    result = historical_engine.run_backtest(cfg, fetcher)
    generated_report = report_path
    if generated_report is None and not output_csv:
        from screener.reporting import temp_report_path

        generated_report = temp_report_path("backtest-historical")
    if generated_report:
        from screener.backtester.tearsheet import render_tearsheet

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
        render_tearsheet(
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
