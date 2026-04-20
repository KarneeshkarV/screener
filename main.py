from datetime import date, datetime

import click

from screener import history
from screener.criteria import CRITERIA, combine
from screener.scanner import scan, MARKETS
from screener.display import print_results, print_csv


@click.group()
def cli():
    """Stock screener for US and Indian markets."""


@cli.command()
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
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
@click.option("--detail", is_flag=True, help="Show fundamental details (P/E, ROE, etc.).")
def screen(market, criteria_names, limit, order_by, output_csv, detail):
    """Screen stocks based on technical criteria."""
    criteria_fns = [CRITERIA[name] for name in criteria_names]
    filters = combine(*criteria_fns)()

    label = "+".join(criteria_names)

    total, df = scan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
    )

    if output_csv:
        print_csv(df)
        return

    run_id = history.save_run(market, label, total, df)
    prev = history.previous_run(market, label, before_id=run_id)
    if prev is None:
        added, removed, first_run = [], [], True
    else:
        added, removed = history.diff(df, prev)
        first_run = False

    print_results(
        df,
        total,
        market,
        label,
        added=added,
        removed=removed,
        first_run=first_run,
    )


_DEFAULT_BENCHMARK = {"us": "SPY", "india": "^NSEI"}


@cli.command(name="backtest-historical")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--as-of",
    "as_of",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Signal evaluation date (YYYY-MM-DD).",
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
@click.option("--stop-loss", type=float, default=None, help="Stop loss (fraction, e.g. 0.08).")
@click.option("--take-profit", type=float, default=None, help="Take profit (fraction).")
@click.option("--trailing-stop", type=float, default=None, help="Trailing stop (fraction).")
@click.option("--slippage-bps", type=float, default=0.0, help="Slippage per fill (bps).")
@click.option("--commission-bps", type=float, default=0.0, help="Commission per fill (bps).")
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option("--benchmark", default=None, help="Benchmark symbol (default: SPY for US, ^NSEI for India).")
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list (overrides default universe).",
)
@click.option("--universe-file", default=None, help="Path to newline-separated ticker file.")
@click.option("--max-universe", type=int, default=200, help="Cap on default universe size.")
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
def backtest_historical(
    market,
    as_of,
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
    initial_capital,
    benchmark,
    tickers,
    universe_file,
    max_universe,
    output_csv,
):
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
    from screener.backtester import BacktestConfig, run_backtest
    from screener.backtester.data import YFinancePriceFetcher
    from screener.backtester.display import print_backtest, print_ledger_csv
    from screener.backtester.strategies import resolve_strategy

    if strategy_name:
        s = resolve_strategy(strategy_name)
        entry_expr = entry_expr or s.entry
        exit_expr = exit_expr or s.exit

    if not entry_expr:
        raise click.UsageError("--entry (or --strategy) is required.")

    bench = benchmark or _DEFAULT_BENCHMARK.get(market, "SPY")
    as_of_date: date = as_of.date() if isinstance(as_of, datetime) else as_of

    ticker_tuple = None
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())

    cfg = BacktestConfig(
        market=market,
        as_of=as_of_date,
        hold=int(hold),
        top=int(top),
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
    )

    fetcher = click.get_current_context().obj or YFinancePriceFetcher()
    result = run_backtest(cfg, fetcher)

    if output_csv:
        print_ledger_csv(result)
        return

    print_backtest(result)


if __name__ == "__main__":
    cli()
