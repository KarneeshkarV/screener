import click

from screener import history
from screener.backtest import run_backtest
from screener.criteria import CRITERIA, combine
from screener.historical_criteria import HIST_CRITERIA, FUND_CRITERIA
from screener.scanner import scan, MARKETS
from screener.display import (
    print_backtest,
    print_backtest_csv,
    print_historical_backtest,
    print_historical_backtest_csv,
    print_results,
    print_csv,
)


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


@cli.command("backtest-last-run")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default=None,
    help="Filter last-run lookup by market. Default: most recent run of any market.",
)
@click.option(
    "-c",
    "--criteria",
    "criteria_label",
    default=None,
    help="Filter last-run lookup by criteria label (e.g. 'ema+breakout').",
)
@click.option(
    "--scope",
    type=click.Choice(["next", "all"]),
    default="next",
    help="next = from run timestamp to today; all = ~5y history up to today.",
)
@click.option("--hold", default=0, help="Max holding days (0 = hold to end of scope).")
@click.option("--top", default=0, help="Use only top-N ranked tickers (0 = all).")
@click.option("--benchmark", default=None, help="Override benchmark TV symbol.")
@click.option("--refresh", is_flag=True, help="Force re-fetch prices, ignore Parquet cache.")
@click.option("--csv", "output_csv", is_flag=True, help="Emit per-ticker CSV ledger.")
@click.pass_context
def backtest_last_run(ctx, market, criteria_label, scope, hold, top, benchmark, refresh, output_csv):
    """Backtest the last saved screener run as an equal-weighted basket."""
    if ctx.info_name == "backtest":
        click.echo(
            "[deprecated] 'backtest' is renamed to 'backtest-last-run'. "
            "Please update your command.",
            err=True,
        )
    try:
        result = run_backtest(
            market=market,
            criteria=criteria_label,
            scope=scope,
            hold_days=hold,
            top=top or None,
            benchmark_override=benchmark,
            refresh=refresh,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e))
    if output_csv:
        print_backtest_csv(result)
    else:
        print_backtest(result)


# Keep the old name as a deprecated alias.
cli.add_command(backtest_last_run, name="backtest")


@cli.command("backtest-historical")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="us",
    show_default=True,
    help="Market to screen.",
)
@click.option(
    "-c",
    "--criteria",
    "criteria_name",
    type=click.Choice(sorted(HIST_CRITERIA)),
    default="ema",
    show_default=True,
    help=(
        "Criterion to evaluate historically. "
        f"OHLCV-only: ema, breakout, ema_breakout. "
        f"Needs fundamentals (yfinance quarterly): {', '.join(sorted(FUND_CRITERIA))}."
    ),
)
@click.option(
    "--as-of",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Historical screen date (YYYY-MM-DD). Criteria are evaluated using only data up to this date.",
)
@click.option("--hold", default=252, show_default=True, help="Forward holding period in trading days.")
@click.option("--top", default=20, show_default=True, help="Select top-N ranked matches.")
@click.option(
    "--universe",
    default=None,
    metavar="PATH",
    help=(
        "Path to a ticker universe file (one ticker per line). "
        "Defaults to a bundled S&P 500 / Nifty 500 list for the chosen market. "
        "Note: bundled lists carry survivorship bias."
    ),
)
@click.option("--benchmark", default=None, help="Override benchmark symbol (e.g. AMEX:SPY).")
@click.option("--refresh", is_flag=True, help="Force re-fetch prices, ignore Parquet cache.")
@click.option("--csv", "output_csv", is_flag=True, help="Emit per-ticker CSV ledger.")
def backtest_historical(market, criteria_name, as_of, hold, top, universe, benchmark, refresh, output_csv):
    """Screen a universe as of a historical date and backtest forward.

    Answers: "Which stocks matched this screen on --as-of, and how did they
    perform over the following --hold trading days?"

    Criteria are evaluated using only OHLCV data available on or before
    --as-of, so there is no lookahead bias.  TradingView Screener does not
    support historical as-of queries; this command computes criteria locally
    from downloaded price history instead.

    Example:

        uv run python main.py backtest-historical -m us -c ema \\
            --as-of 2025-04-15 --hold 252 --top 20
    """
    from screener.historical_backtest import run_historical_backtest

    as_of_date = as_of.date()

    try:
        result = run_historical_backtest(
            market=market,
            criteria_name=criteria_name,
            as_of=as_of_date,
            hold_days=hold,
            top=top,
            universe_path=universe,
            benchmark_override=benchmark,
            refresh=refresh,
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

    if output_csv:
        print_historical_backtest_csv(result)
    else:
        print_historical_backtest(result)


if __name__ == "__main__":
    cli()
