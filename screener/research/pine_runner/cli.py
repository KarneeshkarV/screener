"""CLI for the research Pine runner."""

from __future__ import annotations

import click

from screener.markets import market_option
from screener.research.pine_runner.output import print_market_table, write_trades_json
from screener.research.pine_runner.run import run_market


@click.command()
@market_option(default="us")
@click.option("--years", type=int, default=3, help="Backtest window length (years).")
@click.option("--limit", type=int, default=0, help="Cap universe size (0 = all).")
@click.option("--refresh", is_flag=True, help="Force re-fetch OHLCV.")
@click.option(
    "--universe",
    type=str,
    default=None,
    help="Named index universe (e.g. nifty500, sp500). Defaults to a top-500-by-volume TV scan.",
)
@click.option(
    "--strategy",
    type=str,
    default=None,
    help="Run only this registered strategy (default: all).",
)
@click.option(
    "--trades-json",
    type=str,
    default=None,
    help="If set, write per-strategy top-trader ticker lists to this JSON file.",
)
def main(
    market: str,
    years: int,
    limit: int,
    refresh: bool,
    universe: str | None,
    strategy: str | None,
    trades_json: str | None,
) -> None:
    try:
        result = run_market(
            market=market,
            years=years,
            limit=limit,
            refresh=refresh,
            universe=universe,
            strategy=strategy,
        )
    except ValueError as exc:
        # An unknown --strategy or --universe is a typo, not a crash.
        raise click.UsageError(str(exc)) from exc
    print_market_table(result)
    if trades_json:
        write_trades_json(result, trades_json)
