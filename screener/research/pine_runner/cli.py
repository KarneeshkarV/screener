"""CLI for the research Pine runner."""
from __future__ import annotations

import click

from screener.research.pine_runner.output import print_market_table, write_trades_json
from screener.research.pine_runner.run import run_market


@click.command()
@click.option("--market", type=click.Choice(["us", "india"]), default="us")
@click.option("--years", type=int, default=3, help="Backtest window length (years).")
@click.option("--limit", type=int, default=0, help="Cap universe size (0 = all).")
@click.option("--refresh", is_flag=True, help="Force re-fetch OHLCV.")
@click.option(
    "--trades-json",
    type=str,
    default=None,
    help="If set, write per-strategy top-trader ticker lists to this JSON file.",
)
def main(market: str, years: int, limit: int, refresh: bool, trades_json: str | None) -> None:
    result = run_market(market=market, years=years, limit=limit, refresh=refresh)
    print_market_table(result)
    if trades_json:
        write_trades_json(result, trades_json)
