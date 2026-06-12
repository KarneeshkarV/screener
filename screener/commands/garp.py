"""Click command for GARP fundamental screens."""

from __future__ import annotations

import click
import pandas as pd

from screener.cache import parse_ttl
from screener.display import print_csv, print_garp_results
from screener.garp import run_garp_screen
from screener.scanner import MARKETS


@click.command(name="garp")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="india",
    help="Market to screen.",
)
@click.option(
    "--universe-size",
    type=int,
    default=200,
    show_default=True,
    help="Number of liquid tickers to enrich before filtering.",
)
@click.option("-n", "--limit", type=int, default=30, show_default=True)
@click.option("--workers", type=int, default=8, show_default=True)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option("--refresh", is_flag=True, help="Bypass cached universe/provider data.")
@click.option(
    "--cache-ttl",
    default="1d",
    show_default=True,
    help="Cache TTL for universe/provider data, e.g. 15m, 1h, 1d, off.",
)
def garp(
    market: str,
    universe_size: int,
    limit: int,
    workers: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """Find GARP stocks using market-specific fundamental data."""
    ttl = parse_ttl(cache_ttl, default=86400)

    def _announce(universe: pd.DataFrame) -> None:
        click.echo(
            f"Universe: {len(universe)} liquid {market.upper()} tickers. Enriching...",
            err=output_csv,
        )

    results = run_garp_screen(
        market,
        int(universe_size),
        limit=int(limit),
        workers=int(workers),
        cache_ttl=ttl,
        refresh=refresh,
        on_universe=_announce,
    )
    if results is None:
        click.echo("No tickers returned from the base universe scan.")
        return

    if output_csv:
        print_csv(results)
        return
    print_garp_results(results, market)
