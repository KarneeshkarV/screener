"""Click command for GARP fundamental screens."""

from __future__ import annotations

import click
from pydantic import ValidationError

from screener.cache import parse_ttl
from screener.commands.requests import GarpRequest
from screener.display import print_csv, print_garp_results
from screener.garp import load_garp_universe, screen_india_garp, screen_us_garp
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
    try:
        req = GarpRequest(
            market=market,
            universe_size=universe_size,
            limit=limit,
            workers=workers,
            output_csv=output_csv,
            refresh=refresh,
            cache_ttl=cache_ttl,
        )
    except ValidationError as exc:
        msg = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise click.UsageError(msg) from exc

    ttl = parse_ttl(req.cache_ttl, default=86400)
    universe = load_garp_universe(
        req.market,
        int(req.universe_size),
        cache_ttl=ttl,
        refresh=req.refresh,
    )
    if universe.empty:
        click.echo("No tickers returned from the base universe scan.")
        return

    click.echo(
        f"Universe: {len(universe)} liquid {req.market.upper()} tickers. Enriching...",
        err=req.output_csv,
    )
    if req.market == "india":
        results = screen_india_garp(
            universe,
            limit=int(req.limit),
            workers=int(req.workers),
            cache_ttl=ttl,
            refresh=req.refresh,
        )
    else:
        results = screen_us_garp(
            universe, limit=int(req.limit), workers=int(req.workers)
        )

    if req.output_csv:
        print_csv(results)
        return
    print_garp_results(results, req.market)
