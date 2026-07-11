"""GARP ``screen -c`` compatibility alias."""

from __future__ import annotations

from typing import Any

import click

from screener.cache import parse_ttl
from screener.display import print_csv, print_garp_results
from screener.garp import run_garp_screen

_DEFAULT_UNIVERSE_SIZE = 200
_DEFAULT_WORKERS = 8


def garp_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    ttl = parse_ttl(cache_ttl, default=86400)

    def _announce(universe: Any) -> None:
        click.echo(
            f"Universe: {len(universe)} liquid {market.upper()} tickers. Enriching...",
            err=output_csv,
        )

    results = run_garp_screen(
        market,
        _DEFAULT_UNIVERSE_SIZE,
        limit=int(limit),
        workers=_DEFAULT_WORKERS,
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
