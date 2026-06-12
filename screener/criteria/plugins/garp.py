"""GARP pipeline criterion — ``screen -c garp`` runs the GARP scan."""

from __future__ import annotations

from typing import Any

import click

from screener.cache import parse_ttl
from screener.criteria import criterion
from screener.display import print_csv, print_garp_results
from screener.garp import run_garp_screen

# Screen-context defaults for options the generic ``screen`` command does not
# expose (the standalone ``garp`` command's own defaults).
_DEFAULT_UNIVERSE_SIZE = 200
_DEFAULT_WORKERS = 8


@criterion("garp", pipeline=True)
def garp_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
    **_: Any,
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
