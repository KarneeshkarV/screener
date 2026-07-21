"""Mark Minervini Trend Template screen command."""

from __future__ import annotations

from datetime import datetime

import click
from rich.console import Console

from screener.backtester.data import build_price_fetcher
from screener.markets import (
    as_of_option,
    get_price_fetcher,
    market_option,
    resolve_as_of,
)
from screener.minervini import render_rows, scan_minervini


@click.command(name="mark-minervini")
@market_option(
    default="us",
    help="Market to screen.",
)
@as_of_option()
@click.option(
    "-n",
    "--limit",
    type=int,
    default=30,
    show_default=True,
    help="Number of results.",
)
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="Universe cache TTL, e.g. 30s, 15m, 1h, off.",
)
@click.option("--refresh", is_flag=True, help="Bypass cached universe/prices.")
def mark_minervini(
    market: str,
    as_of_arg: datetime | None,
    limit: int,
    cache_ttl: str,
    refresh: bool,
) -> None:
    """Screen for stocks matching Mark Minervini's Trend Template."""
    as_of = resolve_as_of(as_of_arg)
    console = Console()
    rows = scan_minervini(
        market,
        as_of=as_of,
        limit=int(limit),
        cache_ttl=cache_ttl,
        refresh=refresh,
        fetcher=get_price_fetcher(
            click.get_current_context().obj, builder=build_price_fetcher
        ),
    )
    render_rows(rows, console, market)
    console.print(
        "\n[dim]RS Rank is a universe-relative 12-month percentile proxy, not "
        "Investor's Business Daily's proprietary RS Rating.[/dim]"
    )


__all__ = ["mark_minervini"]
