"""Click command for the single-ticker conviction card."""

from __future__ import annotations

import json
from datetime import date, datetime

import click
from rich.console import Console

from screener.backtester.data import build_price_fetcher
from screener.cache import parse_ttl
from screener.conviction import build_conviction_card, render_card


@click.command(name="conviction")
@click.argument("ticker")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market the ticker trades in.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to evaluate (default: today).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the card as JSON instead of a table.",
)
@click.option("--refresh", is_flag=True, help="Bypass cached provider data.")
@click.option(
    "--cache-ttl",
    default="1d",
    show_default=True,
    help="Cache TTL for provider data, e.g. 15m, 1h, 1d, off.",
)
def conviction(
    ticker: str,
    market: str,
    as_of_arg: datetime | None,
    as_json: bool,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """One composite conviction card for TICKER, fusing the screen pillars."""
    as_of = as_of_arg.date() if isinstance(as_of_arg, datetime) else date.today()
    ttl = parse_ttl(cache_ttl, default=86400)
    fetcher = click.get_current_context().obj or build_price_fetcher(refresh=refresh)
    card = build_conviction_card(
        ticker, market, as_of, fetcher, cache_ttl=ttl, refresh=refresh
    )
    if as_json:
        click.echo(json.dumps(card.to_dict(), indent=2))
        return
    render_card(card, Console())
