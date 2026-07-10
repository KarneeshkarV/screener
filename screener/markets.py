"""Market metadata and shared Click option helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypeVar

import click

from screener.backtester.data import PriceFetcher, build_price_fetcher


@dataclass(frozen=True)
class Market:
    name: str
    tv_market: str
    benchmark: str
    default_universe: str
    min_price: float
    min_adv: float
    screen_min_close: float
    rs_breakout_min_close: float


MARKETS: dict[str, Market] = {
    "us": Market(
        name="us",
        tv_market="america",
        benchmark="SPY",
        default_universe="sp500",
        min_price=1.0,
        min_adv=1_000.0,
        screen_min_close=1.0,
        rs_breakout_min_close=5.0,
    ),
    "india": Market(
        name="india",
        tv_market="india",
        benchmark="^NSEI",
        default_universe="nifty50",
        min_price=10.0,
        min_adv=100_000.0,
        screen_min_close=10.0,
        rs_breakout_min_close=50.0,
    ),
}

MARKET_NAMES = tuple(MARKETS)
TV_MARKETS = {name: market.tv_market for name, market in MARKETS.items()}


def get_market(name: str) -> Market:
    return MARKETS[name]


F = TypeVar("F", bound=Callable[..., Any])


def market_option(
    *,
    default: str = "india",
    choices: tuple[str, ...] = MARKET_NAMES,
    help: str = "Market to screen.",
    show_default: bool = False,
) -> Callable[[F], F]:
    return click.option(
        "-m",
        "--market",
        type=click.Choice(list(choices)),
        default=default,
        show_default=show_default,
        help=help,
    )


def as_of_option(
    *,
    param_name: str = "as_of_arg",
    required: bool = False,
    help: str = "Trading date to evaluate (default: today).",
) -> Callable[[F], F]:
    return click.option(
        "--as-of",
        param_name,
        required=required,
        type=click.DateTime(formats=["%Y-%m-%d"]),
        default=None,
        help=help,
    )


def resolve_as_of(as_of_arg: datetime | date | None) -> date:
    if isinstance(as_of_arg, datetime):
        return as_of_arg.date()
    if isinstance(as_of_arg, date):
        return as_of_arg
    return date.today()


def get_price_fetcher(
    ctx_obj_or_none: PriceFetcher | None,
    *,
    builder: Callable[..., PriceFetcher] = build_price_fetcher,
    **build_kwargs: Any,
) -> PriceFetcher:
    """Return an injected Click context fetcher or build the production fetcher."""
    return ctx_obj_or_none or builder(**build_kwargs)
