from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from click.testing import CliRunner
import click
import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.markets import (
    MARKETS,
    TV_MARKETS,
    as_of_option,
    get_market,
    get_price_fetcher,
    market_option,
    resolve_as_of,
)


class DummyFetcher:
    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        return {}


def test_market_metadata_preserves_existing_values() -> None:
    assert get_market("us").benchmark == "SPY"
    assert get_market("us").tv_market == "america"
    assert get_market("us").default_universe == "sp500"
    assert get_market("us").min_price == 1.0
    assert get_market("us").min_adv == 1_000.0
    assert get_market("us").screen_min_close == 1.0
    assert get_market("us").rs_breakout_min_close == 5.0

    assert get_market("india").benchmark == "^NSEI"
    assert get_market("india").tv_market == "india"
    assert get_market("india").default_universe == "nifty50"
    assert get_market("india").min_price == 10.0
    assert get_market("india").min_adv == 100_000.0
    assert get_market("india").screen_min_close == 10.0
    assert get_market("india").rs_breakout_min_close == 50.0
    assert TV_MARKETS == {name: market.tv_market for name, market in MARKETS.items()}


def test_resolve_as_of_normalizes_datetime_date_and_default() -> None:
    assert resolve_as_of(datetime(2026, 7, 9, 12, 30)) == date(2026, 7, 9)
    assert resolve_as_of(date(2026, 7, 8)) == date(2026, 7, 8)
    assert resolve_as_of(None) == date.today()


def test_get_price_fetcher_prefers_injected_fetcher() -> None:
    injected = DummyFetcher()

    def builder(**_kwargs: object) -> PriceFetcher:  # pragma: no cover
        raise AssertionError("builder should not be called")

    assert get_price_fetcher(injected, builder=builder) is injected


def test_get_price_fetcher_calls_builder_with_kwargs() -> None:
    calls: list[dict[str, object]] = []
    built = DummyFetcher()

    def builder(**kwargs: object) -> PriceFetcher:
        calls.append(kwargs)
        return built

    assert get_price_fetcher(None, builder=builder, refresh=True) is built
    assert calls == [{"refresh": True}]


def test_market_and_as_of_options_wrap_click_command() -> None:
    @click.command()
    @market_option(default="us", show_default=True)
    @as_of_option()
    def command(market: str, as_of_arg: datetime | None) -> None:
        click.echo(f"{market}:{resolve_as_of(as_of_arg).isoformat()}")

    result = CliRunner().invoke(command, ["--market", "india", "--as-of", "2026-07-09"])

    assert result.exit_code == 0
    assert result.output.strip() == "india:2026-07-09"
