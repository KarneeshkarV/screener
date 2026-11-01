"""Bounded missing price-provider history warnings on the price-panel path."""

from __future__ import annotations

import pandas as pd

from screener.backtester.pine import parse
from screener.backtester.price_panel import (
    PricePanelInputs,
    _warn_missing_price_history,
    build_price_panel,
)
from tests.conftest import StubPriceFetcher, make_bars


def test_warn_missing_price_history_is_bounded_and_skips_partial_series() -> None:
    warnings: list[str] = []
    bars = {
        "AAA": make_bars(start="2024-01-01", n=5),
        "IPO": make_bars(start="2024-06-01", n=3),
        "GONE": pd.DataFrame(),
        "MISSING": pd.DataFrame(),
        "EMPTY2": pd.DataFrame(),
        "EMPTY3": pd.DataFrame(),
        "EMPTY4": pd.DataFrame(),
        "EMPTY5": pd.DataFrame(),
        "EMPTY6": pd.DataFrame(),
    }
    _warn_missing_price_history(bars, warnings)
    assert len(warnings) == 1
    message = warnings[0]
    assert message.startswith("7 requested symbols have no price-provider history")
    assert "EMPTY2" in message
    assert "+2 more" in message
    assert "AAA" not in message
    assert "GONE" not in message.split(":", 1)[1].split(".", 1)[0]
    assert "mid-window IPOs" in message


def test_build_price_panel_records_empty_provider_history_without_live_fetch() -> None:
    aaa = make_bars(start="2024-01-02", n=10)
    fetcher = StubPriceFetcher({"AAA": aaa, "SPY": aaa})
    warnings: list[str] = []
    inputs = PricePanelInputs(
        market="us",
        benchmark="SPY",
        tickers=("AAA", "BBB"),
        universe_file=None,
        membership_windows=(),
        dynamic_universe_size=None,
        max_universe=0,
        interval="1d",
        price_adjustment="full",
        strategy_name=None,
        fundamentals_provider=None,
    )
    panel = build_price_panel(
        inputs,
        fetcher,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=2,
        start_ts=pd.Timestamp("2024-01-03"),
        end_ts=pd.Timestamp("2024-01-12"),
        warnings=warnings,
    )
    assert panel.bars_by_tv["AAA"].empty is False
    assert panel.bars_by_tv["BBB"].empty is True
    assert any(
        "no price-provider history" in warning and "BBB" in warning
        for warning in warnings
    )
