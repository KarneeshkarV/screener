"""Backtest lab comparison tests."""

from __future__ import annotations

from datetime import date

from screener.backtester import lab

from tests.conftest import StubPriceFetcher, make_bars


def test_compare_payload_runs_multiple_named_strategies(monkeypatch):
    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    bars_b = make_bars(n=80, seed=22, open_base=50.0)
    spy = make_bars(n=80, seed=23, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    monkeypatch.setattr(lab, "build_price_fetcher", lambda auto_adjust=True: fetcher)

    payload = lab.compare_payload(
        market="us",
        strategies=["ema_trend", "breakout"],
        tickers=("AAA", "BBB"),
        start_date=bars_a.index[20].date(),
        end_date=bars_a.index[60].date(),
        hold=5,
        top=2,
        initial_capital=100_000,
    )

    assert [item["strategy"] for item in payload["results"]] == [
        "ema_trend",
        "breakout",
    ]
    assert payload["request"]["tickers"] == ("AAA", "BBB")
    assert all("metrics" in item for item in payload["results"])
    assert all("curves" in item for item in payload["results"])


def test_compare_payload_requires_strategy_and_ticker():
    try:
        lab.compare_payload(
            market="us",
            strategies=[],
            tickers=("AAA",),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
            hold=5,
            top=1,
            initial_capital=100_000,
        )
    except ValueError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("expected missing strategy error")
