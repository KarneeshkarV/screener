"""Tests for the earnings-drift backtest engine."""

from __future__ import annotations

from datetime import date

import pandas as pd

import screener.earnings_backtest.engine as engine_module
from screener.earnings_backtest.engine import run_earnings_backtest


IDX = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)


def _bars(idx: pd.DatetimeIndex) -> pd.DataFrame:
    close = [100.0 + i for i in range(len(idx))]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10_000.0] * len(idx),
        },
        index=idx,
    )


def _events(event_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "earnings_date": event_date,
                "eps_estimate": 1.0,
                "reported_eps": 1.1,
                "surprise_pct": 10.0,
            }
        ]
    )


def test_earnings_backtest_skips_current_snapshot_signals_for_historical_entries(
    monkeypatch,
) -> None:
    event_date = IDX[30].date()
    data = {"AAA": _bars(IDX)}

    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: _events(event_date)
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: data)

    def fail_live_snapshot_fetch(*args, **kwargs):
        raise AssertionError("historical backtest must not fetch live snapshot signals")

    monkeypatch.setattr(
        engine_module, "fetch_analyst_sentiment", fail_live_snapshot_fetch
    )
    monkeypatch.setattr(engine_module, "fetch_iv_sentiment", fail_live_snapshot_fetch)

    trades = run_earnings_backtest(
        market="us",
        years=3,
        strategy="combined_score",
        days_before=1,
        min_score=0.0,
        tickers=["AAA"],
    )

    assert len(trades) == 1
    trade = trades[0]
    assert set(trade.details["scores"]) == {"price_momentum", "volume_surge"}
    assert trade.details["signals"]["analyst_sentiment"]["reason"] == (
        "current_snapshot_unavailable_for_historical_entry"
    )
    assert trade.details["signals"]["iv_sentiment"]["reason"] == (
        "current_snapshot_unavailable_for_historical_entry"
    )
