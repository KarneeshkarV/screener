"""Tests for the earnings-drift backtest engine."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import screener.earnings_backtest.engine as engine_module
from screener.backtester.execution import net_round_trip_return
from screener.earnings_backtest.engine import run_earnings_backtest
from screener.earnings_backtest.metrics import compute_backtest_summary


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


def _patch_single_event(monkeypatch, event_date: date | None = None) -> date:
    ed = event_date or IDX[30].date()
    data = {"AAA": _bars(IDX)}
    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: _events(ed)
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: data)

    def fail_live_snapshot_fetch(*args, **kwargs):
        raise AssertionError("historical backtest must not fetch live snapshot signals")

    monkeypatch.setattr(
        engine_module, "fetch_analyst_sentiment", fail_live_snapshot_fetch
    )
    monkeypatch.setattr(engine_module, "fetch_iv_sentiment", fail_live_snapshot_fetch)
    return ed


def test_earnings_backtest_skips_current_snapshot_signals_for_historical_entries(
    monkeypatch,
) -> None:
    _patch_single_event(monkeypatch)

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


def test_earnings_flat_cost_model_parity_with_legacy_round_trip(monkeypatch) -> None:
    """cost_model='flat' must match the legacy single round-trip commission drag."""
    _patch_single_event(monkeypatch)

    trades = run_earnings_backtest(
        market="us",
        years=3,
        strategy="price_momentum",
        days_before=1,
        min_score=0.0,
        commission_bps=10.0,
        cost_model="flat",
        slippage_bps=0.0,
        tickers=["AAA"],
    )
    assert len(trades) == 1
    trade = trades[0]
    entry = trade.entry_price
    exit_ = trade.exit_price
    raw, net = net_round_trip_return(entry, exit_, 10.0)
    assert trade.details["raw_return_pct"] == pytest.approx(raw * 100, abs=1e-4)
    assert trade.return_pct == pytest.approx(net * 100, abs=1e-4)
    assert "commission" in trade.details["fees"]
    assert trade.details["fees"]["commission"] == pytest.approx(
        entry * 0.0005 + exit_ * 0.0005, abs=1e-6
    )


def test_earnings_india_cost_model_applies_more_cost_than_flat_zero(
    monkeypatch,
) -> None:
    _patch_single_event(monkeypatch)
    kwargs = dict(
        market="us",
        years=3,
        strategy="price_momentum",
        days_before=1,
        min_score=0.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        tickers=["AAA"],
    )
    flat = run_earnings_backtest(**kwargs, cost_model="flat")
    india = run_earnings_backtest(**kwargs, cost_model="india")
    assert len(flat) == 1 and len(india) == 1
    assert india[0].return_pct < flat[0].return_pct
    assert india[0].details["raw_return_pct"] == pytest.approx(
        flat[0].details["raw_return_pct"]
    )
    # India statutory components show up in the fee breakdown.
    assert "stt" in india[0].details["fees"]
    summary = compute_backtest_summary(india, strategy="price_momentum")
    assert float(summary["total_fees"]) > 0.0
    assert float(summary["fee_stt"]) > 0.0
