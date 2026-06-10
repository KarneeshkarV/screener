"""Tests for the PEAD (post-earnings-announcement drift) backtest."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from click.testing import CliRunner

import screener.earnings_backtest.pead as pead_module
from screener.cli import cli
from screener.earnings_backtest.pead import (
    PeadTrade,
    compute_pead_summary,
    run_pead_backtest,
    surprise_quintiles,
)
from tests.conftest import StubPriceFetcher

EVENT_COLUMNS = [
    "ticker",
    "earnings_date",
    "eps_estimate",
    "reported_eps",
    "surprise_pct",
]

# Anchor synthetic data to today so the years-based cutoff never bites.
IDX = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)


def _frame(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic bars: open[i] = 100 + i, close[i] = open[i] + 1."""
    opens = [100.0 + i for i in range(len(idx))]
    closes = [o + 1.0 for o in opens]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [c + 1.0 for c in closes],
            "low": [o - 1.0 for o in opens],
            "close": closes,
            "volume": [10_000.0] * len(idx),
        },
        index=idx,
    )


def _events(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _event(ticker: str, earnings_date: date, surprise: float | None) -> dict:
    return {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "eps_estimate": 1.0,
        "reported_eps": 1.1,
        "surprise_pct": surprise,
    }


def _patch_events(monkeypatch, events_df: pd.DataFrame) -> None:
    monkeypatch.setattr(
        pead_module, "collect_earnings_events", lambda *a, **kw: events_df
    )


def _run(
    monkeypatch,
    events_df: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    **kwargs,
) -> list[PeadTrade]:
    _patch_events(monkeypatch, events_df)
    defaults = {
        "market": "us",
        "years": 3,
        "min_surprise": 5.0,
        "hold_days": 5,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "tickers": sorted(data),
        "fetcher": StubPriceFetcher(data),
    }
    defaults.update(kwargs)
    return run_pead_backtest(**defaults)


# ── Engine ───────────────────────────────────────────────────────────────


def test_pead_filters_by_surprise_and_enters_next_open(monkeypatch):
    ed = IDX[9].date()
    events = _events(
        [
            _event("AAA", ed, 10.0),  # taken
            _event("BBB", ed, 2.0),  # below threshold
            _event("CCC", ed, None),  # no surprise data
        ]
    )
    data = {t: _frame(IDX) for t in ("AAA", "BBB", "CCC")}
    trades = _run(monkeypatch, events, data, min_surprise=5.0, hold_days=5)

    assert [t.ticker for t in trades] == ["AAA"]
    trade = trades[0]
    assert trade.earnings_date == ed
    # Entry at the open of the next trading day; exit at the close 5
    # sessions later (entry day counts as day 1).
    assert trade.entry_date == IDX[10].date()
    assert trade.exit_date == IDX[14].date()
    assert trade.entry_price == pytest.approx(110.0)
    assert trade.exit_price == pytest.approx(115.0)
    assert trade.return_pct == pytest.approx((115.0 / 110.0 - 1.0) * 100, abs=1e-3)
    assert trade.surprise_pct == pytest.approx(10.0)
    assert trade.holding_days == 5
    assert trade.passed_filter is True


def test_pead_weekend_announcement_enters_next_trading_day(monkeypatch):
    fridays = IDX[IDX.weekday == 4]
    friday = fridays[1]
    saturday = (friday + pd.Timedelta(days=1)).date()
    events = _events([_event("AAA", saturday, 25.0)])
    trades = _run(monkeypatch, events, {"AAA": _frame(IDX)}, hold_days=3)

    assert len(trades) == 1
    next_bar = IDX[IDX > pd.Timestamp(saturday)][0]
    assert trades[0].entry_date == next_bar.date()


def test_pead_skips_incomplete_drift_window(monkeypatch):
    ed = IDX[-3].date()  # only two bars left after the announcement
    events = _events([_event("AAA", ed, 50.0)])
    trades = _run(monkeypatch, events, {"AAA": _frame(IDX)}, hold_days=40)

    assert trades == []


def test_pead_skips_tickers_without_price_data(monkeypatch):
    ed = IDX[9].date()
    events = _events([_event("AAA", ed, 10.0), _event("ZZZ", ed, 10.0)])
    data = {"AAA": _frame(IDX), "ZZZ": pd.DataFrame()}
    trades = _run(monkeypatch, events, data, hold_days=5)

    assert [t.ticker for t in trades] == ["AAA"]


def test_pead_applies_slippage_and_commission(monkeypatch):
    ed = IDX[9].date()
    events = _events([_event("AAA", ed, 10.0)])
    trades = _run(
        monkeypatch,
        events,
        {"AAA": _frame(IDX)},
        hold_days=5,
        commission_bps=10.0,
        slippage_bps=5.0,
    )

    entry = 110.0 * (1 + 5 / 10_000)
    exit_ = 115.0 * (1 - 5 / 10_000)
    expected = ((exit_ / entry - 1.0) - 10 / 10_000) * 100
    assert trades[0].return_pct == pytest.approx(expected, abs=1e-3)
    assert trades[0].details["raw_return_pct"] > trades[0].return_pct


def test_pead_ignores_events_outside_lookback(monkeypatch):
    old = date.today() - timedelta(days=5 * 365)
    events = _events([_event("AAA", old, 10.0)])
    trades = _run(monkeypatch, events, {"AAA": _frame(IDX)}, years=3)

    assert trades == []


def test_pead_rejects_non_positive_hold_days(monkeypatch):
    _patch_events(monkeypatch, _events([]))
    with pytest.raises(ValueError, match="hold_days"):
        run_pead_backtest("us", hold_days=0, tickers=["AAA"])


# ── Summary / quintiles ──────────────────────────────────────────────────


def _trade(surprise: float, ret: float) -> PeadTrade:
    return PeadTrade(
        ticker="AAA",
        earnings_date=date(2024, 1, 10),
        entry_date=date(2024, 1, 11),
        exit_date=date(2024, 3, 8),
        entry_price=100.0,
        exit_price=100.0 * (1 + ret / 100),
        return_pct=ret,
        surprise_pct=surprise,
        holding_days=40,
    )


def test_surprise_quintiles_orders_buckets_by_surprise():
    trades = [_trade(float(i * 2), float(i)) for i in range(1, 11)]
    quintiles = surprise_quintiles(trades)

    assert sorted(quintiles) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert all(q["trades"] == 2 for q in quintiles.values())
    assert quintiles["Q5"]["avg_surprise_pct"] > quintiles["Q1"]["avg_surprise_pct"]
    assert quintiles["Q5"]["avg_return_pct"] > quintiles["Q1"]["avg_return_pct"]


def test_surprise_quintiles_degrades_gracefully():
    assert surprise_quintiles([]) == {}
    assert surprise_quintiles([_trade(5.0, 1.0)] * 4) == {}
    # Identical surprises: qcut cannot form bins
    assert surprise_quintiles([_trade(5.0, 1.0)] * 10) == {}


def test_compute_pead_summary_reports_drift_stats():
    trades = [_trade(float(i * 2), float(i - 3)) for i in range(1, 11)]
    summary = compute_pead_summary(trades, min_surprise=5.0, hold_days=40)

    assert summary["total_events"] == 10
    assert summary["trades_taken"] == 10
    assert summary["min_surprise_pct"] == 5.0
    assert summary["hold_days"] == 40
    assert summary["win_rate"] == pytest.approx(70.0)
    assert summary["avg_return_pct"] == pytest.approx(2.5)
    assert summary["median_return_pct"] == pytest.approx(2.5)
    assert "surprise_quintiles" in summary


# ── CLI ──────────────────────────────────────────────────────────────────


def _patch_cli_inputs(monkeypatch) -> date:
    ed = IDX[9].date()
    events = _events([_event("AAA", ed, 12.5)])
    _patch_events(monkeypatch, events)
    monkeypatch.setattr(
        pead_module, "fetch_price_data", lambda *a, **kw: {"AAA": _frame(IDX)}
    )
    return ed


def test_cli_earnings_pead_prints_summary(monkeypatch):
    _patch_cli_inputs(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["earnings-pead", "--tickers", "AAA", "--hold-days", "5"],
        catch_exceptions=False,
    )

    assert res.exit_code == 0
    assert "PEAD Backtest Summary" in res.output
    assert "AAA" in res.output


def test_cli_earnings_pead_csv_ledger(monkeypatch):
    ed = _patch_cli_inputs(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["earnings-pead", "--tickers", "AAA", "--hold-days", "5", "--csv"],
        catch_exceptions=False,
    )

    assert res.exit_code == 0
    header = (
        "ticker,earnings_date,entry_date,exit_date,"
        "entry_price,exit_price,return_pct,surprise_pct,holding_days"
    )
    assert header in res.output
    assert f"AAA,{ed.isoformat()}" in res.output


def test_cli_earnings_pead_no_events(monkeypatch):
    _patch_events(monkeypatch, _events([]))
    runner = CliRunner()
    res = runner.invoke(
        cli, ["earnings-pead", "--tickers", "AAA"], catch_exceptions=False
    )

    assert res.exit_code == 0
    assert "No qualifying PEAD events" in res.output
