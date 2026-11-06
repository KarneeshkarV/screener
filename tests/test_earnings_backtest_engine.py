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


def test_earnings_flat_cost_model_uses_per_fill_commission(monkeypatch) -> None:
    """Flat commission is charged once at entry and once at exit."""
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
        entry * 0.001 + exit_ * 0.001, abs=1e-6
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


def test_earnings_total_fees_count_only_taken_trades(monkeypatch) -> None:
    falling = _bars(IDX)
    falling["close"] = [200.0 - i for i in range(len(IDX))]
    data = {"UP": _bars(IDX), "DOWN": falling}
    events = pd.concat(
        [_events(IDX[30].date()).assign(ticker=ticker) for ticker in data],
        ignore_index=True,
    )
    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: events
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: data)

    trades = run_earnings_backtest(
        market="us",
        strategy="price_momentum",
        min_score=0.55,
        commission_bps=10.0,
        slippage_bps=0.0,
        tickers=list(data),
    )

    assert [t.passed_filter for t in trades] == [True, False]
    summary = compute_backtest_summary(trades, strategy="price_momentum")
    assert float(summary["total_fees"]) == pytest.approx(
        sum(trades[0].details["fees"].values()), abs=1e-6
    )


def test_earnings_window_holds_signal_history_for_the_first_event(
    monkeypatch,
) -> None:
    # Holidays leave fewer than 21 sessions in a 30-day pad, so the first
    # event used to score 0 for missing data.
    full = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=400)
    first, second = full[200], full[300]
    holidays = full[full.get_loc(first) - 15 : full.get_loc(first) - 12]
    bars = _bars(full.drop(holidays))

    def fetch(tickers, start, end, **kwargs):
        return {t: bars.loc[str(start) : str(end)] for t in tickers}

    events = pd.concat(
        [_events(first.date()), _events(second.date())], ignore_index=True
    )
    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: events
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", fetch)

    trades = run_earnings_backtest(
        market="us",
        strategy="price_momentum",
        min_score=0.55,
        tickers=["AAA"],
    )

    assert [t.passed_filter for t in trades] == [True, True]


def test_report_before_the_close_exits_at_the_prior_close(monkeypatch) -> None:
    # yfinance stamps a pre-market report at 06:00 New York. That day's close
    # already prices the result, so the pre-earnings trade must exit one bar
    # earlier; exiting at it booked the report-day jump as drift.
    from screener.earnings_backtest import earnings_dates

    report_day = IDX[30]
    bars = _bars(IDX)
    bars.loc[report_day, ["open", "high", "low", "close"]] = [
        150.0,
        151.0,
        149.0,
        150.0,
    ]
    stamps = pd.DatetimeIndex(
        [report_day + pd.Timedelta(hours=6)], tz="America/New_York"
    )
    reports = pd.DataFrame(
        {"EPS Estimate": [1.0], "Reported EPS": [1.1], "Surprise(%)": [10.0]},
        index=stamps,
    )
    monkeypatch.setattr(earnings_dates, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        earnings_dates, "cached_json_call", lambda *a, **kw: kw["fetch"]()
    )
    monkeypatch.setattr(
        earnings_dates.yf,
        "Ticker",
        lambda ticker: type("T", (), {"earnings_dates": reports})(),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": bars}
    )

    trades = run_earnings_backtest(
        market="us",
        strategy="price_momentum",
        days_before=1,
        min_score=0.0,
        tickers=["AAA"],
    )

    assert len(trades) == 1
    assert trades[0].earnings_date == report_day.date()
    assert trades[0].entry_date == IDX[28].date()
    assert trades[0].exit_date == IDX[29].date()
    assert trades[0].exit_price < 150.0


def test_report_after_the_close_still_exits_at_its_own_close(monkeypatch) -> None:
    from screener.earnings_backtest import earnings_dates

    report_day = IDX[30]
    stamps = pd.DatetimeIndex(
        [report_day + pd.Timedelta(hours=16)], tz="America/New_York"
    )
    reports = pd.DataFrame(
        {"EPS Estimate": [1.0], "Reported EPS": [1.1], "Surprise(%)": [10.0]},
        index=stamps,
    )
    monkeypatch.setattr(earnings_dates, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        earnings_dates, "cached_json_call", lambda *a, **kw: kw["fetch"]()
    )
    monkeypatch.setattr(
        earnings_dates.yf,
        "Ticker",
        lambda ticker: type("T", (), {"earnings_dates": reports})(),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars(IDX)}
    )

    trades = run_earnings_backtest(
        market="us",
        strategy="price_momentum",
        days_before=1,
        min_score=0.0,
        tickers=["AAA"],
    )

    assert [(t.entry_date, t.exit_date) for t in trades] == [
        (IDX[29].date(), report_day.date())
    ]
