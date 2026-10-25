"""Hand-calculated short cash flows and causal signal tests for Connors research."""

import numpy as np
import pandas as pd
import pytest

from scripts.connors_crash_research import (
    ConnorsMarketData,
    connors_crash_indicators,
    connors_nominal_bars,
    lean_wilder_rsi,
    load_connors_market,
    simulate_connors_crash,
)


def connors_test_market(count=6, symbols=("AAA",)):
    """Supply synthetic daily bars with explicit prior-day signals."""
    calendar = pd.bdate_range("2024-01-01", periods=count)
    shape = (count, len(symbols))
    arrays = {
        key: np.full(shape, value, dtype=float)
        for key, value in {
            "open": 100,
            "high": 100,
            "low": 99,
            "close": 100,
            "turnover": 2e6,
            "dividend": 0,
            "split": 1,
            "crsi": 50,
            "volatility": 110,
        }.items()
    }
    return ConnorsMarketData(
        "us",
        calendar,
        list(symbols),
        arrays,
        np.ones(shape, dtype=bool),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
    )


def test_connors_limit_persists_after_signal_expires_and_marks_short():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][2, 0] = 105
    data.arrays["close"][2:, 0] = 102
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.summary["orders_submitted"] == 1
    assert result.summary["entries_filled"] == 1
    assert result.summary["closed_trades"] == 0
    assert result.summary["final_equity"] == pytest.approx(100025)
    assert result.open_positions[0]["entry_notional"] == 2575
    assert result.summary["max_entry_wait_calendar_days"] == 1


def test_connors_signal_uses_previous_close_and_cover_next_open():
    data = connors_test_market()
    data.arrays["crsi"][1, 0] = 95
    data.arrays["high"][1:3, 0] = 104
    data.arrays["crsi"][2, 0] = 20
    data.arrays["open"][3, 0] = 90
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.trades[0]["entry_date"] == str(data.calendar[2].date())
    assert result.trades[0]["exit_date"] == str(data.calendar[3].date())
    assert result.trades[0]["pnl"] == 325
    assert result.summary["final_equity"] == 100325


def test_connors_exit_keeps_slot_occupied_until_next_morning():
    data = connors_test_market(symbols=("AAA", "BBB"))
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["crsi"][1, 0] = 20
    data.arrays["crsi"][1, 1] = 99
    result = simulate_connors_crash(
        data, data.calendar[1], data.calendar[-1], max_shorts=1
    )
    assert result.summary["entries_filled"] == 1
    assert result.summary["orders_submitted"] == 1


def test_connors_split_dividend_borrow_and_fees_reconcile():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["close"][1, 0] = 104
    data.arrays["split"][2, 0] = 2
    data.arrays["dividend"][2, 0] = 1
    for key in ("open", "high", "low", "close"):
        data.arrays[key][2:, 0] = 50
    data.arrays["crsi"][2, 0] = 20
    result = simulate_connors_crash(
        data, data.calendar[1], data.calendar[3], annual_borrow=0.365, execution_bps=10
    )
    trade = result.trades[0]
    assert trade["exit_quantity"] == 50
    assert trade["dividends_owed"] == 50
    assert trade["borrow_cost"] == pytest.approx(2.6 + 2.5)
    assert trade["execution_cost"] == pytest.approx(2.575 + 2.5)
    assert trade["pnl"] == pytest.approx(75 - 50 - 5.1 - 5.075)
    assert result.summary["reconciliation_error"] == pytest.approx(0, abs=1e-8)


def test_connors_missing_exit_does_not_invent_fill():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["crsi"][1, 0] = 20
    for key in ("open", "high", "low", "close"):
        data.arrays[key][2:4, 0] = np.nan
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[3])
    assert result.summary["closed_trades"] == 0
    assert result.summary["unfilled_exit_days"] == 2
    assert result.summary["stale_position_days"] == 2


def test_connors_liquidity_removal_forces_cover():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["turnover"][1, 0] = 100
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.trades[0]["exit_reason"] == "universe_removal"
    assert result.trades[0]["exit_date"] == str(data.calendar[2].date())


def test_connors_nominal_price_reconstruction_handles_splits_and_adjustment():
    frame = pd.DataFrame(
        {
            "open": [36.0],
            "high": [40.0],
            "low": [35.0],
            "close": [38.0],
            "volume": [100.0],
            "dollar_volume": [4000.0],
            "split_factor": [4.0],
            "dividend": [0.25],
            "stock_splits": [0.0],
        }
    )
    nominal = connors_nominal_bars(frame)
    assert nominal.close.iloc[0] == 160
    assert nominal.open.iloc[0] == pytest.approx(36 * 160 / 38)
    assert nominal.dividend.iloc[0] == 1
    assert nominal.signal_close.iloc[0] == 38
    assert nominal.turnover.iloc[0] == 4000


def test_connors_percent_rank_and_volatility_use_prior_100_and_99_returns():
    close = pd.Series(np.exp(np.cumsum(np.linspace(-0.05, 0.08, 105))))
    features = connors_crash_indicators(close)
    assert features.crsi.iloc[:99].isna().all()
    assert features.crsi.iloc[-1] > 99
    expected = np.std(np.diff(np.log(close.iloc[-100:])), ddof=0) * np.sqrt(252) * 100
    assert features.volatility.iloc[-1] == pytest.approx(expected)
    changed = close.copy()
    changed.iloc[-1] *= 2
    assert features.iloc[:-1].equals(connors_crash_indicators(changed).iloc[:-1])


def test_connors_loader_keeps_actions_when_quote_has_zero_turnover(tmp_path):
    calendar = pd.bdate_range("2024-01-01", periods=105)
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
            "dollar_volume": 10000.0,
            "split_factor": 1.0,
            "dividend": 0.0,
            "stock_splits": 0.0,
        },
        index=calendar,
    )
    frame.to_parquet(tmp_path / "SPY__full_turnover_v2.parquet")
    frame.loc[calendar[101], "volume"] = 0
    frame.loc[calendar[101], "dollar_volume"] = 0
    frame.loc[calendar[101], "stock_splits"] = 2
    frame.loc[calendar[101], "dividend"] = 1
    frame.to_parquet(tmp_path / "AAA__full_turnover_v2.parquet")
    membership = tmp_path / "membership.csv"
    pd.DataFrame({"effective_date": ["2024-01-01"], "symbol": ["AAA"]}).to_csv(
        membership, index=False
    )
    data = load_connors_market("us", tmp_path, membership, calendar[-1], calendar[0])
    assert np.isnan(data.arrays["close"][101, 0])
    assert data.arrays["split"][101, 0] == 2
    assert data.arrays["dividend"][101, 0] == 1
    assert not data.arrays["complete_window"][104, 0]


def test_connors_fmp_fallback_does_not_select_by_future_coverage(tmp_path):
    calendar = pd.bdate_range("2024-01-01", periods=110)
    cache = tmp_path / "prices"
    cache.mkdir()
    fmp = tmp_path / "fmp_prices"
    fmp.mkdir()
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
            "dollar_volume": 10000.0,
            "split_factor": 1.0,
            "dividend": 0.0,
            "stock_splits": 0.0,
        },
        index=calendar,
    )
    frame.to_parquet(cache / "SPY__full_turnover_v2.parquet")
    frame.iloc[50:].to_parquet(cache / "AAA__full_turnover_v2.parquet")
    frame.to_parquet(fmp / "fmp_AAA__full_turnover_v2.parquet")
    membership = tmp_path / "membership.csv"
    pd.DataFrame({"effective_date": ["2024-01-01"], "symbol": ["AAA"]}).to_csv(
        membership, index=False
    )
    data = load_connors_market(
        "us", cache, membership, calendar[-1], calendar[0], allow_fmp_proxy=True
    )
    assert data.coverage.iloc[0]["status"] == "available"
    assert data.coverage.iloc[0]["missing_member_sessions"] == 50
    (cache / "AAA__full_turnover_v2.parquet").unlink()
    fallback = load_connors_market(
        "us", cache, membership, calendar[-1], calendar[0], allow_fmp_proxy=True
    )
    assert fallback.coverage.iloc[0]["status"] == "fmp_total_return_proxy"
    assert fallback.evidence["classification"] == "observed_index_proxy"


def test_connors_new_order_uses_split_morning_price_basis():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["split"][1, 0] = 2
    for key in ("open", "low", "close"):
        data.arrays[key][1:, 0] = 50
    data.arrays["high"][1:, 0] = 53
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.orders[0]["quantity"] == 50
    assert result.orders[0]["limit"] == 51.5
    assert result.summary["entries_filled"] == 1


def test_connors_split_event_survives_missing_quote():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["split"][2, 0] = 2
    data.arrays["crsi"][1, 0] = 20
    for key in ("open", "high", "low", "close"):
        data.arrays[key][2, 0] = np.nan
        data.arrays[key][3:, 0] = 50
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.trades[0]["exit_quantity"] == 50
    assert result.trades[0]["pnl"] == 75


def test_connors_quality_gate_cancels_a_stale_pending_limit():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][3, 0] = 110
    data.arrays["entry_quality"] = np.ones_like(data.membership)
    data.arrays["entry_quality"][2:, 0] = False
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.summary["orders_submitted"] == 1
    assert result.summary["entries_filled"] == 0
    assert result.orders[-1]["event"] == "cancel_data_quality"


def test_connors_wilder_rsi_seed_is_hand_calculated():
    result = lean_wilder_rsi(np.array([10.0, 11.0, 13.0, 12.0, 14.0]), 3)
    assert result[3] == 75
    assert result[4] == pytest.approx(100 - 100 / 7)
