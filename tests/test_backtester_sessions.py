"""Session-aware intraday exits (--intraday-only)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from screener.backtester.models import BacktestConfig
from screener.backtester.sessions import is_session_last, market_timezone, session_dates
from screener.cli import cli
from tests.backtest_helpers import simulate_single_ticker
from tests.conftest import StubPriceFetcher


def _us_5m_index(*, sessions: int = 2, bars_per: int = 6) -> pd.DatetimeIndex:
    """Naive-UTC 5m stamps for US regular sessions (14:30–… UTC = 09:30 ET)."""
    # 2024-03-04/05 are EST (UTC-5); DST starts 2024-03-10.
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-03-04 14:30:00")
    for _ in range(sessions):
        stamps.extend(day + pd.Timedelta(minutes=5 * b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _india_5m_index(*, sessions: int = 2, bars_per: int = 6) -> pd.DatetimeIndex:
    """Naive-UTC 5m stamps for India sessions (03:45 UTC = 09:15 IST)."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-03-04 03:45:00")
    for _ in range(sessions):
        stamps.extend(day + pd.Timedelta(minutes=5 * b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _rising_frame(index: pd.DatetimeIndex, start_px: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = pd.Series(np.linspace(start_px, start_px + n, n), index=index, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 0.5)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([openp, close], axis=1).min(axis=1) - 0.5
    vol = pd.Series(50_000.0, index=index, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 5),
        hold=100,
        top=1,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=None,
        tickers=None,
        interval="5m",
        entry_order_type="moc",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_market_timezone_us_and_india():
    assert market_timezone("us") == "America/New_York"
    assert market_timezone("india") == "Asia/Kolkata"


def test_is_session_last_two_us_sessions():
    idx = _us_5m_index(sessions=2, bars_per=6)
    mask = is_session_last(idx, "America/New_York")
    assert mask.dtype == bool
    assert len(mask) == 12
    # Last bar of each session + final bar.
    assert list(np.where(mask)[0]) == [5, 11]
    labels = session_dates(idx, "America/New_York")
    assert labels[0] == date(2024, 3, 4)
    assert labels[6] == date(2024, 3, 5)


def test_is_session_last_india_session():
    idx = _india_5m_index(sessions=2, bars_per=4)
    mask = is_session_last(idx, "Asia/Kolkata")
    assert list(np.where(mask)[0]) == [3, 7]


def test_is_session_last_half_day():
    # Session 1 full (6 bars), session 2 half-day (3 bars).
    s1 = _us_5m_index(sessions=1, bars_per=6)
    s2_start = pd.Timestamp("2024-03-05 14:30:00")
    s2 = [s2_start + pd.Timedelta(minutes=5 * b) for b in range(3)]
    idx = pd.DatetimeIndex(list(s1) + s2)
    mask = is_session_last(idx, "America/New_York")
    assert list(np.where(mask)[0]) == [5, 8]


def test_is_session_last_empty():
    idx = pd.DatetimeIndex([])
    assert list(is_session_last(idx, "America/New_York")) == []


def test_data_policy_rejects_intraday_only_on_daily():
    with pytest.raises(ValidationError, match="intraday_only"):
        _cfg(interval="1d", intraday_only=True)
    # Intraday + flag is fine.
    ok = _cfg(interval="15m", intraday_only=True)
    assert ok.intraday_only is True


def test_simulate_single_ticker_session_exit():
    bars = _rising_frame(_us_5m_index(sessions=2, bars_per=6))
    # MOC/MOO enter at signal_idx+1. Signal at bar 1 → entry bar 2; session-last
    # of session 1 is bar 5.
    cfg = _cfg(intraday_only=True, hold=100, entry_order_type="moc")
    outcome = simulate_single_ticker(bars, signal_idx=1, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "session"
    assert outcome.trade.exit_date == bars.index[5].to_pydatetime()


def test_simulate_single_ticker_without_flag_holds_past_session():
    bars = _rising_frame(_us_5m_index(sessions=2, bars_per=6))
    cfg = _cfg(intraday_only=False, hold=100, entry_order_type="moc")
    outcome = simulate_single_ticker(bars, signal_idx=1, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "eod"
    assert outcome.trade.exit_date == bars.index[-1].to_pydatetime()


def test_entry_guard_skips_session_last_bar():
    bars = _rising_frame(_us_5m_index(sessions=2, bars_per=6))
    # Session-last of session 1 is bar 5. signal_idx=4 → entry on bar 5 → skip.
    cfg = _cfg(intraday_only=True, hold=100, entry_order_type="moc")
    outcome = simulate_single_ticker(bars, signal_idx=4, cfg=cfg)
    assert outcome.trade is None
    assert outcome.warning is not None
    assert "session-last" in outcome.warning


def test_cli_intraday_only_error_on_daily():
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--entry",
            "close > 0",
            "--interval",
            "1d",
            "--intraday-only",
            "--start",
            "2024-03-01",
            "--end",
            "2024-03-10",
        ],
        obj=StubPriceFetcher({}),
    )
    assert res.exit_code != 0
    assert "intraday_only" in res.output


def test_cli_intraday_only_flag_accepted_for_intraday():
    """Smoke: flag is recognized (run may fail later on empty data, not on parse)."""
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--entry",
            "close > 0",
            "--interval",
            "15m",
            "--intraday-only",
            "--start",
            "2024-03-04",
            "--end",
            "2024-03-05",
            "--hold",
            "2",
        ],
        obj=StubPriceFetcher({}),
    )
    # Empty universe prices → clean exit or usage; must not reject the flag itself.
    assert "No such option" not in res.output
    assert "intraday_only requires" not in res.output
