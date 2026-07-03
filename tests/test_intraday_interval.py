"""Phase 1 intraday-interval support.

Covers the three seams threaded end-to-end for intraday bars:

* ``data.py`` normalization keeps intraday timestamps distinct (daily still
  collapses to one bar per day) and cache keys are namespaced per interval;
* ``metrics.py`` annualizes over the right number of periods per interval while
  the daily default stays byte-for-byte unchanged;
* the rolling engine run over synthetic 15m bars does not collapse the bars,
  and the emitted trades carry full intraday timestamps entered at signal+1.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from screener.backtester.data import (
    YFinancePriceFetcher,
    _naive_normalized_index,
    _normalize_frame,
)
from screener.backtester.metrics import (
    compute_metrics,
    periods_per_year_for_interval,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from tests.conftest import StubPriceFetcher


# --------------------------------------------------------------------------- #
# Synthetic intraday bars
# --------------------------------------------------------------------------- #
_BARS_PER_SESSION = 26  # 09:30..15:45 at 15m, matching a US regular session


def _intraday_index(sessions: int) -> pd.DatetimeIndex:
    """A tz-naive 15m index of ``sessions`` regular sessions."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-03-04 09:30")  # a Monday
    for _ in range(sessions):
        base = day
        stamps.extend(base + pd.Timedelta(minutes=15 * b) for b in range(_BARS_PER_SESSION))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _rising_frame(index: pd.DatetimeIndex, start_px: float, volume: float) -> pd.DataFrame:
    """A monotonically rising OHLCV frame so the entry signal fires every bar."""
    n = len(index)
    close = pd.Series(np.linspace(start_px, start_px + n, n), index=index, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 0.5)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([openp, close], axis=1).min(axis=1) - 0.5
    vol = pd.Series(volume, index=index, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# --------------------------------------------------------------------------- #
# data.py: normalization + cache keys
# --------------------------------------------------------------------------- #
def test_intraday_normalization_preserves_times():
    idx = _intraday_index(sessions=2)
    normalized = _naive_normalized_index(idx, "15m")
    # Times are preserved (not truncated to midnight) and every bar is distinct.
    assert list(normalized) == list(idx)
    assert normalized.tz is None
    assert len(normalized.normalize().unique()) == 2  # two distinct calendar days


def test_daily_normalization_still_collapses_to_midnight():
    idx = _intraday_index(sessions=2)
    normalized = _naive_normalized_index(idx, "1d")
    # Daily path truncates every timestamp to midnight (legacy behaviour).
    assert (normalized == normalized.normalize()).all()
    assert set(normalized.time) == {datetime(2024, 1, 1).time()}


def test_normalize_frame_does_not_collapse_intraday_bars():
    idx = _intraday_index(sessions=2)
    frame = _rising_frame(idx, 100.0, 10_000.0)
    frame.columns = [c.capitalize() for c in frame.columns]  # yfinance-style names

    intraday = _normalize_frame(frame, "15m")
    assert len(intraday) == 2 * _BARS_PER_SESSION  # nothing dedup-collapsed

    daily = _normalize_frame(frame, "1d")
    assert len(daily) == 2  # one surviving bar per calendar day (keep last)


def test_cache_key_namespaces_intraday_but_not_daily():
    daily = YFinancePriceFetcher(interval="1d")
    assert daily._cache_key("AAPL") == "AAPL"

    daily_raw = YFinancePriceFetcher(interval="1d", auto_adjust=False)
    assert daily_raw._cache_key("AAPL") == "AAPL__raw"

    intraday = YFinancePriceFetcher(interval="15m")
    assert intraday._cache_key("AAPL") == "AAPL__15m"

    intraday_raw = YFinancePriceFetcher(interval="15m", auto_adjust=False)
    assert intraday_raw._cache_key("AAPL") == "AAPL__15m__raw"


# --------------------------------------------------------------------------- #
# metrics.py: annualization factor
# --------------------------------------------------------------------------- #
def test_periods_per_year_mapping():
    assert periods_per_year_for_interval("1d") == 252
    assert periods_per_year_for_interval("1h") == 252 * 7
    assert periods_per_year_for_interval("30m") == 252 * 13
    assert periods_per_year_for_interval("15m") == 252 * 26
    assert periods_per_year_for_interval("5m") == 252 * 78
    assert periods_per_year_for_interval("1m") == 252 * 390


def test_metrics_daily_default_unchanged():
    equity = pd.Series(
        np.linspace(100_000.0, 130_000.0, 60),
        index=pd.bdate_range("2024-01-01", periods=60),
        dtype=float,
    )
    benchmark = pd.Series(
        np.linspace(400.0, 440.0, 60), index=equity.index, dtype=float
    )
    default = compute_metrics(equity, benchmark, [], 1)
    explicit_daily = compute_metrics(equity, benchmark, [], 1, periods_per_year=252)
    assert default == explicit_daily

    intraday = compute_metrics(
        equity, benchmark, [], 1, periods_per_year=periods_per_year_for_interval("15m")
    )
    # Scaling the annualization factor changes the annualized figures.
    assert intraday["sharpe"] != pytest.approx(default["sharpe"])
    assert intraday["vol_annual"] != pytest.approx(default["vol_annual"])


# --------------------------------------------------------------------------- #
# Rolling engine over intraday bars
# --------------------------------------------------------------------------- #
def _intraday_cfg() -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=datetime(2024, 3, 5, 15, 45),
        interval="15m",
        hold=3,  # 3 bars, not 3 days
        top=1,
        strategy_name=None,
        entry_expr="close > sma(close, 2)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA",),
    )


def test_rolling_intraday_carries_timestamps_and_does_not_collapse():
    index = _intraday_index(sessions=2)
    data = {
        "AAA": _rising_frame(index, 100.0, 500_000.0),
        "SPY": _rising_frame(index, 400.0, 1_000_000.0),
    }
    fetcher = StubPriceFetcher(data)
    cfg = _intraday_cfg()

    result = run_rolling_backtest(
        cfg,
        fetcher,
        start_date=index[0].date(),
        end_date=index[-1].date(),
    )

    assert result.trades, "expected at least one intraday trade"

    # Bars are NOT collapsed: the simulation calendar (equity curve index) has
    # intraday resolution — far more points than the 2 calendar days spanned.
    assert len(result.equity_curve) > 2 * 10
    assert any(ts.time() != datetime(2024, 1, 1).time() for ts in result.equity_curve.index)

    bars = data["AAA"]
    for trade in result.trades:
        # Every stamp is a full datetime carrying a real intraday time-of-day.
        assert isinstance(trade.entry_date, datetime)
        assert isinstance(trade.exit_date, datetime)
        assert isinstance(trade.signal_date, datetime)
        assert trade.entry_date.time() >= datetime(2024, 3, 4, 9, 30).time()

        # Entry fires on the bar immediately after the signal bar.
        sig_pos = bars.index.get_loc(pd.Timestamp(trade.signal_date))
        assert pd.Timestamp(trade.entry_date) == bars.index[sig_pos + 1]

    # First trade: rising series + hold=3 with no stop/target -> a time exit
    # exactly three bars after entry.
    first = result.trades[0]
    entry_pos = bars.index.get_loc(pd.Timestamp(first.entry_date))
    exit_pos = bars.index.get_loc(pd.Timestamp(first.exit_date))
    assert first.exit_reason == "time"
    assert exit_pos - entry_pos == cfg.hold


def test_rolling_daily_stamps_stay_dates():
    """The daily engine keeps emitting plain ``date`` stamps (unchanged)."""
    index = pd.bdate_range("2024-01-01", periods=30)
    data = {
        "AAA": _rising_frame(index, 100.0, 500_000.0),
        "SPY": _rising_frame(index, 400.0, 1_000_000.0),
    }
    cfg = _intraday_cfg().model_copy(
        update={"interval": "1d", "as_of": date(2024, 2, 9)}
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=index[0].date(),
        end_date=index[-1].date(),
    )
    assert result.trades
    for trade in result.trades:
        assert isinstance(trade.entry_date, date)
        assert not isinstance(trade.entry_date, datetime)
