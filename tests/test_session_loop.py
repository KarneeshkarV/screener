"""Session-loop generalization of the day loop.

``run_day_loop`` groups an intraday bar calendar into exchange sessions keyed
on :func:`screener.backtester.sessions.is_session_last` (positional, so
half-days end at their actual last bar), running the same per-bar skeleton in
the same order as the flat loop plus ``DayLoop.flatten_at_session_end`` on each
session's last bar — the driver-level guarantee that ``intraday_only`` runs
never hold a position overnight.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.core import _SlotState
from screener.backtester.day_loop import DayLoop, run_day_loop
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.rolling_simulation import run_rolling_backtest
from tests.conftest import StubPriceFetcher


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 4),
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
        interval="5m",
        intraday_only=True,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _us_5m_index(day: str, bars: int, start: str = "14:30") -> pd.DatetimeIndex:
    base = pd.Timestamp(f"{day} {start}")
    return pd.DatetimeIndex([base + pd.Timedelta(minutes=5 * b) for b in range(bars)])


def _frame(index: pd.DatetimeIndex, start_px: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = pd.Series(np.linspace(start_px, start_px + n, n), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": 10_000.0,
        },
        index=index,
    )


def _state(bars: pd.DataFrame, ticker: str = "AAA", entry_idx: int = 0) -> _SlotState:
    return _SlotState(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=bars.index[entry_idx].to_pydatetime(),
        entry_fill=float(bars.iloc[entry_idx]["close"]),
        signal_date=bars.index[entry_idx].to_pydatetime(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=10_000,
        peak=float(bars.iloc[entry_idx]["close"]),
        exit_signal=None,
    )


def _open(portfolio: Portfolio, state: _SlotState) -> None:
    portfolio.assign(state.ticker, state.rank, state.signal_date)
    portfolio.open(
        ticker=state.ticker,
        entry_date=state.entry_date,
        entry_price=state.entry_fill,
    )


# --------------------------------------------------------------------------- #
# run_day_loop: session grouping keyed on is_session_last
# --------------------------------------------------------------------------- #
class _RecordingDayLoop:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def process_exits_for_day(self, day: pd.Timestamp):
        self.calls.append(("exits", day))
        return []

    def flatten_at_session_end(self, day: pd.Timestamp):
        self.calls.append(("flatten", day))
        return []


class _RecordingSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def before_exits(self, day: pd.Timestamp) -> None:
        self.calls.append(("before", day))

    def after_exits(self, day: pd.Timestamp, freed) -> None:
        self.calls.append(("after", day))


def test_run_day_loop_flattens_only_on_session_last_bars():
    # Session 1 full (3 bars), session 2 a half-day (2 bars).
    days = list(_us_5m_index("2024-03-04", 3)) + list(_us_5m_index("2024-03-05", 2))
    loop = _RecordingDayLoop()
    source = _RecordingSource()

    run_day_loop(days, loop, source, market_tz="America/New_York")

    flatten_days = [day for kind, day in loop.calls if kind == "flatten"]
    assert flatten_days == [days[2], days[4]]  # last bar of each session

    # Per-bar ordering is the flat loop's: before → exits → after, with the
    # flatten spliced in on session-last bars before the refill.
    kinds = [kind for kind, _ in loop.calls]
    assert kinds == [
        "exits",
        "exits",
        "exits",
        "flatten",  # session 1 last bar
        "exits",
        "exits",
        "flatten",  # session 2 (half-day) last bar
    ]
    assert [day for kind, day in source.calls if kind == "before"] == days
    assert [day for kind, day in source.calls if kind == "after"] == days


def test_run_day_loop_without_tz_never_flattens():
    days = list(_us_5m_index("2024-03-04", 3)) + list(_us_5m_index("2024-03-05", 2))
    loop = _RecordingDayLoop()
    source = _RecordingSource()

    run_day_loop(days, loop, source)  # legacy daily path

    assert [kind for kind, _ in loop.calls] == ["exits"] * 5
    assert [day for kind, day in source.calls if kind == "after"] == days


def test_run_day_loop_empty_calendar():
    loop = _RecordingDayLoop()
    source = _RecordingSource()
    run_day_loop([], loop, source, market_tz="America/New_York")
    assert loop.calls == []
    assert source.calls == []


# --------------------------------------------------------------------------- #
# DayLoop.flatten_at_session_end
# --------------------------------------------------------------------------- #
def _loop(cfg: BacktestConfig, portfolio: Portfolio, states, bars_by_slot) -> DayLoop:
    return DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=states,
        slot_bars=bars_by_slot,
    )


def test_flatten_closes_open_slot_at_own_last_session_bar():
    cfg = _cfg()
    index = _us_5m_index("2024-03-04", 6)
    bars = _frame(index)
    state = _state(bars, entry_idx=1)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state)
    loop = _loop(cfg, portfolio, {0: state}, {0: bars})

    freed = loop.flatten_at_session_end(index[5])

    assert [f.slot_id for f in freed] == [0]
    assert loop.slot_states[0] is None
    (trade,) = portfolio.closed_trades()
    assert trade.exit_reason == "session"
    # MOC flat: filled at the session's last bar close (no slippage configured).
    assert trade.exit_date == index[5].to_pydatetime()
    assert trade.exit_price == float(bars.iloc[5]["close"])


def test_flatten_is_noop_without_intraday_only():
    cfg = _cfg(intraday_only=False)
    index = _us_5m_index("2024-03-04", 6)
    bars = _frame(index)
    state = _state(bars, entry_idx=1)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state)
    loop = _loop(cfg, portfolio, {0: state}, {0: bars})

    assert loop.flatten_at_session_end(index[5]) == []
    assert loop.slot_states[0] is state


def test_flatten_skips_slots_without_a_bar_in_the_session():
    cfg = _cfg()
    bars = _frame(_us_5m_index("2024-03-04", 3))
    state = _state(bars, entry_idx=0)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state)
    loop = _loop(cfg, portfolio, {0: state}, {0: bars})

    # Session-last bar two days later: the slot's frame has no bar there.
    later = _us_5m_index("2024-03-06", 6)[-1]
    assert loop.flatten_at_session_end(later) == []
    assert loop.slot_states[0] is state


def test_flatten_skips_slots_past_their_entry_bar_only():
    cfg = _cfg()
    index = _us_5m_index("2024-03-04", 6)
    bars = _frame(index)
    # Entered on the session's last bar: no post-entry bar to flatten against.
    state = _state(bars, entry_idx=5)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state)
    loop = _loop(cfg, portfolio, {0: state}, {0: bars})

    assert loop.flatten_at_session_end(index[5]) == []
    assert loop.slot_states[0] is state


def test_flatten_cleans_slot_whose_position_already_closed():
    cfg = _cfg()
    index = _us_5m_index("2024-03-04", 6)
    bars = _frame(index)
    state = _state(bars, entry_idx=1)
    portfolio = Portfolio(100_000.0, 1)  # never opened: position is None
    loop = _loop(cfg, portfolio, {0: state}, {0: bars})

    freed = loop.flatten_at_session_end(index[5])

    assert [f.slot_id for f in freed] == [0]
    assert loop.slot_states[0] is None
    assert portfolio.closed_trades() == []


# --------------------------------------------------------------------------- #
# Engine integration: rolling run flattens at session close
# --------------------------------------------------------------------------- #
def _rising_frame(index: pd.DatetimeIndex, start_px: float) -> pd.DataFrame:
    n = len(index)
    close = pd.Series(np.linspace(start_px, start_px + n, n), index=index, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 0.5)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([openp, close], axis=1).min(axis=1) - 0.5
    vol = pd.Series(50_000.0, index=index, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_rolling_intraday_only_flattens_at_session_close():
    index = _us_5m_index("2024-03-04", 6).append(_us_5m_index("2024-03-05", 6))
    data = {
        "AAA": _rising_frame(index, 100.0),
        "SPY": _rising_frame(index, 400.0),
    }
    cfg = _cfg(
        as_of=index[1].to_pydatetime(),
        entry_expr="close > sma(close, 2)",
        tickers=("AAA",),
        hold=100,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=date(2024, 3, 4),
        end_date=date(2024, 3, 5),
    )

    assert result.trades, "expected at least one intraday trade"
    tz = "America/New_York"
    for trade in result.trades:
        assert trade.exit_reason == "session"
        entry_local = pd.Timestamp(trade.entry_date).tz_localize("UTC").tz_convert(tz)
        exit_local = pd.Timestamp(trade.exit_date).tz_localize("UTC").tz_convert(tz)
        # Never held overnight: flattened on the entry session's last bar.
        assert exit_local.date() == entry_local.date()
