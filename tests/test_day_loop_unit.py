"""Direct unit tests for the shared day-loop machinery.

``tests/test_day_loop.py`` pins day-loop behaviour end-to-end through full
``run_backtest`` / ``run_rolling_backtest`` runs. These tests instead exercise
:class:`screener.backtester.day_loop.DayLoop` and the ``core.py`` exit helpers
it drives *in isolation*, on hand-built fixtures, so each invariant they encode
is asserted directly rather than inferred from an aggregate trade ledger.

Invariants covered:

* ``DayLoop.process_exits_for_day`` per-slot short-circuits: no bar for the day,
  pre-entry bar, and ``None`` slot are all skipped without a close.
* The invariant per-slot sequence dividends → partial exits → (full-close check)
  → exit check, including the partial that raises the stop the exit check then
  sees, and the partial that fully closes a slot without a separate exit trade.
* Freed slots are returned in slot-id iteration order.
* ``core._check_exit_at_bar`` exit-priority ordering (stop+target → stop, trail
  before target, exit_expr before time) and its peak ratchet.
* ``core._fire_partial_exits_at_bar`` tier firing and the post-fire stop raise.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.core import (
    _check_exit_at_bar,
    _fire_partial_exits_at_bar,
    _SlotState,
)
from screener.backtester.day_loop import DayLoop, FreedSlot
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 1, 1),
        hold=5,
        top=1,
        entry_expr="entry_signal > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _frame(rows: list[dict], start: str = "2024-01-01") -> pd.DataFrame:
    """Build an OHLCV frame with a business-day index from explicit rows."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx)


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 10_000.0,
    **extra: float,
) -> dict:
    row = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    row.update(extra)
    return row


def _state(
    bars: pd.DataFrame,
    *,
    ticker: str = "AAA",
    entry_idx: int = 0,
    entry_fill: float = 100.0,
    stop_ref: float | None = None,
    target_ref: float | None = None,
    hold_limit_idx: int = 10_000,
    peak: float | None = None,
    exit_signal: pd.Series | None = None,
    partial_targets: tuple[float, ...] = (),
    partial_fractions: tuple[float, ...] = (),
) -> _SlotState:
    return _SlotState(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=bars.index[entry_idx].date(),
        entry_fill=entry_fill,
        signal_date=bars.index[entry_idx].date(),
        rank=1,
        stop_ref=stop_ref,
        target_ref=target_ref,
        hold_limit_idx=hold_limit_idx,
        peak=entry_fill if peak is None else peak,
        exit_signal=exit_signal,
        partial_targets=partial_targets,
        partial_fractions=partial_fractions,
        partial_fired=[False] * len(partial_targets),
    )


def _open(portfolio: Portfolio, state: _SlotState, cfg: BacktestConfig) -> None:
    portfolio.assign(state.ticker, state.rank, state.signal_date)
    portfolio.open(
        ticker=state.ticker,
        entry_date=state.entry_date,
        entry_price=state.entry_fill,
    )


# ── DayLoop.process_exits_for_day short-circuits ─────────────────────


def test_slot_with_no_bar_for_day_is_skipped():
    cfg = _cfg(stop_loss=0.05)
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 101, 99, 100)])
    state = _state(bars, stop_ref=95.0)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    # A calendar day the slot's frame does not contain: no close, not freed.
    freed = loop.process_exits_for_day(pd.Timestamp("2024-06-01"))

    assert freed == []
    assert portfolio.closed_trades() == []
    assert loop.slot_states[0] is state


def test_pre_entry_bar_is_skipped():
    cfg = _cfg(stop_loss=0.05)
    bars = _frame([_bar(100, 101, 99, 100), _bar(90, 91, 80, 85)])
    # entry_idx points at bar 1; bar 0 precedes entry_idx + 1, so even a bar that
    # would breach the stop must not trade.
    state = _state(bars, entry_idx=1, stop_ref=95.0)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[0])

    assert freed == []
    assert portfolio.closed_trades() == []


def test_none_slot_is_skipped():
    cfg = _cfg()
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 101, 99, 100)])
    loop = DayLoop(
        portfolio=Portfolio(100_000.0, 1),
        cfg=cfg,
        slot_states={0: None},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    assert loop.process_exits_for_day(bars.index[1]) == []


# ── DayLoop.process_exits_for_day exit-sequence ──────────────────────


def test_stop_exit_frees_slot_and_records_close():
    cfg = _cfg(stop_loss=0.05, gap_fills=False)
    bars = _frame(
        [_bar(100, 101, 99, 100), _bar(100, 101, 94, 96)]
    )  # bar 1 low pierces the 95 stop
    state = _state(bars, stop_ref=95.0)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[1])

    assert [f.slot_id for f in freed] == [0]
    assert isinstance(freed[0], FreedSlot)
    assert loop.slot_states[0] is None
    trades = portfolio.closed_trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"


def test_partial_fire_precedes_and_raises_stop_for_exit_check():
    """A tier scales out first; the exit check then sees the raised stop."""
    cfg = _cfg(stop_loss=0.05, gap_fills=False)
    # Bar 1 both clears the +5% partial target (high 106 >= 105) and pierces the
    # original 95 stop (low 94). The partial must fire first, raise the stop to
    # entry (100), and the exit check then closes the remainder at that stop.
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 106, 94, 100)])
    state = _state(
        bars,
        stop_ref=95.0,
        partial_targets=(105.0,),
        partial_fractions=(0.5,),
    )
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[1])

    assert [f.slot_id for f in freed] == [0]
    trades = portfolio.closed_trades()
    assert [t.exit_reason for t in trades] == ["target", "stop"]
    assert state.partial_fired == [True]
    # Stop was ratcheted up to the entry fill by the partial fire.
    assert state.stop_ref == pytest.approx(100.0)


def test_partial_full_close_frees_slot_without_exit_trade():
    """A 100% tier closes the whole slot; no separate exit-check trade is cut."""
    cfg = _cfg(gap_fills=False)
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 106, 99, 104)])
    state = _state(
        bars,
        partial_targets=(105.0,),
        partial_fractions=(1.0,),
    )
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[1])

    assert [f.slot_id for f in freed] == [0]
    trades = portfolio.closed_trades()
    assert len(trades) == 1  # only the full-size partial, no second exit trade
    assert trades[0].exit_reason == "target"


def test_dividend_credited_before_close():
    cfg = _cfg(hold=1, price_adjustment="splits_only")
    # Exit bar is an ex-date carrying a $1.25 dividend; the day-loop credits it
    # (via _maybe_credit_dividends) before the time exit closes the lot, so the
    # emitted trade carries the dividend income.
    bars = _frame(
        [
            _bar(100, 101, 99, 100, dividend=0.0),
            _bar(100, 101, 99, 100, dividend=1.25),
        ]
    )
    state = _state(bars, hold_limit_idx=1)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states={0: state},
        slot_bars={0: bars},
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[1])

    assert [f.slot_id for f in freed] == [0]
    trade = portfolio.closed_trades()[0]
    assert trade.exit_reason == "time"
    assert trade.dividend_income > 0


def test_freed_slots_returned_in_slot_id_order():
    cfg = _cfg(stop_loss=0.05, gap_fills=False)
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 101, 94, 96)])
    portfolio = Portfolio(100_000.0, 3)
    states: dict[int, _SlotState | None] = {}
    slot_bars: dict[int, pd.DataFrame] = {}
    # Slots 0 and 2 exit on the day; slot 1 does not (its low holds above stop).
    for slot_id, low, ticker in [(0, 94, "AAA"), (1, 99, "BBB"), (2, 94, "CCC")]:
        frame = _frame([_bar(100, 101, 99, 100), _bar(100, 101, low, 100)])
        st = _state(frame, ticker=ticker, stop_ref=95.0)
        _open(portfolio, st, cfg)
        states[slot_id] = st
        slot_bars[slot_id] = frame
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=states,
        slot_bars=slot_bars,
        fill_model=FillModel(cfg),
    )

    freed = loop.process_exits_for_day(bars.index[1])

    assert [f.slot_id for f in freed] == [0, 2]
    assert loop.slot_states[1] is not None


# ── core._check_exit_at_bar priority ordering ────────────────────────


def _one_bar(**kw) -> pd.DataFrame:
    return _frame([_bar(**kw)])


def test_check_exit_stop_and_target_same_bar_resolves_to_stop():
    cfg = _cfg(stop_loss=0.05, take_profit=0.05, gap_fills=False)
    bars = _one_bar(open_=100, high=106, low=94, close=100)
    state = _state(bars, stop_ref=95.0, target_ref=105.0)
    result = _check_exit_at_bar(state, bars, 0, cfg, FillModel(cfg))
    assert result is not None
    assert result[1] == "stop"


def test_check_exit_trail_beats_target():
    cfg = _cfg(trailing_stop=0.10, take_profit=0.05, gap_fills=False)
    # peak 100 → trail_ref 90; low 89 trips the trail while high 106 also clears
    # the 105 target. Trail is evaluated first.
    bars = _one_bar(open_=100, high=106, low=89, close=95)
    state = _state(bars, target_ref=105.0, peak=100.0)
    result = _check_exit_at_bar(state, bars, 0, cfg, FillModel(cfg))
    assert result is not None
    assert result[1] == "trail"


def test_check_exit_expr_beats_time():
    cfg = _cfg()
    bars = _one_bar(open_=100, high=101, low=99, close=100)
    exit_signal = pd.Series([True], index=bars.index)
    # Both the exit expression and the hold limit fire on this bar; exit_expr wins.
    state = _state(bars, hold_limit_idx=0, exit_signal=exit_signal)
    result = _check_exit_at_bar(state, bars, 0, cfg, FillModel(cfg))
    assert result is not None
    assert result[1] == "exit_expr"


def test_check_exit_time_when_hold_reached():
    cfg = _cfg()
    bars = _one_bar(open_=100, high=101, low=99, close=100)
    state = _state(bars, hold_limit_idx=0)
    result = _check_exit_at_bar(state, bars, 0, cfg, FillModel(cfg))
    assert result is not None
    assert result[1] == "time"


def test_check_exit_ratchets_peak_when_no_exit():
    cfg = _cfg()
    bars = _one_bar(open_=100, high=120, low=99, close=118)
    state = _state(bars, peak=100.0, hold_limit_idx=10_000)
    result = _check_exit_at_bar(state, bars, 0, cfg, FillModel(cfg))
    assert result is None
    assert state.peak == pytest.approx(120.0)


# ── core._fire_partial_exits_at_bar ──────────────────────────────────


def test_fire_partial_exit_scales_out_and_raises_stop():
    cfg = _cfg(gap_fills=False)
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 106, 99, 104)])
    state = _state(
        bars,
        stop_ref=None,
        partial_targets=(105.0,),
        partial_fractions=(0.5,),
    )
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)

    _fire_partial_exits_at_bar(state, bars, 1, cfg, portfolio, FillModel(cfg))

    assert state.partial_fired == [True]
    # Stop is lifted to break-even (entry fill) after the first scale-out.
    assert state.stop_ref == pytest.approx(100.0)
    trades = portfolio.closed_trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "target"


def test_fire_partial_exit_no_tiers_is_noop():
    cfg = _cfg()
    bars = _frame([_bar(100, 101, 99, 100), _bar(100, 106, 99, 104)])
    state = _state(bars)
    portfolio = Portfolio(100_000.0, 1)
    _open(portfolio, state, cfg)

    _fire_partial_exits_at_bar(state, bars, 1, cfg, portfolio, FillModel(cfg))

    assert portfolio.closed_trades() == []
