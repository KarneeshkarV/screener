"""Rank-based exit rebalance (rolling engine).

Covers the ``--rank-exit`` feature: a global trading-bar schedule, strict
top-N membership, force-close with reason ``"rank"``, same-day refill of the
freed slot, prepared-panel reuse across rank knobs, and CLI flag parsing.
"""

from __future__ import annotations

from datetime import datetime

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.cli_common import parse_rank_exit
from screener.backtester.core import _bar_label, _check_exit_at_bar, _SlotState
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import (
    _DailyRankingSource,
    _mark_rank_exits,
    run_rolling_backtest,
)
from screener.cli import cli
from tests.conftest import StubPriceFetcher

_INDEX = pd.bdate_range("2024-01-01", periods=20)


def _ramp(start_px: float, end_px: float, volume: float) -> pd.DataFrame:
    close = pd.Series(np.linspace(start_px, end_px, len(_INDEX)), index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _rise_then_fall(peak: float, volume: float) -> pd.DataFrame:
    """Rises one point per bar for five bars, then falls five points per bar.

    The decline puts every close below its own sma(close, 3), so the entry
    signal - and therefore candidate eligibility - dies on the first falling
    bar while the position is still held.
    """
    vals = [100.0, 101.0, 102.0, 103.0, 104.0]
    vals += [peak - 5 * k for k in range(1, len(_INDEX) - len(vals) + 1)]
    close = pd.Series(vals, index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# Distinct dollar volumes fix the ranking order: AAA > BBB > CCC.
_DATA = {
    "AAA": _rise_then_fall(104.0, 500_000.0),
    "BBB": _ramp(100.0, 130.0, 300_000.0),
    "CCC": _ramp(100.0, 130.0, 200_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}

# Both holdings die together while CCC stays eligible.
_THIN_DATA = {
    "AAA": _rise_then_fall(104.0, 500_000.0),
    "BBB": _rise_then_fall(104.0, 300_000.0),
    "CCC": _ramp(100.0, 130.0, 200_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}

# AAA stays signal-eligible but its dollar volume is overtaken by BBB's, so it
# falls below a top-1 cutoff without ever losing its entry signal.
_LAGGARD_DATA = {
    "AAA": _ramp(100.0, 102.0, 500_000.0),
    "BBB": _ramp(100.0, 200.0, 300_000.0),
    "CCC": _ramp(100.0, 130.0, 100_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}

# Two-name universe: AAA leads early, CCC's rising dollar volume overtakes it
# around bar 6. AAA is additionally missing its bar at index 8 (halt).
_GAP_AAA = _ramp(100.0, 102.0, 500_000.0).drop(index=_INDEX[8])
_GAP_DATA = {
    "AAA": _GAP_AAA,
    "CCC": _ramp(80.0, 180.0, 450_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}


def _cfg(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {
        "market": "us",
        "as_of": _INDEX[-1].date(),
        "hold": 40,  # longer than the window: only rank/eod exits can fire
        "top": 1,
        "strategy_name": None,
        "entry_expr": "close > sma(close, 3)",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA", "BBB", "CCC"),
        "rank_exit_every": 3,
        "rank_universe_size": 2,
    }
    values.update(overrides)
    return BacktestConfig(**values)  # type: ignore[arg-type]


def _run(
    cfg: BacktestConfig, data: dict[str, pd.DataFrame] | None = None
) -> list[object]:
    return run_rolling_backtest(
        cfg,
        StubPriceFetcher(data if data is not None else _DATA),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    ).trades


def test_rank_exit_closes_holding_that_left_the_top_list():
    # AAA enters first (highest ADV), its signal dies on the first falling bar.
    # The bar-5 sweep judges on bar 4 (AAA still listed); the bar-8 sweep
    # judges on bar 7 (AAA ineligible) and flags it, so it closes at bar 8.
    trades = _run(_cfg())

    aaa = trades[0]
    assert str(aaa.ticker) == "AAA"  # type: ignore[attr-defined]
    assert str(aaa.exit_reason) == "rank"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa.exit_date) == _INDEX[8]  # type: ignore[attr-defined]

    bbb = trades[1]
    assert str(bbb.ticker) == "BBB"  # type: ignore[attr-defined]
    assert str(bbb.exit_reason) == "eod"  # type: ignore[attr-defined]


def test_rank_exit_is_off_by_default():
    trades = _run(_cfg(rank_exit_every=None))
    assert trades
    assert {str(t.exit_reason) for t in trades}.isdisjoint({"rank"})  # type: ignore[attr-defined]


def test_eligible_holding_below_the_rank_cutoff_is_closed():
    """The core semantic: signal-eligible but ranked out means sold.

    AAA keeps its entry signal the whole window, but BBB's dollar volume
    crosses above it between bars 13 and 14. The bar-14 sweep judges on
    bar 13 (AAA still ahead); the bar-17 sweep judges on bar 16 (BBB well
    ahead) and flags AAA, which closes at bar 17's close.
    """
    trades = _run(_cfg(rank_universe_size=1), data=_LAGGARD_DATA)

    aaa = trades[0]
    assert str(aaa.ticker) == "AAA"  # type: ignore[attr-defined]
    assert str(aaa.exit_reason) == "rank"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa.exit_date) == _INDEX[17]  # type: ignore[attr-defined]

    bbb = trades[1]
    assert str(bbb.ticker) == "BBB"  # type: ignore[attr-defined]
    assert str(bbb.exit_reason) == "eod"  # type: ignore[attr-defined]


def test_short_top_list_closes_holdings_not_listed():
    """Strict rule: a list shorter than N still closes everything absent."""
    trades = _run(_cfg(top=2, rank_universe_size=2), data=_THIN_DATA)

    rank_exits = [t for t in trades if str(t.exit_reason) == "rank"]  # type: ignore[attr-defined]
    assert {str(t.ticker) for t in rank_exits} == {"AAA", "BBB"}  # type: ignore[attr-defined]


def test_rank_universe_size_must_not_be_below_slot_count():
    """A cutoff below the slot count would churn every refill candidate."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="rank_universe_size"):
        _cfg(top=2, rank_universe_size=1)


def test_flagged_holding_without_a_bar_closes_on_its_next_bar():
    """A halted ticker is flagged at the sweep but closes when a bar exists.

    CCC's dollar volume overtakes AAA's around bar 6. The bar-8 sweep judges
    bar 7 (CCC ahead) and flags AAA, but AAA has no bar that day (halted), so
    the shared sweep skips it; its next bar is index 9, where it closes.
    """
    trades = _run(_cfg(rank_universe_size=1), data=_GAP_DATA)

    aaa = trades[0]
    assert str(aaa.ticker) == "AAA"  # type: ignore[attr-defined]
    assert str(aaa.exit_reason) == "rank"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa.exit_date) == _INDEX[9]  # type: ignore[attr-defined]


def test_schedule_fires_on_every_nth_trading_bar(monkeypatch: pytest.MonkeyPatch):
    """The counter is global: first sweep on bar N, judging the bar N-1."""
    judged_days: list[pd.Timestamp] = []
    monkeypatch.setattr(
        "screener.backtester.rolling_simulation._mark_rank_exits",
        lambda **kwargs: judged_days.append(kwargs["prev_day"]),
    )
    source = _DailyRankingSource(
        candidate_matrices=None,  # type: ignore[arg-type]
        bars_by_tv={},
        cfg=_cfg(rank_exit_every=5),
        exit_ast=None,
        fill_model=None,  # type: ignore[arg-type]
        portfolio=None,  # type: ignore[arg-type]
        slot_states={},
        slot_bars={},
        end_ts=_INDEX[-1],
        selection_rows=[],
        warnings=[],
        exit_signals={},
        frame_caches={},
    )
    for day in _INDEX[:12]:
        source.before_exits(day)

    assert judged_days == [_INDEX[3], _INDEX[8]]


def test_rank_knobs_reuse_prepared_panels():
    from screener.backtester.rolling_simulation import prepare_rolling_backtest

    cfg = _cfg()
    window = {
        "start_date": _INDEX[0].date(),
        "end_date": _INDEX[-1].date(),
    }
    prepared = prepare_rolling_backtest(cfg, StubPriceFetcher(_DATA), **window)
    swept = cfg.model_copy(update={"rank_universe_size": 7})
    changed_period = cfg.model_copy(update={"rank_exit_every": 9})

    assert prepared.supports(swept)
    assert prepared.supports(changed_period)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("weekly", (5, True)),
        ("monthly", (21, True)),
        ("WEEKLY", (5, True)),
        ("10", (10, False)),
        (None, None),
    ],
)
def test_rank_exit_flag_parsing(raw: str | None, expected: tuple[int, bool] | None):
    assert parse_rank_exit(raw) == expected


def test_rank_exit_flag_rejects_garbage():
    with pytest.raises(click.BadParameter):
        parse_rank_exit("fortnightly")


def test_named_rank_exit_presets_require_daily_bars():
    result = CliRunner().invoke(
        cli,
        ["backtest-rolling", "--rank-exit", "weekly", "--interval", "15m"],
    )
    assert result.exit_code == 2
    assert "require --interval 1d" in result.output


def _slot_state(**overrides: object) -> _SlotState:
    values: dict[str, object] = {
        "ticker": "AAA",
        "entry_idx": 0,
        "entry_date": _INDEX[0].date(),
        "entry_fill": 100.0,
        "signal_date": _INDEX[0].date(),
        "rank": 1,
        "stop_ref": None,
        "target_ref": None,
        "hold_limit_idx": 10**6,
        "peak": 100.0,
        "exit_signal": None,
        "rank_exit": True,
    }
    values.update(overrides)
    return _SlotState(**values)  # type: ignore[arg-type]


_BARS = pd.DataFrame(
    {"open": [99.0], "high": [101.0], "low": [99.5], "close": [100.5]},
    index=[_INDEX[0]],
)


def _check(state: _SlotState, cfg: BacktestConfig) -> str | None:
    result = _check_exit_at_bar(state, _BARS, 0, cfg, FillModel(cfg))
    return None if result is None else str(result[1])


def test_rank_priority_below_protective_exits():
    stopped = _slot_state(stop_ref=99.7)  # low 99.5 breaches the stop
    assert _check(stopped, _cfg(stop_loss=0.03)) == "stop"

    trailed = _slot_state(peak=105.0)  # trail ref 99.75 > low 99.5
    assert _check(trailed, _cfg(trailing_stop=0.05)) == "trail"

    targeted = _slot_state(target_ref=100.8)  # high 101.0 reaches it
    assert _check(targeted, _cfg()) == "target"


def test_rank_priority_above_expr_session_time():
    fired = pd.Series([True], index=_INDEX[:1])
    expr_fired = _slot_state(exit_signal=fired)
    assert _check(expr_fired, _cfg()) == "rank"

    timed = _slot_state(hold_limit_idx=0)
    assert _check(timed, _cfg()) == "rank"

    unflagged_timed = _slot_state(hold_limit_idx=0, rank_exit=False)
    assert _check(unflagged_timed, _cfg()) == "time"


def test_delayed_limit_fill_on_rebalance_day_is_not_liquidated_on_entry_bar():
    """A limit order filled on a rebalance bar must not rank-exit the same day.

    AAA leads on bar 3 (first eligible signal bar) and queues a 5% below-close
    limit order. On bar 4 (``prev_day`` for the bar-5 sweep), AAA stays above
    its limit so the order remains pending, while BBB's higher dollar volume
    drops AAA out of top-1. On bar 5 (rebalance day, ``_day_count == 6``), AAA
    dips to fill the limit order. Even though ``signal_date < prev_day`` and
    AAA was outside top-1 on bar 4, it entered on bar 5 and must not be
    liquidated on its entry bar; it exits on the next rebalance sweep (bar 8)
    once held across ``prev_day``.
    """
    aaa = _ramp(100.0, 119.0, 500_000.0).copy()
    # Keep bar 4 above the bar-3 limit (~97.85), dip on bar 5 to trigger fill.
    aaa.loc[_INDEX[4], "low"] = 102.0
    aaa.loc[_INDEX[5], "low"] = 90.0
    # BBB trails AAA through bar 3, then overtakes AAA in dollar volume from
    # bar 4 onward so AAA is outside top-1 on both bar 4 and bar 7.
    bbb = _ramp(100.0, 119.0, 200_000.0).copy()
    bbb.loc[_INDEX[4:], "volume"] = 1_000_000.0

    data = {
        "AAA": aaa,
        "BBB": bbb,
        "SPY": _ramp(400.0, 440.0, 1_000_000.0),
    }
    trades = _run(
        _cfg(
            tickers=("AAA", "BBB"),
            top=1,
            rank_exit_every=3,
            rank_universe_size=1,
            entry_order_type="limit",
            entry_limit_bps=500.0,
        ),
        data=data,
    )

    aaa_trade = trades[0]
    assert str(aaa_trade.ticker) == "AAA"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.signal_date) == _INDEX[3]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.entry_date) == _INDEX[5]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.exit_date) == _INDEX[8]  # type: ignore[attr-defined]
    assert str(aaa_trade.exit_reason) == "rank"  # type: ignore[attr-defined]


def test_halted_moo_fill_on_rebalance_day_is_not_liquidated_on_entry_bar():
    """An MOO order delayed by a halt on ``prev_day`` must not exit on entry.

    AAA leads on bar 3 (first eligible signal bar) and queues an MOO order,
    then has no bar on bar 4 (halted on ``prev_day`` for the bar-5 sweep, so
    absent from the bar-4 top-1 list). AAA resumes on bar 5 (rebalance day,
    ``_day_count == 6``) and fills at bar 5's open. It must not be flagged for
    rank exit on bar 5; because BBB holds top-1 through bar 7, AAA rank-exits
    on the following rebalance sweep (bar 8).
    """
    aaa = _ramp(100.0, 119.0, 500_000.0).drop(index=_INDEX[4])
    bbb = _ramp(100.0, 119.0, 200_000.0).copy()
    bbb.loc[_INDEX[4:], "volume"] = 1_000_000.0

    data = {
        "AAA": aaa,
        "BBB": bbb,
        "SPY": _ramp(400.0, 440.0, 1_000_000.0),
    }
    trades = _run(
        _cfg(
            tickers=("AAA", "BBB"),
            top=1,
            rank_exit_every=3,
            rank_universe_size=1,
            entry_order_type="moo",
        ),
        data=data,
    )

    aaa_trade = trades[0]
    assert str(aaa_trade.ticker) == "AAA"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.signal_date) == _INDEX[3]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.entry_date) == _INDEX[5]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.exit_date) == _INDEX[8]  # type: ignore[attr-defined]
    assert str(aaa_trade.exit_reason) == "rank"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("signal_earlier", "entry_earlier", "prev_day", "prev_label", "current_day"),
    [
        pytest.param(
            _INDEX[1].date(),
            _INDEX[2].date(),
            _INDEX[3],
            _INDEX[3].date(),
            _INDEX[4].date(),
            id="daily-date",
        ),
        pytest.param(
            pd.Timestamp("2024-01-02 14:30:00").to_pydatetime(),
            pd.Timestamp("2024-01-02 15:00:00").to_pydatetime(),
            pd.Timestamp("2024-01-02 15:30:00"),
            pd.Timestamp("2024-01-02 15:30:00").to_pydatetime(),
            pd.Timestamp("2024-01-02 16:00:00").to_pydatetime(),
            id="intraday-datetime",
        ),
    ],
)
def test_mark_rank_exits_spares_same_day_entry_and_flags_prior_holdings(
    monkeypatch: pytest.MonkeyPatch,
    signal_earlier: object,
    entry_earlier: object,
    prev_day: pd.Timestamp,
    prev_label: object,
    current_day: object,
):
    """Positions entered on ``current_day`` are spared; held slots still exit."""
    monkeypatch.setattr(
        "screener.backtester.rolling_simulation._candidate_rows_for_day",
        lambda *args, **kwargs: ([], []),
    )
    held_prior = _slot_state(
        ticker="OLD",
        signal_date=signal_earlier,
        entry_date=entry_earlier,
        rank_exit=False,
    )
    held_on_prev = _slot_state(
        ticker="PREV",
        signal_date=signal_earlier,
        entry_date=prev_label,
        rank_exit=False,
    )
    entered_today = _slot_state(
        ticker="TODAY",
        signal_date=signal_earlier,
        entry_date=current_day,
        rank_exit=False,
    )
    fresh_signal = _slot_state(
        ticker="FRESH",
        signal_date=prev_label,
        entry_date=current_day,
        rank_exit=False,
    )

    _mark_rank_exits(
        prev_day=prev_day,
        cfg=_cfg(rank_exit_every=1, rank_universe_size=1),
        candidate_matrices=None,  # type: ignore[arg-type]
        slot_states={
            0: held_prior,
            1: held_on_prev,
            2: entered_today,
            3: fresh_signal,
        },
        warnings=[],
    )

    assert held_prior.rank_exit is True
    assert held_on_prev.rank_exit is True
    assert entered_today.rank_exit is False
    assert fresh_signal.rank_exit is False


@pytest.mark.parametrize(
    ("rank_exit_every", "expected_exit_idx"),
    [
        pytest.param(3, 8, id="every-3-bars"),
        pytest.param(1, 6, id="every-1-bar"),
    ],
)
def test_moc_fill_on_rebalance_day_stays_unflagged(
    monkeypatch: pytest.MonkeyPatch,
    rank_exit_every: int,
    expected_exit_idx: int,
):
    """A same-bar MOC fill on a rebalance day stays unflagged after ``after_exits``.

    AAA signals on bar 3, is halted on bar 4 (``prev_day`` for bar 5), and
    fills its MOC order at bar 5's close in ``after_exits`` while BBB already
    holds top-1. Because ``before_exits`` runs ``_mark_rank_exits`` before
    ``after_exits`` fills MOC orders, the newly opened slot remains unflagged
    at the end of bar 5 and is only evaluated on the next rebalance cycle
    (bar 8 for ``rank_exit_every=3``, bar 6 for ``rank_exit_every=1``).
    """
    aaa = _ramp(100.0, 119.0, 500_000.0).drop(index=_INDEX[4])
    bbb = _ramp(100.0, 119.0, 200_000.0).copy()
    bbb.loc[_INDEX[4:], "volume"] = 1_000_000.0

    entry_bar_flags: list[bool] = []
    orig_after_exits = _DailyRankingSource.after_exits

    def _spy_after_exits(self, day: pd.Timestamp, freed) -> None:
        orig_after_exits(self, day, freed)
        if day == _INDEX[5]:
            state = self.slot_states[0]
            assert state is not None
            assert pd.Timestamp(state.entry_date) == _INDEX[5]
            entry_bar_flags.append(state.rank_exit)

    monkeypatch.setattr(_DailyRankingSource, "after_exits", _spy_after_exits)

    data = {
        "AAA": aaa,
        "BBB": bbb,
        "SPY": _ramp(400.0, 440.0, 1_000_000.0),
    }
    trades = _run(
        _cfg(
            tickers=("AAA", "BBB"),
            top=1,
            rank_exit_every=rank_exit_every,
            rank_universe_size=1,
            entry_order_type="moc",
        ),
        data=data,
    )

    assert entry_bar_flags == [False]
    aaa_trade = trades[0]
    assert str(aaa_trade.ticker) == "AAA"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.signal_date) == _INDEX[3]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.entry_date) == _INDEX[5]  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa_trade.exit_date) == _INDEX[expected_exit_idx]  # type: ignore[attr-defined]
    assert str(aaa_trade.exit_reason) == "rank"  # type: ignore[attr-defined]


@pytest.mark.parametrize("tz", ["America/New_York", "UTC"])
def test_mark_rank_exits_tz_aware_intraday_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tz: str,
):
    """Tz-aware intraday timestamps compare cleanly without ``TypeError``."""
    monkeypatch.setattr(
        "screener.backtester.rolling_simulation._candidate_rows_for_day",
        lambda *args, **kwargs: ([], []),
    )
    intraday_cfg = _cfg(interval="15m", rank_exit_every=1, rank_universe_size=1)
    bar_index = pd.date_range("2024-01-02 09:30", periods=4, freq="15min", tz=tz)
    signal_earlier = _bar_label(bar_index[0], intraday_cfg)
    entry_earlier = _bar_label(bar_index[1], intraday_cfg)
    prev_day = bar_index[2]
    prev_label = _bar_label(bar_index[2], intraday_cfg)
    current_day = _bar_label(bar_index[3], intraday_cfg)

    assert prev_day.tzinfo is not None
    assert isinstance(current_day, datetime) and current_day.tzinfo is not None

    held_prior = _slot_state(
        ticker="OLD",
        signal_date=signal_earlier,
        entry_date=entry_earlier,
        rank_exit=False,
    )
    held_on_prev = _slot_state(
        ticker="PREV",
        signal_date=signal_earlier,
        entry_date=prev_label,
        rank_exit=False,
    )
    entered_today = _slot_state(
        ticker="TODAY",
        signal_date=signal_earlier,
        entry_date=current_day,
        rank_exit=False,
    )
    fresh_signal = _slot_state(
        ticker="FRESH",
        signal_date=prev_label,
        entry_date=current_day,
        rank_exit=False,
    )

    _mark_rank_exits(
        prev_day=prev_day,
        cfg=intraday_cfg,
        candidate_matrices=None,  # type: ignore[arg-type]
        slot_states={
            0: held_prior,
            1: held_on_prev,
            2: entered_today,
            3: fresh_signal,
        },
        warnings=[],
    )

    assert held_prior.rank_exit is True
    assert held_on_prev.rank_exit is True
    assert entered_today.rank_exit is False
    assert fresh_signal.rank_exit is False
