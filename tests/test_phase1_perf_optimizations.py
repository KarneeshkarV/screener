"""Phase 1 hot-loop optimizations: exact-parity guards.

Each optimization here trades a little indirection for less per-bar or per-day
work. These tests pin the behaviour that must not move: the reserve rejection
memo only skips work inside one day, the master calendar keeps the set-of-days
answer, the partial-target sweep keeps its price/tier order and fill moments,
the candidate ranking is computed once across retry batches, sector
neutralization is bit-identical, warm-cache snapping keeps the same rows,
agent mode skips only unsolicited reports, and ``markets`` stays light.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from collections import deque
from datetime import date
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest

import screener.screen_workflow as sw
from screener import agentio
from screener.backtester import historical
from screener.backtester.cli_common import resolve_report_path
from screener.backtester.core import (
    _build_frame_cache,
    _eligible_reserve_signal_idx,
    _fire_partial_exits_at_bar,
    _half_spread_at_signal,
    _passes_entry_filters,
    _RunCaches,
    _SlotState,
)
from screener.backtester.day_loop import DayLoop
from screener.backtester.fills import FillModel
from screener.backtester.historical import _master_calendar, _ReserveRotationSource
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import PineError, parse
from screener.backtester.portfolio import Portfolio
from screener.backtester.price_frames import _snapped, range_slice
from screener.backtester.rolling_candidates import (
    _candidate_rows_for_day,
    _RollingCandidateMatrices,
    _sector_neutralize_scores,
)
from screener.backtester.slippage import EstimatedHalfSpreadSlippage
from screener.backtester.rolling_simulation import _DailyRankingSource
from screener.criteria import FilterCriteriaSelection
from tests.test_screen_workflow import _AS_OF, _df, _patch, _request


def _bars(
    index: pd.DatetimeIndex,
    *,
    close: np.ndarray | None = None,
    volume: float = 10_000.0,
) -> pd.DataFrame:
    values = np.linspace(10.0, 20.0, len(index)) if close is None else close
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "volume": np.full(len(index), volume),
        },
        index=index,
    )


def _cfg(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {
        "market": "us",
        "as_of": date(2024, 3, 1),
        "benchmark": "SPY",
        "entry_expr": "close > 0",
        "exit_expr": None,
        "hold": 5,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "top": 1,
        "strategy_name": None,
        "initial_capital": 100_000.0,
    }
    values.update(overrides)
    return BacktestConfig(**values)  # type: ignore[arg-type]


def _state(
    *,
    partial_targets: tuple[float, ...] = (),
    target_ref: float | None = None,
    entry_fill: float = 100.0,
    entry_idx: int = 0,
) -> _SlotState:
    return _SlotState(
        ticker="AAA",
        entry_idx=entry_idx,
        entry_date=date(2024, 1, 1),
        entry_fill=entry_fill,
        signal_date=date(2024, 1, 1),
        rank=1,
        stop_ref=None,
        target_ref=target_ref,
        hold_limit_idx=entry_idx + 5,
        peak=entry_fill,
        exit_signal=None,
        partial_targets=partial_targets,
        partial_fractions=tuple(0.5 for _ in partial_targets),
        partial_fired=[False] * len(partial_targets),
    )


def _open_portfolio(price: float = 100.0, shares: float = 10.0) -> Portfolio:
    portfolio = Portfolio(100_000.0, 1)
    portfolio.open("AAA", date(2024, 1, 1), price, budget=price * shares)
    return portfolio


# ---------------------------------------------------------------------------
# Reserve rotation: signal before filters, and the per-day rejection memo
# ---------------------------------------------------------------------------


def test_eligible_reserve_checks_the_cached_signal_before_the_filters(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=8)
    bars = _bars(index)
    cfg = _cfg(min_price=1.0)
    caches = _RunCaches()
    entry_ast = parse("close > 0")
    # Pre-seed a false signal so no AST walk happens; the filter must not run.
    caches.entry_signals["AAA"] = pd.Series(False, index=index)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("filters ran for a reserve with no signal")

    monkeypatch.setattr("screener.backtester.core._passes_entry_filters", unexpected)

    assert (
        _eligible_reserve_signal_idx(
            bars, index[5], cfg, entry_ast, 1, ticker="AAA", caches=caches
        )
        is None
    )

    # A real signal reaches the filters, and a failing filter still rejects it.
    caches.entry_signals["AAA"] = pd.Series(True, index=index)
    seen: list[pd.Timestamp] = []

    def reject(bars_arg, as_of, cfg_arg, **_kwargs):
        seen.append(as_of)
        return False, "rejected"

    monkeypatch.setattr("screener.backtester.core._passes_entry_filters", reject)
    assert (
        _eligible_reserve_signal_idx(
            bars, index[5], cfg, entry_ast, 1, ticker="AAA", caches=caches
        )
        is None
    )
    assert seen == [index[5]]


def test_eligible_reserve_pine_error_still_rejects_before_filters(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=8)
    bars = _bars(index)
    caches = _RunCaches()
    caches.entry_signals["AAA"] = PineError("boom")
    called: list[object] = []
    monkeypatch.setattr(
        "screener.backtester.core._passes_entry_filters",
        lambda *args, **kwargs: called.append(args) or (True, None),
    )

    assert (
        _eligible_reserve_signal_idx(
            bars, index[5], _cfg(), parse("close > 0"), 1, ticker="AAA", caches=caches
        )
        is None
    )
    assert called == []


def _reserve_source(
    *,
    tickers: tuple[str, ...],
    index: pd.DatetimeIndex,
    slots: int,
    cfg: BacktestConfig,
) -> _ReserveRotationSource:
    bars_by_tv = {ticker: _bars(index) for ticker in tickers}
    reserve_queue: deque[dict] = deque(
        {"ticker": ticker, "rank": rank + 1} for rank, ticker in enumerate(tickers)
    )
    return _ReserveRotationSource(
        portfolio=Portfolio(cfg.initial_capital, slots),
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=1,
        fill_model=FillModel(cfg),
        bars_by_tv=bars_by_tv,
        slot_states={slot_id: None for slot_id in range(slots)},
        slot_bars={},
        reentries_left={slot_id: 0 for slot_id in range(slots)},
        pending_reentry={},
        pending_entry_orders={},
        reserve_queue=reserve_queue,
        taken=set(),
        warnings=[],
        caches=_RunCaches(),
    )


def test_a_rejected_reserve_is_checked_once_per_day_not_once_per_slot(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=10)
    cfg = _cfg(top=2)
    source = _reserve_source(tickers=("R1", "R2"), index=index, slots=2, cfg=cfg)
    calls: list[str] = []

    def no_signal(bars, *_args, ticker=None, **_kwargs):
        calls.append(str(ticker))
        return None

    monkeypatch.setattr(historical, "_eligible_reserve_signal_idx", no_signal)

    source.after_exits(index[7], [])

    # Two empty slots, two reserves, but each rejection is learned once.
    assert calls == ["R1", "R2"]
    assert [row["ticker"] for row in source.reserve_queue] == ["R1", "R2"]

    # The memo is per call: the next day rescans both reserves.
    source.after_exits(index[8], [])
    assert calls == ["R1", "R2", "R1", "R2"]
    assert [row["ticker"] for row in source.reserve_queue] == ["R1", "R2"]


def test_budget_and_affordability_rejections_are_also_memoized_per_day(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=10)
    cfg = _cfg(top=2)
    source = _reserve_source(tickers=("R1", "R2"), index=index, slots=2, cfg=cfg)
    calls: list[str] = []

    monkeypatch.setattr(
        historical,
        "_eligible_reserve_signal_idx",
        lambda _bars, _day, *_args, ticker=None, **_kwargs: (
            calls.append(str(ticker)) or 3
        ),
    )
    monkeypatch.setattr(historical, "entry_budget_for", lambda *args, **kwargs: 0.0)

    source.after_exits(index[7], [])
    # Once per reserve even though both slots scan: the budget inputs are fixed.
    assert calls == ["R1", "R2"]

    # India whole-share affordability rejections are also memoized per day.
    india_cfg = _cfg(market="india", top=2, initial_capital=1_000.0)
    india_source = _reserve_source(
        tickers=("R1", "R2"), index=index, slots=2, cfg=india_cfg
    )
    for ticker in ("R1", "R2"):
        india_source.bars_by_tv[ticker] = _bars(
            index, close=np.full(len(index), 2_000.0)
        )
    india_calls: list[str] = []
    monkeypatch.setattr(
        historical,
        "_eligible_reserve_signal_idx",
        lambda _bars, _day, *_args, ticker=None, **_kwargs: (
            india_calls.append(str(ticker)) or 7
        ),
    )
    monkeypatch.setattr(historical, "entry_budget_for", lambda *args, **kwargs: 500.0)

    india_source.after_exits(index[7], [])
    assert india_calls == ["R1", "R2"]
    assert india_source.pending_entry_orders == {}


# ---------------------------------------------------------------------------
# Master calendar
# ---------------------------------------------------------------------------


def _calendar_oracle(bars_by_tv, as_of_ts, horizon_end):
    days: set[pd.Timestamp] = set()
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        idx = bars.index
        days.update(idx[(idx > as_of_ts) & (idx <= horizon_end)])
    return sorted(days)


def _tick_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.bdate_range("2024-01-01", periods=12), id="us-default"),
        pytest.param(pd.bdate_range("2024-01-01", periods=12).as_unit("ns"), id="ns"),
        pytest.param(
            pd.bdate_range("2024-01-01", periods=12).as_unit("s"), id="seconds"
        ),
        pytest.param(pd.date_range("2024-01-01", periods=12, freq="5h"), id="intraday"),
        pytest.param(
            pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"]),
            id="unsorted-duplicates",
        ),
    ],
)
def test_master_calendar_matches_the_set_of_days(index):
    bars = {"a": _tick_frame(index), "b": _tick_frame(index[2:])}
    as_of = pd.Timestamp("2023-12-30")
    end = pd.Timestamp("2024-02-01")

    got = _master_calendar(bars, as_of, end)

    assert got == _calendar_oracle(bars, as_of, end)
    # The fast path keeps the input resolution instead of normalising to ns.
    assert {stamp.unit for stamp in got} == {index.unit}


def test_master_calendar_falls_back_for_tz_aware_and_mixed_units():
    aware = pd.date_range("2024-01-04 09:30", periods=6, freq="1h", tz="UTC")
    mixed = {
        "us": _tick_frame(pd.bdate_range("2024-01-01", periods=5)),
        "ns": _tick_frame(pd.bdate_range("2024-01-01", periods=5).as_unit("ns")),
    }
    start = pd.Timestamp("2023-12-01")
    end = pd.Timestamp("2024-02-01")

    assert _master_calendar(
        {"a": _tick_frame(aware)}, start.tz_localize("UTC"), end.tz_localize("UTC")
    ) == _calendar_oracle(
        {"a": _tick_frame(aware)}, start.tz_localize("UTC"), end.tz_localize("UTC")
    )
    assert _master_calendar(mixed, start, end) == _calendar_oracle(mixed, start, end)


def test_master_calendar_is_empty_without_bars():
    assert (
        _master_calendar({}, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))
        == []
    )
    assert (
        _master_calendar(
            {"a": _tick_frame(pd.DatetimeIndex([]))},
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
        )
        == []
    )


# ---------------------------------------------------------------------------
# Partial-target sweep
# ---------------------------------------------------------------------------


def test_partial_sweep_presorts_targets_with_the_full_target_first_on_ties():
    state = _state(partial_targets=(105.0, 105.0), target_ref=105.0)
    assert state.target_levels == ((105.0, -1), (105.0, 0), (105.0, 1))

    index = pd.bdate_range("2024-01-01", periods=2)
    bars = _bars(index, close=np.array([100.0, 100.0]))
    bars.loc[index[1], ["open", "high", "low", "close"]] = [104.0, 106.0, 99.0, 105.0]
    portfolio = _open_portfolio()

    _fire_partial_exits_at_bar(state, bars, 1, _cfg(), portfolio, FillModel(_cfg()))

    # The full target sorts before the partial tiers it ties with, so the
    # position closes whole and no partial sleeve fires.
    assert state.partial_fired == [False, False]
    assert portfolio.get_position("AAA") is None


def test_partial_sweep_at_open_returns_when_no_target_can_fill(monkeypatch):
    state = _state()
    index = pd.bdate_range("2024-01-01", periods=2)
    bars = _bars(index)
    portfolio = _open_portfolio()

    import screener.backtester.core as core

    monkeypatch.setattr(
        core, "_bar_label", lambda *args, **kwargs: pytest.fail("label computed")
    )
    _fire_partial_exits_at_bar(
        state, bars, 1, _cfg(), portfolio, FillModel(_cfg()), at_open=True
    )
    assert portfolio.get_position("AAA") is not None


def test_partial_sweep_intrabar_still_leaves_the_full_target_to_the_exit_check(
    monkeypatch,
):
    state = _state(target_ref=105.0)
    index = pd.bdate_range("2024-01-01", periods=2)
    bars = _bars(index)
    bars.loc[index[1], ["open", "high", "low", "close"]] = [104.0, 106.0, 99.0, 105.0]
    portfolio = _open_portfolio()

    import screener.backtester.core as core

    monkeypatch.setattr(
        core, "_bar_label", lambda *args, **kwargs: pytest.fail("label computed")
    )
    _fire_partial_exits_at_bar(
        state, bars, 1, _cfg(), portfolio, FillModel(_cfg()), at_open=False
    )
    assert portfolio.get_position("AAA") is not None


def test_day_loop_closes_full_target_intrabar_after_all_partials_have_fired():
    """Once every partial has fired, the intrabar full target still closes via remaining."""
    index = pd.bdate_range("2024-01-01", periods=3)
    bars = _bars(index, close=np.array([100.0, 106.0, 109.0]))
    bars.loc[index[1], ["open", "high", "low", "close"]] = [101.0, 106.0, 100.0, 106.0]
    bars.loc[index[2], ["open", "high", "low", "close"]] = [108.0, 111.0, 107.0, 109.0]

    cfg = _cfg(slippage_bps=0.0, commission_bps=0.0)
    portfolio = _open_portfolio(price=100.0, shares=10.0)
    state = _state(
        partial_targets=(105.0,),
        target_ref=110.0,
    )
    slot_states: dict[int, _SlotState | None] = {0: state}
    loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars={0: bars},
    )

    assert loop.process_exits_for_day(index[1]) == []
    assert state.partial_fired == [True]
    assert portfolio.get_position("AAA") is not None

    freed = loop.process_exits_for_day(index[2])
    assert len(freed) == 1
    assert slot_states[0] is None
    assert portfolio.get_position("AAA") is None
    trades = portfolio.closed_trades()
    assert [t.exit_reason for t in trades] == ["target", "target"]
    assert [t.exit_price for t in trades] == [
        pytest.approx(105.0),
        pytest.approx(110.0),
    ]
    assert [t.shares for t in trades] == [pytest.approx(5.0), pytest.approx(5.0)]


def test_partial_sweep_computes_the_bar_label_only_when_a_fill_lands(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=2)
    portfolio = _open_portfolio()

    import screener.backtester.core as core

    calls: list[int] = []
    original = core._bar_label

    def counting(stamp, cfg):
        calls.append(1)
        return original(stamp, cfg)

    monkeypatch.setattr(core, "_bar_label", counting)

    # Below the target: the reach test breaks before any fill.
    no_fill = _state(partial_targets=(105.0,))
    bars = _bars(index)
    bars.loc[index[1], ["open", "high", "low", "close"]] = [104.0, 104.0, 99.0, 103.0]
    _fire_partial_exits_at_bar(no_fill, bars, 1, _cfg(), portfolio, FillModel(_cfg()))
    assert calls == []

    # At the open, the full target fills and the label is computed once.
    fills = _state(target_ref=105.0)
    bars.loc[index[1], ["open", "high", "low", "close"]] = [106.0, 107.0, 99.0, 106.0]
    _fire_partial_exits_at_bar(
        fills, bars, 1, _cfg(), portfolio, FillModel(_cfg()), at_open=True
    )
    assert calls == [1]


def test_partial_sweep_precomputed_levels_match_the_old_sorted_build():
    for partials in ((), (101.0,), (105.0, 103.0), (110.0, 110.0)):
        for target in (None, 104.0, 110.0):
            state = _state(partial_targets=partials, target_ref=target)
            expected = [(price, tier) for tier, price in enumerate(partials)]
            if target is not None:
                expected.append((target, -1))
            assert state.target_levels == tuple(sorted(expected))


# ---------------------------------------------------------------------------
# Pending entries and frame-cache lookups
# ---------------------------------------------------------------------------


def test_fill_pending_entries_returns_before_marking_equity(monkeypatch):
    import screener.backtester.book as book

    cfg = _cfg(sizing_rule="reinvested_equal_slot")
    monkeypatch.setattr(
        book,
        "marked_portfolio_equity",
        lambda *args, **kwargs: pytest.fail("equity marked with nothing to fill"),
    )
    monkeypatch.setattr(
        book,
        "entry_budget_for",
        lambda *args, **kwargs: pytest.fail("budgeted with nothing to fill"),
    )

    filled = book.fill_pending_entries(
        day=pd.Timestamp("2024-01-02"),
        at_close=cfg.entry_order_type == "moc",
        pending_orders={},
        bars_by_ticker={},
        cfg=cfg,
        portfolio=Portfolio(100_000.0, 1),
        slot_states={},
        slot_bars={},
        fill_model=FillModel(cfg),
        caches=_RunCaches(),
        exit_ast=None,
        warnings=[],
    )

    assert filled == []


def test_entry_filters_with_the_frame_cache_match_the_pandas_path():
    index = pd.bdate_range("2024-01-01", periods=6)
    close = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    bars = _bars(index, close=close)
    bars["volume"] = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    cache = _build_frame_cache(bars)

    for cfg in (
        _cfg(),
        _cfg(min_price=5.5),
        _cfg(min_price=50.0),
        _cfg(min_avg_dollar_volume=1_000.0),
        _cfg(min_avg_dollar_volume=1e9),
    ):
        for as_of in index:
            expected = _passes_entry_filters(bars, as_of, cfg)
            assert (
                _passes_entry_filters(bars, as_of, cfg, frame_cache=cache) == expected
            )


def test_half_spread_uses_the_cached_array_after_the_first_lookup():
    index = pd.bdate_range("2024-01-01", periods=8)
    bars = _bars(index)
    cache = _build_frame_cache(bars)
    cfg = _cfg(spread_proxy=True, slippage_model=EstimatedHalfSpreadSlippage())

    first = _half_spread_at_signal(bars, 3, cfg, cache)
    assert cache.half_spread_arr is not None
    assert np.array_equal(
        cache.half_spread_arr, cache.half_spread_s.to_numpy(dtype=float)
    )
    # Every positional read now comes from the ndarray and agrees with pandas.
    for position in range(len(bars)):
        assert _half_spread_at_signal(bars, position, cfg, cache) == float(
            cache.half_spread_s.iloc[position]
        )
    assert _half_spread_at_signal(bars, 3, cfg, cache) == first


def _matrices(
    day: pd.Timestamp, tickers: tuple[str, ...], dollar_vol: np.ndarray
) -> _RollingCandidateMatrices:
    columns = list(tickers)
    signal = np.ones((1, len(tickers)), dtype=bool)
    values = dollar_vol.reshape(1, -1)
    frame = pd.DataFrame(values, index=[day], columns=columns)
    bool_frame = pd.DataFrame(signal, index=[day], columns=columns)
    return _RollingCandidateMatrices(
        signal_mat=bool_frame,
        lookback_ok_mat=bool_frame,
        filter_mat=None,
        dollar_vol_mat=frame,
        close_mat=frame,
        volume_mat=frame,
        bar_idx_mat=pd.DataFrame(
            np.zeros((1, len(tickers)), dtype=int), index=[day], columns=columns
        ),
        rank_score_mat=None,
        tickers=tickers,
        row_by_day={day: 0},
        col_by_ticker={ticker: i for i, ticker in enumerate(tickers)},
        signal_np=signal,
        lookback_ok_np=signal,
        filter_np=None,
        dollar_vol_np=values,
        close_np=values,
        volume_np=values,
        bar_idx_np=np.zeros((1, len(tickers)), dtype=int),
        rank_score_np=None,
    )


def test_fill_candidate_queue_ranks_the_day_once_across_retry_batches(monkeypatch):
    import screener.backtester.rolling_simulation as rolling_simulation

    day = pd.Timestamp("2026-01-05")
    tickers = tuple(f"T{i:02d}" for i in range(9))
    dollar_vol = np.arange(len(tickers), 0, -1, dtype=float) * 100.0
    matrices = _matrices(day, tickers, dollar_vol)

    cfg = _cfg(top=1, hold=3)
    bars_by_tv = {ticker: _bars(pd.DatetimeIndex([day])) for ticker in tickers}
    slots = {0: None}
    source = _DailyRankingSource(
        candidate_matrices=matrices,
        bars_by_tv=bars_by_tv,
        cfg=cfg,
        exit_ast=None,
        fill_model=FillModel(cfg),
        portfolio=Portfolio(cfg.initial_capital, 1),
        slot_states=slots,
        slot_bars={},
        end_ts=day,
        selection_rows=[],
        warnings=[],
        exit_signals={},
        frame_caches={},
        master_dates=(day,),
    )

    ranking_calls: list[pd.Timestamp] = []
    original = rolling_simulation._ranked_candidates_for_day

    def counting(*args, **kwargs):
        ranking_calls.append(pd.Timestamp(args[0]))
        return original(*args, **kwargs)

    offsets: list[int] = []
    original_rows = rolling_simulation._materialize_candidate_rows

    def materialize(*args, **kwargs):
        offsets.append(int(kwargs.get("offset", 0)))
        return original_rows(*args, **kwargs)

    # Every open fails, so the slot walks the whole ranking through batches.
    monkeypatch.setattr(rolling_simulation, "entry_budget_for", lambda *a, **k: 0.0)
    monkeypatch.setattr(rolling_simulation, "_ranked_candidates_for_day", counting)
    monkeypatch.setattr(rolling_simulation, "_materialize_candidate_rows", materialize)

    source.after_exits(day, [])

    assert ranking_calls == [day]
    assert offsets == [0, 8]
    assert source.pending_entry_orders == {}


def test_candidate_batches_from_one_ranking_match_the_per_batch_scan():
    day = pd.Timestamp("2026-01-05")
    tickers = tuple(f"T{i:02d}" for i in range(7))
    dollar_vol = np.array([700.0, 600.0, 600.0, 400.0, 300.0, 200.0, 100.0])
    matrices = _matrices(day, tickers, dollar_vol)

    expected, warnings = _candidate_rows_for_day(
        day, matrices, exclude=set(), limit=None
    )
    batches = [
        _candidate_rows_for_day(day, matrices, exclude=set(), limit=2, offset=offset)[0]
        for offset in (0, 2, 4, 6)
    ]

    assert [row for batch in batches for row in batch] == expected
    assert [row["rank"] for row in expected] == list(range(1, 8))
    assert warnings == []


# ---------------------------------------------------------------------------
# Transposed sector neutralization
# ---------------------------------------------------------------------------


def _sector_neutralize_scores_oracle(rank_score_mat, sector_by_tv):
    """The pre-optimization stack/groupby/unstack implementation."""
    if rank_score_mat.empty:
        return rank_score_mat
    sectors = {col: sector_by_tv.get(col, "UNKNOWN") for col in rank_score_mat.columns}
    long = pd.Series(rank_score_mat.stack(future_stack=True), dtype=float)
    long.index = long.index.set_names(["date", "ticker"])
    sector = pd.Series(
        long.index.get_level_values("ticker").map(sectors), index=long.index
    )
    date = pd.Series(long.index.get_level_values("date"), index=long.index)
    group_keys = [date, sector]
    mu = long.groupby(group_keys, sort=False).transform("mean")
    centered = long - mu
    var = centered.pow(2).groupby(group_keys, sort=False).transform("mean")
    sigma = var.pow(0.5)
    zscore = centered / sigma
    zscore = zscore.mask(~(sigma > 0), 0.0)
    zscore = zscore.mask(long.isna(), np.nan)
    neutralized = zscore.unstack(level="ticker")
    if isinstance(neutralized, pd.Series):
        neutralized = neutralized.to_frame().T
    return neutralized.reindex(
        index=rank_score_mat.index, columns=rank_score_mat.columns
    )


def _assert_bit_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_index_equal(left.index, right.index)
    pd.testing.assert_index_equal(left.columns, right.columns)
    assert np.array_equal(left.to_numpy().view("i8"), right.to_numpy().view("i8")), (
        "sector-neutralized cells differ"
    )


def test_sector_neutralize_is_bit_identical_on_hand_picked_cases():
    index = pd.bdate_range("2024-01-02", periods=3)
    cases = {
        "all-equal-0.1": (
            pd.DataFrame(np.full((3, 7), 0.1), index=index, columns=list("ABCDEFG")),
            {column: "X" for column in "ABCDEFG"},
        ),
        "single-member": (
            pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index),
            {"A": "Tech"},
        ),
        "unknown-and-nan": (
            pd.DataFrame(
                {
                    "A": [1.0, np.nan, 3.0],
                    "B": [2.0, 2.0, np.nan],
                    "C": [np.nan, 5.0, 6.0],
                    "D": [4.0, 5.0, 9.0],
                },
                index=index,
            ),
            {"A": "Tech", "B": "Tech", "D": "Fin"},
        ),
        "ties": (
            pd.DataFrame(
                {"A": [3.0, 3.0], "B": [1.0, 1.0], "C": [2.0, 2.0]},
                index=index[:2],
            ),
            {"A": "Tech", "B": "Tech", "C": "Tech"},
        ),
    }
    for label, (scores, sectors) in cases.items():
        expected = _sector_neutralize_scores_oracle(scores, sectors)
        actual = _sector_neutralize_scores(scores, sectors)
        assert actual.shape == expected.shape, label
        _assert_bit_identical(expected, actual)


@pytest.mark.parametrize("seed", range(25))
def test_sector_neutralize_is_bit_identical_on_random_fields(seed):
    rng = np.random.default_rng(seed)
    days = int(rng.integers(1, 12))
    tickers = int(rng.integers(1, 25))
    index = pd.bdate_range("2024-01-02", periods=days)
    columns = [f"T{i}" for i in range(tickers)]
    values = rng.normal(size=(days, tickers)) * 10 ** int(rng.integers(-5, 5))
    values[rng.random(values.shape) < 0.15] = np.nan
    if seed % 3 == 0:
        values = np.round(values, 2)
    scores = pd.DataFrame(values, index=index, columns=columns)
    sectors = {
        column: str(rng.choice(["Tech", "Fin", "Energy", "UNKNOWN"]))
        for column in columns
    }

    expected = _sector_neutralize_scores_oracle(scores, sectors)
    actual = _sector_neutralize_scores(scores, sectors)

    _assert_bit_identical(expected, actual)


# ---------------------------------------------------------------------------
# Warm-cache snapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_snapped_returns_the_exact_bound_when_the_unit_conversion_is_lossless(unit):
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01 09:15", periods=40, freq="5min")
    ).as_unit(unit)
    midnight = pd.Timestamp("2024-01-02")

    assert _snapped(index, midnight, "left") == midnight
    assert _snapped(index, midnight, "right") == midnight


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_snapped_sub_unit_bounds_keep_the_mask_rows(side, unit):
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01 09:15", periods=40, freq="5min")
    ).as_unit(unit)
    frame = pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-02") - pd.Timedelta(1, "ns")
    expected = frame.loc[(frame.index >= start) & (frame.index <= end)]

    pd.testing.assert_frame_equal(range_slice(frame, start, end), expected)
    if side == "right":
        # A sub-unit part cannot match a row, so the upper snap is at or below
        # the bound (and never rounds up into the next bar).
        assert _snapped(index, end, "right") <= end


def test_sector_neutralize_handles_all_none_sector_values():
    index = pd.bdate_range("2024-01-01", periods=3)
    mat = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0]}, index=index)
    # Normalizing None/empty entries into UNKNOWN keeps behavior identical to missing keys.
    with_none = _sector_neutralize_scores(mat, {"A": None, "B": ""})  # type: ignore[dict-item]
    with_missing = _sector_neutralize_scores(mat, {})
    pd.testing.assert_frame_equal(with_none, with_missing)


# ---------------------------------------------------------------------------
# Agent-mode reports
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_agent_mode():
    agentio.reset()
    yield
    agentio.reset()


def test_resolve_report_path_skips_the_unsolicited_temp_file_in_agent_mode():
    agentio.configure(enabled=True)

    assert resolve_report_path(None, False, "backtest") is None
    assert resolve_report_path(None, True, "backtest") is None
    assert resolve_report_path(None, True, "backtest", open_report=True) is None
    opened = resolve_report_path(None, False, "backtest", open_report=True)
    assert opened is not None
    assert "backtest" in opened.name
    explicit = Path("explicit.html")
    assert resolve_report_path(explicit, False, "backtest") == explicit


def test_screen_workflow_skips_the_temp_report_when_skip_temp_report_is_set(
    monkeypatch,
):
    frame = _df("AAA")

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "value", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: pytest.fail("temp report path requested"),
        render_screen_report=lambda *args, **kwargs: pytest.fail(
            "unsolicited report rendered"
        ),
    )

    request = dataclasses.replace(_request(defer_report=True), skip_temp_report=True)
    outcome = sw.run_screen_workflow(request)

    assert outcome.report_path is None
    assert outcome.render_report is None


def test_screen_workflow_default_keeps_rendering_in_agent_mode_for_api_callers(
    monkeypatch, tmp_path
):
    """Python API callers (``skip_temp_report=False``) still get a report under agent env."""
    agentio.configure(enabled=True)
    frame = _df("AAA")
    temp = tmp_path / "api_temp.html"

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "value", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: temp,
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )

    outcome = sw.run_screen_workflow(_request())

    assert outcome.report_path == temp
    assert temp.read_text(encoding="utf-8") == "report"


def test_screen_workflow_honors_an_explicit_report_with_skip_temp_report(
    monkeypatch, tmp_path
):
    frame = _df("AAA")
    explicit = tmp_path / "explicit.html"

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "value", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )

    request = dataclasses.replace(_request(report_path=explicit), skip_temp_report=True)
    outcome = sw.run_screen_workflow(request)

    assert outcome.report_path == explicit
    assert explicit.read_text(encoding="utf-8") == "report"


def test_screen_workflow_honors_open_report_with_skip_temp_report(
    monkeypatch, tmp_path
):
    frame = _df("AAA")
    temp = tmp_path / "temp.html"

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "value", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: temp,
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )
    request = dataclasses.replace(
        _request(defer_report=True), open_report=True, skip_temp_report=True
    )

    outcome = sw.run_screen_workflow(request)

    assert outcome.report_path == temp
    assert outcome.render_report is not None


def test_screen_cli_passes_skip_temp_report_based_on_agent_mode(monkeypatch):
    from screener.commands import screen as screen_cmd

    captured: list[sw.ScreenRequest] = []

    def fake_workflow(request: sw.ScreenRequest) -> sw.ScreenOutcome:
        captured.append(request)
        return sw.ScreenOutcome(
            mode=sw.ScreenMode.CSV,
            market=request.market,
            label="ema",
            total=0,
            df=pd.DataFrame(),
            as_of=_AS_OF,
        )

    monkeypatch.setattr(screen_cmd, "run_screen_workflow", fake_workflow)
    monkeypatch.setattr(screen_cmd, "print_csv", lambda _df: None)

    runner = CliRunner()
    agentio.configure(enabled=True)
    assert runner.invoke(screen_cmd.screen, []).exit_code == 0
    agentio.configure(enabled=False)
    assert runner.invoke(screen_cmd.screen, []).exit_code == 0

    assert [req.skip_temp_report for req in captured] == [True, False]


# ---------------------------------------------------------------------------
# markets lazy import
# ---------------------------------------------------------------------------


def test_importing_markets_does_not_load_the_data_layer():
    code = (
        "import sys, screener.markets; "
        "assert 'screener.backtester.data' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_get_price_fetcher_resolves_the_default_builder_lazily(monkeypatch):
    from screener.backtester import data as data_module
    from screener.markets import get_price_fetcher

    sentinel = object()
    calls: list[dict[str, object]] = []

    def builder(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(data_module, "build_price_fetcher", builder)

    assert get_price_fetcher(None, refresh=True) is sentinel
    # A falsy non-None ctx.obj (e.g. {}) must still build the real fetcher.
    assert get_price_fetcher({}, refresh=False) is sentinel  # type: ignore[arg-type]
    assert calls == [{"refresh": True}, {"refresh": False}]
