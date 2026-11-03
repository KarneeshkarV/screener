"""Historical and rolling engines must agree on who is bought, when, and why.

Each test pins one defect found in the November 2026 engine review. Every one
of them fails on the code before the fix.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from screener.backtester import BacktestConfig, run_backtest
from screener.backtester.core import _SlotState
from screener.backtester.historical import (
    _rolling_only_gate_warnings,
    select_candidates,
)
from screener.backtester.metrics import compute_regime_metrics
from screener.backtester.pine import parse
from screener.backtester.rolling_simulation import _mark_rank_exits
from tests.conftest import StubPriceFetcher

_DAYS = pd.bdate_range("2024-01-01", periods=40)


def _frame(
    index: pd.DatetimeIndex,
    *,
    close: float = 100.0,
    volume: float = 1_000_000.0,
    signal: list[bool] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame["sig"] = 1.0 if signal is None else [1.0 if s else 0.0 for s in signal]
    return frame


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=_DAYS[5].date(),
        hold=2,
        top=1,
        entry_expr="sig > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _signal_on(*days: int) -> list[bool]:
    return [i in days for i in range(len(_DAYS))]


def _run(panel: dict[str, pd.DataFrame], **overrides):
    panel = dict(panel)
    panel.setdefault("SPY", _frame(_DAYS))
    tickers = tuple(name for name in panel if name != "SPY")
    return run_backtest(_cfg(tickers=tickers, **overrides), StubPriceFetcher(panel))


def test_historical_ranks_factor_strategy_by_rank_score():
    liquid = _frame(_DAYS, volume=1_000_000.0)
    liquid["rank_score"] = 1.0
    strong = _frame(_DAYS, close=50.0, volume=10_000.0)
    strong["rank_score"] = 999.0
    unscored = _frame(_DAYS, volume=5_000_000.0)
    unscored["rank_score"] = np.nan
    selection, _ = select_candidates(
        {"LIQUID": liquid, "STRONG": strong, "UNSCORED": unscored},
        parse("sig > 0"),
        _DAYS[5],
        top_n=1,
        lookback_required=0,
        cfg=_cfg(reserve_multiple=3),
    )
    assert selection["ticker"].tolist() == ["STRONG", "LIQUID"]
    assert selection["role"].tolist() == ["active", "reserve"]


def test_historical_applies_min_score_to_the_signal_field():
    bars = {
        name: _frame(_DAYS, volume=volume)
        for name, volume in (("A", 4e6), ("B", 3e6), ("C", 2e6), ("D", 1e6))
    }
    selection, _ = select_candidates(
        bars,
        parse("sig > 0"),
        _DAYS[5],
        top_n=4,
        lookback_required=0,
        cfg=_cfg(min_score=60.0),
    )
    assert selection["ticker"].tolist() == ["A", "B"]


def test_historical_warns_about_gates_it_does_not_apply():
    assert _rolling_only_gate_warnings(_cfg()) == []
    (warning,) = _rolling_only_gate_warnings(
        _cfg(sector_neutral=True, earnings_blackout_days=3)
    )
    assert "earnings_blackout_days" in warning
    assert "sector_neutral" in warning


def test_non_finite_close_never_passes_with_filters_off():
    bars = _frame(_DAYS)
    bars.iloc[5, bars.columns.get_loc("close")] = np.nan
    selection, _ = select_candidates(
        {"NAN": bars},
        parse("sig > 0"),
        _DAYS[5],
        top_n=1,
        lookback_required=0,
        cfg=_cfg(),
    )
    assert selection.empty


def test_limit_order_without_limit_bps_is_refused():
    with pytest.raises(ValueError, match="requires entry_limit_bps"):
        _cfg(entry_order_type="limit", entry_limit_bps=None)


def test_weekend_as_of_signs_and_stamps_the_prior_session():
    result = _run({"AAA": _frame(_DAYS)}, as_of=date(2024, 1, 6))
    (trade,) = result.trades[:1]
    assert pd.Timestamp(trade.signal_date) == pd.Timestamp("2024-01-05")
    assert pd.Timestamp(trade.entry_date) == pd.Timestamp("2024-01-08")


def test_intraday_date_as_of_reads_that_session():
    hours = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day.date()} {hour}:00")
            for day in pd.bdate_range("2024-03-04", periods=4)
            for hour in (14, 15, 16)
        ]
    )
    sig = [stamp == pd.Timestamp("2024-03-05 16:00") for stamp in hours]
    panel = {"AAA": _frame(hours, signal=sig), "SPY": _frame(hours)}
    result = run_backtest(
        _cfg(tickers=("AAA",), as_of=date(2024, 3, 5), interval="1h"),
        StubPriceFetcher(panel),
    )
    (trade,) = result.trades
    assert pd.Timestamp(trade.signal_date) == pd.Timestamp("2024-03-05 16:00")
    assert pd.Timestamp(trade.entry_date) == pd.Timestamp("2024-03-06 14:00")


def test_intraday_benchmark_starts_at_the_signal_bar():
    hours = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day.date()} {hour}:00")
            for day in pd.bdate_range("2024-03-04", periods=4)
            for hour in (14, 15, 16)
        ]
    )
    sig = [stamp == pd.Timestamp("2024-03-05 16:00") for stamp in hours]
    # The benchmark doubles before the signal bar and is flat after it.
    spy = _frame(hours)
    spy.loc[hours >= pd.Timestamp("2024-03-05 16:00"), ["open", "close"]] = 200.0
    panel = {"AAA": _frame(hours, signal=sig), "SPY": spy}
    result = run_backtest(
        _cfg(tickers=("AAA",), as_of=date(2024, 3, 5), interval="1h"),
        StubPriceFetcher(panel),
    )
    assert result.metrics["benchmark_return"] == pytest.approx(0.0)


def test_halted_reserve_does_not_trade_its_last_stale_signal():
    # Halted from the first bar after as_of through every possible exit day.
    halted_days = _DAYS.delete(list(range(6, 14)))
    reserve = _frame(
        halted_days, volume=10.0, signal=[d == _DAYS[5] for d in halted_days]
    )
    active = _frame(_DAYS, signal=_signal_on(5))
    result = _run({"ACTIVE": active, "RESERVE": reserve}, reserve_multiple=2)
    assert [t.ticker for t in result.trades] == ["ACTIVE"]


def test_reentry_does_not_hold_the_slot_against_a_signalling_reserve():
    active = _frame(_DAYS, signal=_signal_on(5))
    reserve = _frame(_DAYS, volume=10.0, signal=_signal_on(5, 8, 9))
    result = _run(
        {"ACTIVE": active, "RESERVE": reserve},
        reserve_multiple=2,
        allow_reentry=True,
        max_reentries=1,
    )
    assert [t.ticker for t in result.trades] == ["ACTIVE", "RESERVE"]


def test_reserve_that_misses_one_rotation_stays_queued():
    active = _frame(_DAYS, signal=_signal_on(5))
    late = _frame(_DAYS, volume=10.0, signal=_signal_on(5, 9))
    result = _run({"ACTIVE": active, "LATE": late}, reserve_multiple=2)
    tickers = [t.ticker for t in result.trades]
    assert tickers == ["ACTIVE", "LATE"]
    assert pd.Timestamp(result.trades[1].signal_date) == _DAYS[9]


def test_cancelled_reserve_order_frees_the_name_for_a_later_rotation():
    active = _frame(_DAYS, signal=_signal_on(5))
    flaky = _frame(_DAYS, volume=10.0)
    flaky.iloc[9, flaky.columns.get_loc("open")] = np.nan
    result = _run({"ACTIVE": active, "FLAKY": flaky}, reserve_multiple=2)
    flaky_trades = [t for t in result.trades if t.ticker == "FLAKY"]
    assert flaky_trades, "a cancelled order must not blacklist the name"


def test_regime_metrics_bucket_by_signal_date():
    benchmark = pd.Series(
        np.r_[np.linspace(100.0, 200.0, 260), [50.0]],
        index=pd.bdate_range("2023-01-02", periods=261),
    )
    signal_day, entry_day = benchmark.index[-2], benchmark.index[-1]
    trade = SimpleNamespace(
        signal_date=signal_day, entry_date=entry_day, return_pct=0.1
    )
    metrics = compute_regime_metrics(benchmark, [trade])
    assert metrics["regime_bull_trades"] == 1
    assert "regime_bear_trades" not in metrics


def _state(ticker: str, signal_date: pd.Timestamp) -> _SlotState:
    return _SlotState(
        ticker=ticker,
        entry_idx=0,
        entry_date=signal_date,
        entry_fill=1.0,
        signal_date=signal_date,
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=10,
        peak=1.0,
        exit_signal=None,
    )


def test_rank_exit_spares_a_name_picked_from_the_ranking_it_is_judged_on(
    monkeypatch,
):
    monkeypatch.setattr(
        "screener.backtester.rolling_simulation._candidate_rows_for_day",
        lambda *args, **kwargs: ([], []),
    )
    prev_day = _DAYS[10]
    fresh = _state("FRESH", prev_day)
    old = _state("OLD", _DAYS[5])
    _mark_rank_exits(
        prev_day=prev_day,
        cfg=_cfg(rank_exit_every=1, rank_universe_size=1),
        candidate_matrices=None,  # type: ignore[arg-type]
        slot_states={0: fresh, 1: old},
        warnings=[],
    )
    assert fresh.rank_exit is False
    assert old.rank_exit is True
