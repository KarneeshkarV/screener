"""Behavioral tests for Stage2Stocks (Stan Weinstein) strategies and runner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.stage2stocks import (
    STAGE2_EXIT_EXPR,
    VARIANTS,
    compute_stage_series,
    prepare_stage2_bars,
    required_history_bars,
)
from screener.strategies.spec import (
    ExpressionStrategySpec,
    PrepareCtx,
    discover_plugins,
    resolve_strategy_spec,
)
from scripts.run_stage2stocks_backtest import (
    InMemoryPriceFetcher,
    benchmark_stats,
    make_request,
)
from tests.conftest import StubPriceFetcher

_N = 360
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _ohlcv(close: pd.Series, volume: float | pd.Series = 50_000.0) -> pd.DataFrame:
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) * 1.005
    low = pd.concat([openp, close], axis=1).min(axis=1) * 0.995
    vol_s = (
        volume
        if isinstance(volume, pd.Series)
        else pd.Series(float(volume), index=close.index, dtype=float)
    )
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close.astype(float),
            "volume": vol_s,
        },
        index=close.index,
    )


def _cycle_close() -> pd.Series:
    """Synthetic price series that traverses Stage 4 -> 1 -> 2 -> 3 -> 4."""
    # 0..149: steady decline -> Stage 4 once SMA200 warms up
    # 150..219: flat/rising base -> SMA50 turns up below SMA200 (Stage 1)
    # 220..309: strong advance -> SMA50 > SMA200 and rising (Stage 2)
    # 310..359: sharp rollover -> SMA50 still > SMA200 but falling (Stage 3)
    p1 = np.linspace(200.0, 90.0, 150)
    p2 = np.linspace(90.0, 115.0, 70)
    p3 = np.linspace(116.0, 260.0, 90)
    p4 = np.linspace(255.0, 165.0, 50)
    return pd.Series(np.concatenate([p1, p2, p3, p4]), index=_INDEX, dtype=float)


def test_all_stage2_variants_register_with_lookback() -> None:
    discover_plugins()
    for name, variant in VARIANTS.items():
        spec = resolve_strategy_spec(name)
        assert isinstance(spec, ExpressionStrategySpec)
        assert spec.entry == variant.entry
        assert spec.exit == STAGE2_EXIT_EXPR
        assert spec.prepare_bars is not None
        assert spec.required_lookback is not None
        assert spec.required_lookback() == required_history_bars()
        assert spec.required_lookback() > 252


def test_stage_classifier_covers_all_four_weinstein_stages_and_weeks() -> None:
    close = _cycle_close()
    stage, weeks = compute_stage_series(close)

    # First 199 bars lack SMA200 warmup -> NaN stage, 0 weeks.
    assert stage.iloc[:199].isna().all()
    assert (weeks.iloc[:199] == 0.0).all()

    observed = set(stage.dropna().unique().tolist())
    assert {1.0, 2.0, 3.0} <= observed

    # Pure declining series produces Stage 4 once warmed up.
    declining = pd.Series(np.linspace(300.0, 80.0, _N), index=_INDEX)
    dec_stage, dec_weeks = compute_stage_series(declining)
    assert (dec_stage.iloc[220:] == 4.0).all()
    assert (dec_weeks == 0.0).all()

    # Weeks in Stage 2 start at 1 on entry, increment every 5 sessions, reset on exit.
    s2_mask = stage == 2.0
    first_s2 = int(np.flatnonzero(s2_mask.to_numpy())[0])
    assert weeks.iloc[first_s2] == 1.0
    assert weeks.iloc[first_s2 + 5] == 2.0
    after_s2 = stage.iloc[first_s2:] != 2.0
    if after_s2.any():
        first_exit = first_s2 + int(np.flatnonzero(after_s2.to_numpy())[0])
        assert weeks.iloc[first_exit] == 0.0


def test_stage_classifier_flat_deadband_hysteresis_and_causality() -> None:
    # Flat price after an advance keeps Stage 2 inside the deadband rather than flipping.
    up = np.linspace(100.0, 220.0, 240)
    flat = np.full(40, 220.0)
    close = pd.Series(np.concatenate([up, flat]), index=_INDEX[:280])
    stage, _ = compute_stage_series(close)
    assert stage.iloc[235] == 2.0
    assert stage.iloc[245] == 2.0

    # Strict causality: appending a crash after bar `t` must not alter bar `t`.
    crashed = close.copy()
    crashed.iloc[245:] = 50.0
    stage_crashed, weeks_crashed = compute_stage_series(crashed)
    assert stage_crashed.iloc[244] == stage.iloc[244]
    assert weeks_crashed.iloc[244] == compute_stage_series(close)[1].iloc[244]


def test_prepare_stage2_masks_out_of_membership_and_dormant_shells() -> None:
    # A and B are active PIT members; FUTURE joins only at bar 330; DORMANT has 0 volume.
    bars_a = _ohlcv(pd.Series(100.0 * (1.003 ** np.arange(_N)), index=_INDEX))
    bars_b = _ohlcv(pd.Series(100.0 * (1.001 ** np.arange(_N)), index=_INDEX))
    bars_future = _ohlcv(pd.Series(100.0 * (1.006 ** np.arange(_N)), index=_INDEX))
    bars_dormant = _ohlcv(
        pd.Series(100.0 * (1.008 ** np.arange(_N)), index=_INDEX), volume=0.0
    )

    ctx = PrepareCtx(
        market="india",
        benchmark="^NSEI",
        bars_by_tv={
            "NSE:A": bars_a,
            "NSE:B": bars_b,
            "NSE:FUTURE": bars_future,
            "NSE:DORMANT": bars_dormant,
            "NSE:EMPTY": pd.DataFrame(),
        },
        price_panel={"^NSEI": bars_b},
        tv_symbols=["NSE:A", "NSE:B", "NSE:FUTURE", "NSE:DORMANT", "NSE:EMPTY"],
        start=_INDEX[260].date(),
        end=_INDEX[-1].date(),
        fetcher=StubPriceFetcher({}),
        warnings=[],
        membership_windows=(
            ("NSE:A", _INDEX[0].date(), None),
            ("NSE:B", _INDEX[0].date(), None),
            ("NSE:FUTURE", _INDEX[330].date(), None),
            ("NSE:DORMANT", _INDEX[0].date(), None),
        ),
    )

    # RS rank mode: FUTURE must be NaN before its join date and not displace A.
    prep_rs = prepare_stage2_bars("rs_rank")(ctx)
    probe = _INDEX[300]
    assert pd.isna(prep_rs["NSE:FUTURE"].loc[probe, "rs_rank"])
    assert pd.isna(prep_rs["NSE:DORMANT"].loc[probe, "rs_rank"])
    assert pd.isna(prep_rs["NSE:DORMANT"].loc[probe, "s2_ext50"])
    assert (
        prep_rs["NSE:A"].loc[probe, "rs_rank"] > prep_rs["NSE:B"].loc[probe, "rs_rank"]
    )
    assert prep_rs["NSE:A"].loc[probe, "breadth_s2"] == pytest.approx(1.0)
    assert prep_rs["NSE:EMPTY"].empty

    # Setup score mode: computes bounded 0..10 scores and rejects unknown rank_mode.
    prep_setup = prepare_stage2_bars("setup_score")(ctx)
    score_a = prep_setup["NSE:A"].loc[probe, "rank_score"]
    score_b = prep_setup["NSE:B"].loc[probe, "rank_score"]
    assert 0.0 <= score_b < score_a <= 10.0

    with pytest.raises(ValueError, match="unknown stage2stocks rank_mode"):
        prepare_stage2_bars("invalid_mode")


def test_stage2_early_entry_runs_end_to_end_on_rolling_engine() -> None:
    discover_plugins()
    # Gentle Stage 1 -> Stage 2 -> Stage 3 cycle (stays within 18% of SMA50 in early Stage 2)
    # alongside a perpetually declining Stage 4 stock that must never be entered.
    n_long = 440
    idx = pd.bdate_range("2021-01-04", periods=n_long)
    steady = np.concatenate(
        [
            np.linspace(130.0, 95.0, 140),  # decline
            np.linspace(95.0, 102.0, 130),  # long Stage 1 base
            np.linspace(102.5, 138.0, 110),  # calm Stage 2 advance (~5-10% above SMA50)
            np.linspace(137.0, 100.0, 60),  # Stage 3 rollover -> exit
        ]
    )
    cycle_bars = _ohlcv(pd.Series(steady, index=idx), volume=100_000.0)
    declining_bars = _ohlcv(
        pd.Series(np.linspace(300.0, 90.0, n_long), index=idx), volume=100_000.0
    )
    bench_bars = _ohlcv(
        pd.Series(np.linspace(15000.0, 18000.0, n_long), index=idx),
        volume=1_000_000.0,
    )
    fetcher = StubPriceFetcher(
        {
            "CYCLE.NS": cycle_bars,
            "DECL.NS": declining_bars,
            "^NSEI": bench_bars,
        }
    )

    spec = resolve_strategy_spec("stage2_early_entry")
    assert isinstance(spec, ExpressionStrategySpec)
    cfg = BacktestConfig(
        market="india",
        as_of=idx[-1].date(),
        benchmark="^NSEI",
        tickers=("NSE:CYCLE", "NSE:DECL"),
        strategy_name="stage2_early_entry",
        entry_expr=spec.entry,
        exit_expr=spec.exit,
        hold=5000,
        top=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        commission_bps=0.0,
        initial_capital=1_000_000.0,
        cost_model="india",
        slippage_bps=10.0,
        sizing_rule="reinvested_equal_slot",
        compounding=True,
    )
    result = run_rolling_backtest(
        cfg,
        fetcher,
        start_date=idx[260].date(),
        end_date=idx[-1].date(),
    )

    assert len(result.trades) >= 1
    assert all(trade.ticker == "NSE:CYCLE" for trade in result.trades)
    assert any(trade.exit_reason == "exit_expr" for trade in result.trades)


def test_runner_request_and_helpers() -> None:
    req = make_request(
        strategy_name="stage2_leaders",
        universe="india_turnover500_pit",
        universe_config=Path("universes.yaml"),
        start=date(2021, 11, 23),
        end=date(2026, 11, 23),
    )
    assert req.strategy_name == "stage2_leaders"
    assert req.trailing_stop == pytest.approx(0.20)
    assert req.cost_model == "india"
    assert req.point_in_time is True

    # InMemoryPriceFetcher caches on first fetch and slices subsequent calls.
    calls: list[list[str]] = []
    sample = _ohlcv(pd.Series(np.linspace(100.0, 150.0, 30), index=_INDEX[:30]))

    class _CountingFetcher:
        def fetch(self, tickers, start, end):
            calls.append(list(tickers))
            return {t: sample for t in tickers}

    mem = InMemoryPriceFetcher(
        _CountingFetcher(),
        {},
        prefetch_start=_INDEX[0].date(),
        prefetch_end=_INDEX[29].date(),
    )
    first = mem.fetch(["AAA.NS"], _INDEX[0].date(), _INDEX[10].date())
    second = mem.fetch(["AAA.NS"], _INDEX[5].date(), _INDEX[15].date())
    assert len(calls) == 1
    assert len(first["AAA.NS"]) == 11
    assert len(second["AAA.NS"]) == 11

    stats = benchmark_stats(sample["close"], risk_free_rate=0.06)
    assert stats["total_return"] == pytest.approx(0.5)
    assert stats["max_drawdown"] == pytest.approx(0.0)
