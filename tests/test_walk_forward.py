"""Walk-forward optimization tests."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.walk_forward import WalkForwardOptimizer, print_walk_forward

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 6, 1),
        hold=5,
        top=1,
        entry_expr="close > sma(close, 3)",
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
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_walk_forward_splits_windows_correctly():
    """Optimizer should generate non-overlapping test windows."""
    bars = make_bars(n=120, seed=1)
    spy = make_bars(n=120, seed=2, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), as_of=date(2024, 6, 1))
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=3,
        test_months=1,
        param_grid={"hold": [3, 5]},
    )
    windows = optimizer._build_windows(date(2024, 1, 1), date(2024, 12, 31))
    assert len(windows) >= 2
    # Test periods should not overlap
    for i in range(1, len(windows)):
        assert windows[i][2] >= windows[i - 1][3]


def test_walk_forward_returns_combined_trades():
    """At least some OOS trades should be produced with a clear signal."""
    n = 120
    bars = make_bars(n=n, seed=3)
    # Force a reliable entry signal on every bar after lookback
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=4, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        tickers=("AAA",),
        as_of=date(2024, 6, 1),
        entry_expr="entry_signal > 0",
    )
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=2,
        test_months=1,
        param_grid={"hold": [3, 5]},
    )
    result = optimizer.run(date(2024, 1, 1), date(2024, 8, 31))
    assert result.window_results
    assert result.combined_trades or any(
        wr.oos_result.trades for wr in result.window_results
    )


def test_walk_forward_param_stability_range():
    """Stability values should be in [0, 1]."""
    n = 120
    bars = make_bars(n=n, seed=5)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=6, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), as_of=date(2024, 6, 1), entry_expr="entry_signal > 0")
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=2,
        test_months=1,
        param_grid={"hold": [3, 5]},
    )
    result = optimizer.run(date(2024, 1, 1), date(2024, 8, 31))
    for k, v in result.param_stability.items():
        assert 0.0 <= v <= 1.0


def test_walk_forward_empty_param_grid_uses_defaults():
    """With no param grid, the optimizer should still run windows."""
    n = 120
    bars = make_bars(n=n, seed=7)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=8, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), as_of=date(2024, 6, 1), entry_expr="entry_signal > 0")
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=2,
        test_months=1,
        param_grid={},
    )
    result = optimizer.run(date(2024, 1, 1), date(2024, 8, 31))
    assert result.window_results


def test_print_walk_forward_smoke():
    """Smoke test for the rich printer."""
    n = 120
    bars = make_bars(n=n, seed=9)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=10, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), as_of=date(2024, 6, 1), entry_expr="entry_signal > 0")
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=2,
        test_months=1,
        param_grid={"hold": [3, 5]},
    )
    result = optimizer.run(date(2024, 1, 1), date(2024, 8, 31))
    print_walk_forward(result)
