"""Tests for walk-forward optimization."""
from __future__ import annotations

import pytest

from screener.backtester.walk_forward import (
    WalkForwardOptimizer,
    WindowResult,
    WalkForwardResult,
    _generate_param_combinations,
)
from screener.backtester.models import BacktestConfig


def test_generate_param_combinations() -> None:
    grid = {"a": [1, 2], "b": ["x", "y"]}
    combos = _generate_param_combinations(grid)
    assert len(combos) == 4
    assert all(isinstance(c, dict) for c in combos)
    assert {"a": 1, "b": "x"} in combos


def test_generate_empty_grid() -> None:
    combos = _generate_param_combinations({})
    assert combos == [{}]


def test_make_cfg_copies_base() -> None:
    base = BacktestConfig(
        market="us",
        as_of=None,
        hold=20,
        top=5,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=0.05,
        take_profit=0.10,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=None,
        tickers=None,
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        avg_dollar_volume_window=20,
        reserve_multiple=3,
        reinvest=False,
        slippage_model=None,
        gap_fills=True,
        entry_order_type="moo",
        entry_limit_bps=None,
        allow_reentry=False,
        max_reentries=0,
        partial_exits=(),
        price_adjustment="full",
    )
    opt = WalkForwardOptimizer(base_cfg=base, fetcher=None, param_grid={"stop_loss": [0.05, 0.10]})
    cfg = opt._make_cfg({"stop_loss": 0.15})
    assert cfg.stop_loss == pytest.approx(0.15)
    assert cfg.hold == base.hold  # unchanged
    assert cfg.market == base.market
