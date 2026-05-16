"""Tests for parameter sensitivity analysis."""
from __future__ import annotations

import pytest

from screener.backtester.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
)
from screener.backtester.models import BacktestConfig


def test_sensitivity_result_structure() -> None:
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
    analyzer = SensitivityAnalyzer(
        base_cfg=base,
        fetcher=None,
        param_grid={"stop_loss": [0.05, 0.10]},
        start_date=None,
        end_date=None,
    )
    cfg = analyzer._make_cfg({"stop_loss": 0.07})
    assert cfg.stop_loss == pytest.approx(0.07)
    assert cfg.take_profit == pytest.approx(0.10)
