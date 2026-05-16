"""Parameter sensitivity analysis tests."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.sensitivity import (
    SensitivityAnalyzer,
    print_sensitivity,
    save_heatmap,
)

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


def test_sensitivity_grid_search_produces_results():
    n = 80
    bars = make_bars(n=n, seed=1)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=2, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), entry_expr="entry_signal > 0")
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid={"hold": [3, 5, 10]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )
    result = analyzer.run()
    assert len(result.values) == 3
    assert result.param_names == ("hold",)


def test_sensitivity_two_param_grid():
    n = 80
    bars = make_bars(n=n, seed=3)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=4, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), entry_expr="entry_signal > 0")
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid={"hold": [3, 5], "stop_loss": [0.05, 0.10]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )
    result = analyzer.run()
    assert len(result.values) == 4
    assert result.param_names == ("hold", "stop_loss")


def test_sensitivity_metrics_map_populated():
    n = 80
    bars = make_bars(n=n, seed=5)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=6, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), entry_expr="entry_signal > 0")
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid={"hold": [3, 5]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )
    result = analyzer.run()
    assert "sharpe" in result.metrics_map or "total_return" in result.metrics_map


def test_print_sensitivity_smoke():
    n = 80
    bars = make_bars(n=n, seed=7)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=8, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), entry_expr="entry_signal > 0")
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid={"hold": [3, 5]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )
    result = analyzer.run()
    print_sensitivity(result)


def test_save_heatmap_skips_without_matplotlib():
    n = 80
    bars = make_bars(n=n, seed=9)
    bars["entry_signal"] = 0.0
    bars.iloc[10:, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=n, seed=10, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(tickers=("AAA",), entry_expr="entry_signal > 0")
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid={"hold": [3, 5], "stop_loss": [0.05, 0.10]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )
    result = analyzer.run()
    # Should not raise even if matplotlib is unavailable
    save_heatmap(result, metric="sharpe", out_path="/tmp/test_heatmap.png")
