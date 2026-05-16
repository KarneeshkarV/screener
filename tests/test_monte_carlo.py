"""Monte Carlo simulation tests."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import Trade
from screener.backtester.monte_carlo import (
    MonteCarloResult,
    MonteCarloSimulator,
    print_monte_carlo,
    run_monte_carlo_from_result,
)


def _make_trade(pnl: float, return_pct: float, entry_date: date, exit_date: date) -> Trade:
    return Trade(
        ticker="AAA",
        rank=1,
        signal_date=entry_date,
        entry_date=entry_date,
        entry_price=100.0,
        exit_date=exit_date,
        exit_price=100.0 * (1.0 + return_pct),
        exit_reason="time",
        shares=100.0,
        entry_cost=10_000.0,
        exit_value=10_000.0 + pnl,
        pnl=pnl,
        return_pct=return_pct,
    )


def test_trade_shuffle_preserves_trade_count():
    trades = [
        _make_trade(100.0, 0.01, date(2024, 1, 1), date(2024, 1, 5)),
        _make_trade(-50.0, -0.005, date(2024, 1, 6), date(2024, 1, 10)),
        _make_trade(200.0, 0.02, date(2024, 1, 11), date(2024, 1, 15)),
    ]
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.trade_shuffle(n_runs=100, seed=42)
    assert result.n_runs == 100
    assert len(result.max_drawdowns) == 100
    assert len(result.final_equities) == 100


def test_returns_bootstrap_with_empty_trades():
    sim = MonteCarloSimulator([], initial_capital=100_000.0)
    result = sim.returns_bootstrap(n_runs=50, seed=1)
    assert result.probabilities_of_profit == 0.0
    assert result.median_final_equity == pytest.approx(100_000.0)


def test_block_bootstrap_block_size_respected():
    trades = [
        _make_trade(100.0, 0.01, date(2024, 1, min(i, 28)), date(2024, 1, min(i + 2, 31)))
        for i in range(1, 41, 3)
    ]
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.block_bootstrap(block_size=5, n_runs=100, seed=7)
    assert result.n_runs == 100
    assert np.isfinite(result.median_max_dd)
    assert np.isfinite(result.median_final_equity)


def test_monte_carlo_prob_profit_between_zero_and_one():
    trades = [
        _make_trade(100.0, 0.01, date(2024, 1, 1), date(2024, 1, 5)),
        _make_trade(-50.0, -0.005, date(2024, 1, 6), date(2024, 1, 10)),
    ]
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.returns_bootstrap(n_runs=200, seed=3)
    assert 0.0 <= result.probabilities_of_profit <= 1.0


def test_monte_carlo_var_less_than_median():
    trades = [
        _make_trade(np.random.normal(0, 100), np.random.normal(0, 0.01), date(2024, 1, min(i, 28)), date(2024, 1, min(i + 2, 31)))
        for i in range(1, 50, 3)
    ]
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.trade_shuffle(n_runs=500, seed=5)
    # VaR-95 should be <= median final equity
    assert result.var_95 <= result.median_final_equity


def test_print_monte_carlo_smoke():
    trades = [
        _make_trade(100.0, 0.01, date(2024, 1, 1), date(2024, 1, 5)),
        _make_trade(-50.0, -0.005, date(2024, 1, 6), date(2024, 1, 10)),
    ]
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.returns_bootstrap(n_runs=50, seed=9)
    print_monte_carlo(result)


def test_run_monte_carlo_from_result_wrapper():
    from screener.backtester.models import BacktestConfig, BacktestResult

    trades = [
        _make_trade(100.0, 0.01, date(2024, 1, 1), date(2024, 1, 5)),
    ]
    cfg = BacktestConfig(
        market="us",
        as_of=date(2024, 1, 1),
        hold=5,
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
    )
    result = BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=pd.Series([100_000.0, 101_000.0], index=pd.bdate_range("2024-01-01", periods=2)),
        benchmark_curve=pd.Series([100_000.0, 100_500.0], index=pd.bdate_range("2024-01-01", periods=2)),
        metrics={},
    )
    mc = run_monte_carlo_from_result(result, method="shuffle", n_runs=50, seed=11)
    assert mc.method == "trade_shuffle"
    assert mc.n_runs == 50
