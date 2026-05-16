"""Tests for Monte Carlo simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import Trade
from screener.backtester.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    _trade_returns,
    _rebuild_equity_from_returns,
)


def _make_trades(n: int = 20, seed: int = 42) -> list[Trade]:
    rng = np.random.default_rng(seed)
    trades = []
    for i in range(n):
        ret = rng.normal(0.02, 0.05)
        entry = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 5)
        exit_ = entry + pd.Timedelta(days=3)
        trades.append(
            Trade(
                ticker="AAPL",
                rank=1,
                signal_date=entry.date(),
                entry_date=entry.date(),
                entry_price=100.0,
                exit_date=exit_.date(),
                exit_price=100.0 * (1 + ret),
                exit_reason="hold_limit",
                shares=10.0,
                entry_cost=1000.0,
                exit_value=1000.0 * (1 + ret),
                pnl=1000.0 * ret,
                return_pct=ret,
                dividend_income=0.0,
            )
        )
    return trades


def test_trade_returns() -> None:
    trades = _make_trades(10)
    returns = _trade_returns(trades)
    assert len(returns) == 10
    assert all(isinstance(r, (float, np.floating)) for r in returns)


def test_rebuild_equity_from_returns() -> None:
    returns = np.array([0.01, -0.005, 0.02])
    calendar = pd.bdate_range("2024-01-01", periods=5)
    equity = _rebuild_equity_from_returns(returns, 10000.0, calendar)
    assert len(equity) == 4  # min(len(returns)+1, len(calendar))
    assert equity.iloc[0] == pytest.approx(10000.0)
    assert equity.iloc[1] == pytest.approx(10100.0)
    assert equity.iloc[2] == pytest.approx(10100.0 * 0.995)


def test_simulator_trade_shuffle() -> None:
    trades = _make_trades(30)
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.trade_shuffle(n_runs=100, seed=42)
    assert isinstance(result, MonteCarloResult)
    assert result.method == "trade_shuffle"
    assert result.n_runs == 100
    assert 0 <= result.probabilities_of_profit <= 1
    assert result.median_max_dd <= 0
    assert result.median_final_equity > 0


def test_simulator_returns_bootstrap() -> None:
    trades = _make_trades(30)
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.returns_bootstrap(n_runs=100, seed=42)
    assert isinstance(result, MonteCarloResult)
    assert result.method == "returns_bootstrap"
    assert result.n_runs == 100


def test_simulator_block_bootstrap() -> None:
    trades = _make_trades(30)
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.block_bootstrap(block_size=5, n_runs=100, seed=42)
    assert isinstance(result, MonteCarloResult)
    assert result.method == "block_bootstrap"
    assert result.n_runs == 100


def test_simulator_empty_trades() -> None:
    sim = MonteCarloSimulator([], initial_capital=100_000.0)
    result = sim.returns_bootstrap(n_runs=10)
    assert result.probabilities_of_profit == 0.0
    assert result.median_final_equity == pytest.approx(100_000.0)


def test_summarize_percentiles() -> None:
    trades = _make_trades(50)
    sim = MonteCarloSimulator(trades, initial_capital=100_000.0)
    result = sim.trade_shuffle(n_runs=200, seed=42)
    assert "max_drawdown" in result.percentile_breakdown
    assert "final_equity" in result.percentile_breakdown
    assert "sharpe" in result.percentile_breakdown
    # Check that percentiles are ordered
    dd = result.percentile_breakdown["max_drawdown"]
    assert dd["p5"] <= dd["p50"] <= dd["p95"]
