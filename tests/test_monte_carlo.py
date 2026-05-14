from __future__ import annotations

import pandas as pd
import pytest

from screener.backtester.monte_carlo import run_monte_carlo
from screener.backtester.pine import parse
from screener.backtester.vbt_adapter import run_vbt

from tests.backtester_synthetic import (
    STRATEGY_FIXTURES,
    fixture_config,
    run_core_portfolio_path,
    synthetic_ohlcv_panel,
)


def _fixture_result(name: str):
    fixture = next(item for item in STRATEGY_FIXTURES if item.name == name)
    return run_core_portfolio_path(fixture_config(fixture), synthetic_ohlcv_panel())


def test_monte_carlo_accepts_vbt_result():
    fixture = next(item for item in STRATEGY_FIXTURES if item.name == "buy_and_hold")
    cfg = fixture_config(fixture)
    result = run_vbt(
        cfg,
        synthetic_ohlcv_panel(),
        parse(cfg.entry_expr),
        parse(cfg.exit_expr) if cfg.exit_expr else None,
    )

    mc_result = run_monte_carlo(result, n_sims=20, seed=101)

    assert mc_result.sim_metrics.shape[0] == 20
    assert mc_result.sim_equity_curves.shape[1] == 20


def test_monte_carlo_is_deterministic_with_fixed_seed():
    result = _fixture_result("sma_cross")
    left = run_monte_carlo(result, n_sims=100, method="bootstrap_returns", seed=123)
    right = run_monte_carlo(result, n_sims=100, method="bootstrap_returns", seed=123)

    pd.testing.assert_frame_equal(left.sim_equity_curves, right.sim_equity_curves)
    pd.testing.assert_frame_equal(left.sim_metrics, right.sim_metrics)
    assert left.confidence == right.confidence


@pytest.mark.parametrize(
    "method",
    ["bootstrap_trades", "bootstrap_returns", "block_bootstrap"],
)
def test_monte_carlo_percentiles_are_ordered(method):
    result = _fixture_result("rsi_meanreversion")
    mc_result = run_monte_carlo(result, n_sims=100, method=method, seed=456)

    for row in mc_result.confidence.values():
        assert row["p05"] <= row["p50"] <= row["p95"]


def test_bootstrap_trades_mean_terminal_return_tracks_realized_buy_and_hold():
    result = _fixture_result("buy_and_hold")
    mc_result = run_monte_carlo(
        result,
        n_sims=1000,
        method="bootstrap_trades",
        seed=789,
    )
    realized = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1.0)
    simulated_mean = float(mc_result.sim_metrics["terminal_return"].mean())
    assert simulated_mean == pytest.approx(realized, rel=0.05)


def test_block_bootstrap_preserves_daily_volatility():
    result = _fixture_result("buy_and_hold")
    realized_vol = float(result.equity_curve.pct_change().dropna().std(ddof=0))
    mc_result = run_monte_carlo(
        result,
        n_sims=400,
        method="block_bootstrap",
        block_size=5,
        seed=321,
    )
    simulated_vol = float(mc_result.sim_metrics["vol"].mean())
    assert simulated_vol == pytest.approx(realized_vol, rel=0.10)
