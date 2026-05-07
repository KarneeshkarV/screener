from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from main import cli
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization import grid as grid_module
from screener.backtester.optimization.grid import _cache_key, grid_search, parameter_combinations
from screener.backtester.optimization.metrics import (
    expectancy,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from screener.backtester.optimization.monte_carlo import simulate_monte_carlo
from screener.backtester.optimization.walk_forward import generate_walk_forward_windows
from screener.backtester.slippage import FixedBpsSlippage, HalfSpreadSlippage
from tests.conftest import StubPriceFetcher, make_bars


def _trade(pnl: float, return_pct: float) -> Trade:
    return Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 3),
        exit_price=100.0 * (1.0 + return_pct),
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=100.0 + pnl,
        pnl=pnl,
        return_pct=return_pct,
    )


def _config(**overrides) -> BacktestConfig:
    values = {
        "market": "us",
        "as_of": date(2024, 1, 1),
        "hold": 5,
        "top": 1,
        "entry_expr": "close > sma(close, 3)",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA",),
    }
    values.update(overrides)
    return BacktestConfig(**values)


def test_grid_parameter_combinations_count():
    combos = parameter_combinations(
        {
            "stop_loss": [None, 0.05],
            "take_profit": [0.1, 0.2, 0.3],
            "hold": [10, 20],
        }
    )
    assert len(combos) == 12
    assert {"stop_loss": None, "take_profit": 0.1, "hold": 10} in combos


def test_walk_forward_generates_expected_windows():
    windows = generate_walk_forward_windows(
        date(2024, 1, 1),
        date(2024, 4, 30),
        train_days=30,
        test_days=10,
        step_days=20,
    )
    assert windows[0].train_start == date(2024, 1, 1)
    assert windows[0].train_end == date(2024, 1, 30)
    assert windows[0].test_start == date(2024, 1, 31)
    assert windows[0].test_end == date(2024, 2, 9)
    assert len(windows) == 5


def test_monte_carlo_reproducible_with_same_seed():
    trades = [_trade(10.0, 0.10), _trade(-5.0, -0.05), _trade(3.0, 0.03)]
    a = simulate_monte_carlo(trades, iterations=100, seed=7)
    b = simulate_monte_carlo(trades, iterations=100, seed=7)
    assert a == b


def test_monte_carlo_bootstrap_has_terminal_return_distribution():
    trades = [
        _trade(10.0, 0.10),
        _trade(-5.0, -0.05),
        _trade(3.0, 0.03),
        _trade(-2.0, -0.02),
    ]
    result = simulate_monte_carlo(trades, iterations=1000, seed=7)
    assert result.return_p05 < result.return_p95


def test_grid_cache_key_includes_min_trades(monkeypatch, tmp_path):
    calls = 0

    def fake_run_backtest(cfg, fetcher):
        nonlocal calls
        calls += 1
        equity = pd.Series([100_000.0, 101_000.0])
        return BacktestResult(
            config=cfg,
            trades=[_trade(10.0, 0.10)],
            equity_curve=equity,
            benchmark_curve=equity,
            metrics={"total_return": 0.01},
        )

    monkeypatch.setattr(grid_module, "run_backtest", fake_run_backtest)
    cfg = _config()
    cache_path = tmp_path / "grid.json"

    first = grid_search(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5]},
        metric="total_return",
        min_trades=1,
        cache_path=cache_path,
    )
    second = grid_search(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5]},
        metric="total_return",
        min_trades=2,
        cache_path=cache_path,
    )

    assert calls == 2
    assert first[0].score == pytest.approx(0.01)
    assert second[0].score == float("-inf")


def test_grid_cache_key_includes_slippage_model():
    base = _config(slippage_bps=5.0, slippage_model=FixedBpsSlippage(5.0))
    half_spread = _config(slippage_bps=5.0, slippage_model=HalfSpreadSlippage(5.0))

    base_key = _cache_key(
        base,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
    )
    half_spread_key = _cache_key(
        half_spread,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
    )

    assert base_key != half_spread_key


def test_metrics_calculations_and_edge_cases():
    equity = pd.Series([100.0, 110.0, 105.0, 120.0])
    trades = [_trade(10.0, 0.10), _trade(-5.0, -0.05)]

    assert maximum_drawdown(equity) == pytest.approx(-5 / 110)
    assert profit_factor(trades) == pytest.approx(2.0)
    assert win_rate(trades) == pytest.approx(0.5)
    assert expectancy(trades) == pytest.approx(0.025)
    assert sharpe_ratio(pd.Series([100.0])) == 0.0
    assert profit_factor([]) == 0.0
    assert win_rate([]) == 0.0


def test_all_losses_and_single_trade_metrics():
    losses = [_trade(-10.0, -0.10), _trade(-5.0, -0.05)]
    assert profit_factor(losses) == 0.0
    assert expectancy([_trade(3.0, 0.03)]) == pytest.approx(0.03)


def test_cli_help_includes_optimize_commands():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "optimize" in res.output
    res = runner.invoke(cli, ["optimize", "--help"])
    assert res.exit_code == 0
    assert "grid" in res.output
    assert "walk-forward" in res.output
    assert "validate" in res.output


def test_optimize_grid_offline_with_injected_fetcher():
    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    bars_b = make_bars(n=80, seed=22, open_base=50.0)
    spy = make_bars(n=80, seed=99, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA,BBB",
            "--start",
            bars_a.index[20].date().isoformat(),
            "--end",
            bars_a.index[60].date().isoformat(),
            "--entry",
            "close > sma(close, 3)",
            "--stop-loss",
            "none,0.05",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5,10",
            "--top-n",
            "2",
            "--workers",
            "1",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Grid Search Results" in res.output
