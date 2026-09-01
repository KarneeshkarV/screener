"""Equity-curve Monte Carlo engine and the ``backtest-monte-carlo`` command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.metrics import result_view
from screener.backtester.optimization.monte_carlo import (
    equity_monte_carlo_metrics,
    simulate_equity_monte_carlo,
)
from screener.cli import cli
from tests.conftest import StubPriceFetcher, make_bars


def _equity(n: int = 250, drift: float = 0.0005, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = 1.0 + rng.normal(drift, 0.01, n)
    return pd.Series(
        100_000.0 * np.cumprod(steps),
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


def test_same_seed_reproduces_the_same_distribution():
    kwargs = {"iterations": 200, "block": 10}
    first = simulate_equity_monte_carlo(_equity(), seed=3, **kwargs)
    second = simulate_equity_monte_carlo(_equity(), seed=3, **kwargs)
    third = simulate_equity_monte_carlo(_equity(), seed=4, **kwargs)

    assert first == second
    assert third.median_return != first.median_return


def test_percentiles_are_ordered_and_drawdowns_are_negative():
    result = simulate_equity_monte_carlo(_equity(), iterations=400, block=20)

    assert result.return_p05 < result.median_return < result.return_p95
    # Drawdowns are signed: the 5th percentile is deeper than the median, and
    # the worst path is deeper still.
    assert result.worst_drawdown <= result.drawdown_p05 <= result.median_drawdown <= 0.0
    assert 0.0 <= result.probability_of_profit <= 1.0
    assert result.bars == len(_equity()) - 1
    assert result.initial_capital == pytest.approx(_equity().iloc[0])


def test_block_length_is_capped_at_the_number_of_bars():
    result = simulate_equity_monte_carlo(_equity(n=30), iterations=50, block=500)

    assert result.block == 29  # 30 bars -> 29 returns


def test_blocks_preserve_the_autocorrelation_iid_draws_destroy():
    """Persistent losing streaks survive block draws; single-day draws break them.

    This is the whole reason the bootstrap is blocked. The curve below is built
    from AR(1) returns, so down days cluster into runs. Resampling one day at a
    time shuffles those runs apart and reports a shallower tail drawdown than
    the strategy can actually produce.
    """
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, 600)
    returns = np.empty(600)
    previous = 0.0
    for i, shock in enumerate(noise):
        previous = 0.8 * previous + shock
        returns[i] = previous
    equity = pd.Series(
        100_000.0 * np.cumprod(1.0 + returns),
        index=pd.date_range("2024-01-01", periods=600, freq="B"),
    )
    blocked = simulate_equity_monte_carlo(equity, iterations=400, block=40, seed=1)
    iid = simulate_equity_monte_carlo(equity, iterations=400, block=1, seed=1)

    assert blocked.drawdown_p05 < iid.drawdown_p05


def test_a_collapsing_curve_reports_ruin():
    equity = pd.Series(
        100_000.0 * np.cumprod(np.full(200, 0.97)),
        index=pd.date_range("2024-01-01", periods=200, freq="B"),
    )
    result = simulate_equity_monte_carlo(equity, iterations=100, block=5)

    assert result.risk_of_ruin == 1.0
    assert result.probability_of_profit == 0.0


def test_flat_curve_never_profits_and_never_ruins():
    equity = pd.Series(
        100_000.0, index=pd.date_range("2024-01-01", periods=50, freq="B")
    )
    result = simulate_equity_monte_carlo(equity, iterations=50, block=5)

    assert result.median_return == 0.0
    assert result.risk_of_ruin == 0.0


def test_single_point_curve_yields_an_empty_result():
    equity = pd.Series([100_000.0], index=pd.date_range("2024-01-01", periods=1))
    result = simulate_equity_monte_carlo(equity, iterations=10)

    assert result.bars == 0
    assert result.median_return == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": 0}, "iterations must be positive"),
        ({"block": 0}, "block must be positive"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_equity_monte_carlo(_equity(n=20), **kwargs)


def test_zero_starting_capital_is_rejected():
    equity = pd.Series([0.0, 1.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="must start above zero"):
        simulate_equity_monte_carlo(equity)


def test_metrics_render_with_declared_labels():
    metrics = equity_monte_carlo_metrics(
        simulate_equity_monte_carlo(_equity(n=60), iterations=20, block=5)
    )
    labels = {row.key: row.label for row in result_view(metrics)}

    assert labels["mc_median_return"] == "MC Median Return"
    assert labels["mc_risk_of_ruin"] == "MC Risk of Ruin"
    # Percent metrics must not fall through to the raw ratio format.
    kinds = {row.key: row.kind for row in result_view(metrics)}
    assert kinds["mc_worst_drawdown"] == "pct"
    assert kinds["mc_iterations"] == "int"


def _stub_fetcher() -> StubPriceFetcher:
    return StubPriceFetcher(
        {
            "AAA": make_bars(n=90, drift=0.004, seed=1),
            "BBB": make_bars(n=90, drift=0.002, seed=2),
            "SPY": make_bars(n=90, drift=0.001, seed=3),
        }
    )


def test_cli_reports_monte_carlo_rows(tmp_path):
    json_path = tmp_path / "mc.json"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-monte-carlo",
            "-m",
            "us",
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > 0",
            "--hold",
            "5",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
            "--iterations",
            "50",
            "--block",
            "5",
            "--json",
            str(json_path),
            "--report",
            str(tmp_path / "mc.html"),
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 0, res.output
    assert "Monte Carlo: 50 paths" in res.output
    assert "MC Median Return" in res.output
    payload = json.loads(json_path.read_text())
    assert payload["iterations"] == 50
    assert payload["block"] == 5


def test_cli_help_lists_monte_carlo_and_shared_run_flags():
    res = CliRunner().invoke(cli, ["backtest-monte-carlo", "--help"])

    assert res.exit_code == 0
    for flag in ("--iterations", "--block", "--seed", "--ruin-threshold", "--json"):
        assert flag in res.output, f"missing flag in help: {flag}"
    # The command must expose the same run definition as backtest-rolling.
    for flag in ("--universe", "--rank-exit", "--sizing ", "--point-in-time"):
        assert flag in res.output, f"missing shared flag in help: {flag}"
    # Reporting-only rolling flags stay on backtest-rolling.
    assert "--compare-reinvestment" not in res.output
    assert "--dashboard" not in res.output
