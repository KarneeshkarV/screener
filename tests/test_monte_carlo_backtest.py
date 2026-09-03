"""Equity-curve Monte Carlo engine and the ``backtest-monte-carlo`` command."""

from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import screener.backtester.monte_carlo_cli as monte_carlo_cli
import screener.backtester.tearsheet as tearsheet
from screener.backtester.metrics import result_view
from screener.backtester.optimization.monte_carlo import (
    equity_monte_carlo_metrics,
    simulate_equity_monte_carlo,
    simulate_equity_monte_carlo_paths,
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
    result = simulate_equity_monte_carlo(equity, iterations=10, block=500)

    assert result.bars == 0
    assert result.median_return == 0.0
    # No bar was ever drawn, so the reported block must not claim the request.
    assert result.block == 0


def test_empty_equity_curve_yields_an_empty_result():
    result, paths = simulate_equity_monte_carlo_paths(
        pd.Series(dtype=float), iterations=10, block=5
    )

    assert result.bars == 0
    assert result.initial_capital == 0.0
    assert result.median_return == 0.0
    assert paths.paths.shape == (0, 0)
    assert paths.terminal_returns.size == 0


def test_the_result_records_the_ruin_threshold_it_used():
    result = simulate_equity_monte_carlo(
        _equity(n=60), iterations=20, block=5, ruin_threshold=0.8
    )

    assert result.ruin_threshold == 0.8
    assert equity_monte_carlo_metrics(result)["mc_ruin_threshold"] == 0.8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": 0}, "iterations must be positive"),
        ({"block": 0}, "block must be positive"),
        ({"ruin_threshold": 0.0}, r"ruin_threshold must be in \(0, 1\]"),
        ({"ruin_threshold": 1.5}, r"ruin_threshold must be in \(0, 1\]"),
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


def test_paths_variant_matches_the_summary_only_run():
    equity = _equity(120, seed=5)
    summary = simulate_equity_monte_carlo(equity, iterations=100, block=10, seed=7)
    detailed, paths = simulate_equity_monte_carlo_paths(
        equity, iterations=100, block=10, seed=7, keep_paths=25
    )

    assert detailed == summary
    # Scalars cover every iteration; only the retained curves are capped.
    assert paths.terminal_returns.shape == (100,)
    assert paths.drawdowns.shape == (100,)
    assert paths.paths.shape == (25, summary.bars)
    assert paths.initial_capital == pytest.approx(float(equity.iloc[0]))


def test_retained_paths_are_capped_by_the_iteration_count():
    _, paths = simulate_equity_monte_carlo_paths(
        _equity(60, seed=6), iterations=10, block=5, keep_paths=500
    )

    assert paths.paths.shape[0] == 10


def test_retained_paths_reproduce_the_reported_outcomes():
    equity = _equity(90, seed=8)
    result, paths = simulate_equity_monte_carlo_paths(
        equity, iterations=30, block=6, seed=3, keep_paths=30
    )

    terminal = paths.paths[:, -1] / paths.initial_capital - 1.0
    assert terminal == pytest.approx(paths.terminal_returns, rel=1e-5)
    assert float(np.median(paths.terminal_returns)) == pytest.approx(
        result.median_return
    )


def test_negative_keep_paths_is_rejected():
    with pytest.raises(ValueError, match="keep_paths"):
        simulate_equity_monte_carlo_paths(_equity(30), keep_paths=-1)


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


def test_cli_weekend_only_window_reports_no_data(tmp_path):
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
            "--start",
            "2024-03-02",
            "--end",
            "2024-03-03",
            "--iterations",
            "10",
            "--report",
            str(tmp_path / "weekend.html"),
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 0, res.output
    assert "Monte Carlo: 10 paths, 0 bars, block 0" in res.output


def test_cli_report_has_a_monte_carlo_tab_with_the_paths(tmp_path):
    report = tmp_path / "mc.html"
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
            "40",
            "--block",
            "5",
            "--paths",
            "20",
            "--report",
            str(report),
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 0, res.output
    page = report.read_text()
    assert 'id="tab-montecarlo"' in page
    assert ">Monte Carlo<" in page
    for div in (
        "tearsheet-mc-paths",
        "tearsheet-mc-returns",
        "tearsheet-mc-drawdowns",
        "monte-carlo-percentile-table",
        "monte-carlo-summary-table",
    ):
        assert div in page, f"missing report section: {div}"


def test_rolling_report_has_no_monte_carlo_tab(tmp_path):
    report = tmp_path / "rolling.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
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
            "--report",
            str(report),
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 0, res.output
    assert 'id="tab-montecarlo"' not in report.read_text()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--iterations", "0"], "--iterations must be positive"),
        (["--block", "0"], "--block must be positive"),
        (["--paths", "-1"], "--paths must not be negative"),
        (["--ruin-threshold", "0"], "--ruin-threshold must be a fraction"),
        (["--ruin-threshold", "2"], "--ruin-threshold must be a fraction"),
    ],
)
def test_cli_rejects_bad_flags_before_running_the_backtest(flags, message):
    """A bad flag must be a usage error, not a traceback after a long run."""
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
            *flags,
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 2, res.output
    assert message in res.output
    # The rolling run never started, so no result line was printed.
    assert "Monte Carlo:" not in res.output


def test_cli_rejects_negative_seed_before_resolving_the_backtest(monkeypatch):
    resolve_backtest_run = Mock()
    monkeypatch.setattr(monte_carlo_cli, "resolve_backtest_run", resolve_backtest_run)

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
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
            "--seed",
            "-1",
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 2, res.output
    assert "--seed must not be negative" in res.output
    resolve_backtest_run.assert_not_called()


def test_flat_distribution_marker_labels_have_separate_positions(monkeypatch):
    captured = {}

    def capture_figure(fig, _div_id):
        captured["figure"] = fig
        return ""

    monkeypatch.setattr(tearsheet, "figure_html", capture_figure)

    tearsheet._distribution_html(
        np.zeros(20),
        "flat-distribution",
        realized=0.0,
        label="Return",
    )

    annotations = captured["figure"].layout.annotations
    assert {annotation.text for annotation in annotations} == {
        "p05",
        "median",
        "p95",
        "realized",
    }
    assert len({annotation.yshift for annotation in annotations}) == 4


def test_cli_help_lists_monte_carlo_and_shared_run_flags():
    res = CliRunner().invoke(cli, ["backtest-monte-carlo", "--help"])

    assert res.exit_code == 0
    for flag in (
        "--iterations",
        "--block",
        "--seed",
        "--ruin-threshold",
        "--paths",
        "--json",
    ):
        assert flag in res.output, f"missing flag in help: {flag}"
    # The command must expose the same run definition as backtest-rolling.
    for flag in ("--universe", "--rank-exit", "--sizing ", "--point-in-time"):
        assert flag in res.output, f"missing shared flag in help: {flag}"
    # Reporting-only rolling flags stay on backtest-rolling.
    assert "--compare-reinvestment" not in res.output
    assert "--dashboard" not in res.output
