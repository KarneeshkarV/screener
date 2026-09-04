"""Equity-curve Monte Carlo engine and the ``backtest-monte-carlo`` command."""

from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import screener.backtester.monte_carlo_cli as monte_carlo_cli
import screener.backtester.optimization.monte_carlo as monte_carlo
import screener.backtester.tearsheet as tearsheet
from screener.backtester.metrics import result_view
from screener.backtester.optimization.monte_carlo import (
    _BAND_PERCENTILES,
    equity_monte_carlo_metrics,
    simulate_equity_monte_carlo,
    simulate_equity_monte_carlo_paths,
    validate_equity_monte_carlo_flags,
)
from screener.backtester.optimization.reporting import write_json_report
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


def test_a_block_as_long_as_the_series_is_rejected():
    """A circular block that spans the curve rotates it, and a rotation has the
    same product, so every path would report the identical terminal return.
    This used to be capped silently and published as a p05/p95 spread.
    """
    with pytest.raises(ValueError, match="block must be shorter than the 29"):
        simulate_equity_monte_carlo(_equity(n=30), iterations=50, block=500)


def test_a_block_one_bar_short_of_the_series_is_allowed():
    result = simulate_equity_monte_carlo(_equity(n=30), iterations=50, block=28)

    assert result.block == 28
    assert result.bars == 29


def test_the_reported_block_is_the_one_that_was_requested():
    result = simulate_equity_monte_carlo(_equity(n=250), iterations=50, block=20)

    assert result.block == 20


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
        ({"seed": -1}, "seed must not be negative"),
        ({"ruin_threshold": 0.0}, r"ruin_threshold must be a fraction"),
        ({"ruin_threshold": 1.5}, r"ruin_threshold must be a fraction"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_equity_monte_carlo(_equity(n=20), **{"block": 5, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"iterations": 0}, "iterations"),
        ({"block": 0}, "block"),
        ({"seed": -1}, "seed"),
        ({"keep_paths": -1}, "keep_paths"),
        ({"ruin_threshold": 0.0}, "ruin_threshold"),
    ],
)
def test_every_flag_message_names_the_field_it_rejects(kwargs, field):
    """The CLI renames the leading field to the flag that carries it, so a
    message that does not start with its field would reach the user in the
    engine's vocabulary instead of the command's.
    """
    valid = {
        "iterations": 10,
        "block": 5,
        "seed": 1,
        "keep_paths": 0,
        "ruin_threshold": 0.5,
    }
    with pytest.raises(ValueError) as caught:
        validate_equity_monte_carlo_flags(**{**valid, **kwargs})

    assert str(caught.value).startswith(field)


def test_the_engine_rejects_a_negative_seed():
    with pytest.raises(ValueError, match="seed must not be negative"):
        simulate_equity_monte_carlo_paths(_equity(n=30), block=5, seed=-1)


def test_zero_starting_capital_is_rejected():
    equity = pd.Series([0.0, 1.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="positive at every bar: 0.0 at position 0"):
        simulate_equity_monte_carlo(equity)


def test_equity_touching_zero_mid_curve_is_rejected_by_position():
    """A zero anywhere makes the next bar return -1 and the one after it inf,
    and ``cumprod`` turns that into NaN metrics the table renders as "-".
    """
    values = _equity(n=250).to_numpy().copy()
    values[100] = 0.0
    equity = pd.Series(values, index=_equity(n=250).index)

    with pytest.raises(ValueError, match="positive at every bar: 0.0 at position 100"):
        simulate_equity_monte_carlo(equity, iterations=50, block=5)


def test_a_hole_in_the_curve_is_rejected_by_position():
    """``dropna`` used to swallow this, leaving ``bars`` one short of the real
    curve so the fan chart and the realized line ended on different bars.
    """
    values = _equity(n=250).to_numpy().copy()
    values[100] = np.nan
    equity = pd.Series(values, index=_equity(n=250).index)

    with pytest.raises(ValueError, match="finite at every bar: nan at position 100"):
        simulate_equity_monte_carlo(equity, iterations=50, block=5)


def test_bars_always_spans_the_whole_curve():
    result = simulate_equity_monte_carlo(_equity(n=250), iterations=20, block=5)

    assert result.bars == 250 - 1


def test_a_two_point_curve_has_no_distribution_to_report():
    """One bar return resamples to itself however it is blocked."""
    equity = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
    result = simulate_equity_monte_carlo(equity, iterations=10, block=1)

    assert result.bars == 0
    assert result.block == 0
    assert result.median_return == 0.0


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


def test_bands_cover_every_iteration_not_the_retained_sample():
    """The whole point of computing the bands here. Keeping one path used to
    collapse p05, median and p95 onto it while the summary table still
    reported a spread, so the chart contradicted the table beside it.
    """
    equity = _equity(120, seed=5)
    result, paths = simulate_equity_monte_carlo_paths(
        equity, iterations=500, block=10, seed=7, keep_paths=1
    )

    assert paths.band_iterations == 500
    assert paths.paths.shape[0] == 1
    terminal = paths.bands[:, -1] / paths.initial_capital - 1.0
    # float32 band buffer against float64 summary percentiles.
    assert terminal[0] == pytest.approx(result.return_p05, rel=1e-6)
    assert terminal[1] == pytest.approx(result.median_return, rel=1e-6)
    assert terminal[2] == pytest.approx(result.return_p95, rel=1e-6)


def test_bands_start_at_the_realized_capital_and_span_the_curve():
    equity = _equity(120, seed=5)
    result, paths = simulate_equity_monte_carlo_paths(
        equity, iterations=50, block=10, seed=7, keep_paths=10
    )

    assert paths.band_percentiles == _BAND_PERCENTILES
    # One column per bar of the realized curve, so the two share an x axis.
    assert paths.bands.shape == (len(_BAND_PERCENTILES), result.bars + 1)
    assert paths.bands.shape[1] == len(equity)
    assert paths.bands[:, 0].tolist() == pytest.approx(
        [paths.initial_capital] * len(_BAND_PERCENTILES)
    )


def test_bands_are_ordered_at_every_bar():
    _, paths = simulate_equity_monte_carlo_paths(
        _equity(90, seed=8), iterations=200, block=6, seed=3, keep_paths=5
    )

    assert (paths.bands[0] <= paths.bands[1]).all()
    assert (paths.bands[1] <= paths.bands[2]).all()


def test_bands_stride_when_the_buffer_would_exceed_its_budget(monkeypatch):
    """Above the budget the bands sample the iterations rather than allocate an
    unbounded buffer, and say how many they used.
    """
    monkeypatch.setattr(monte_carlo, "_BAND_CELL_BUDGET", 300)

    _, paths = simulate_equity_monte_carlo_paths(
        _equity(60, seed=6), iterations=100, block=5, keep_paths=5
    )

    # 59 returns, so the budget allows 5 rows: every 20th iteration.
    assert paths.band_iterations == 5
    assert paths.bands.shape == (3, 60)


def test_the_empty_result_still_carries_a_band_row_per_percentile():
    _, paths = simulate_equity_monte_carlo_paths(
        pd.Series(dtype=float), iterations=10, block=5
    )

    assert paths.band_percentiles == _BAND_PERCENTILES
    assert paths.bands.shape == (len(_BAND_PERCENTILES), 0)
    assert paths.band_iterations == 0


def test_the_json_report_holds_no_token_a_strict_parser_rejects(tmp_path):
    """``NaN`` and ``Infinity`` are not JSON. Written raw, as they used to be,
    the payload loads back only in Python.
    """
    path = tmp_path / "mc.json"
    write_json_report(
        {
            "median_return": float("nan"),
            "calmar": float("inf"),
            "worst": float("-inf"),
            "return_p05": -0.24,
        },
        path,
    )
    text = path.read_text()

    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(
        text, parse_constant=lambda token: pytest.fail(f"non-JSON token: {token}")
    )
    assert payload == {
        "median_return": None,
        "calmar": "inf",
        "worst": "-inf",
        "return_p05": -0.24,
    }


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
    assert "Monte Carlo: 50 iterations" in res.output
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
    assert "Monte Carlo: 10 iterations, 0 bars, block 0" in res.output


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
        (["--seed", "-1"], "--seed must not be negative"),
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


def test_cli_reports_an_oversized_block_as_an_error_not_a_traceback(tmp_path):
    """The block can only be checked against the bars the run produced, so this
    one surfaces after the backtest. It must still name the flag to change.
    """
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
            "20",
            "--block",
            "10000",
        ],
        obj=_stub_fetcher(),
    )

    assert res.exit_code == 1, res.output
    assert "--block must be shorter than" in res.output
    assert "Traceback" not in res.output


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


def _fan_paths(keep_paths: int, *, bars: int = 250, iterations: int = 2000):
    equity = _equity(n=bars)
    return equity, simulate_equity_monte_carlo_paths(
        equity, iterations=iterations, block=20, seed=42, keep_paths=keep_paths
    )


@pytest.mark.parametrize("keep_paths", [1000, 1, 0])
def test_the_fan_bands_match_the_summary_percentiles_whatever_is_retained(keep_paths):
    """The chart and the table on the same page must not disagree.

    Bands taken over the retained sample answer a different question from the
    MC Return rows, and at ``--paths 1`` they collapsed onto one arbitrary path
    while the table still reported a spread.
    """
    _, (result, paths) = _fan_paths(keep_paths)

    terminal = paths.bands[:, -1] / paths.initial_capital - 1.0

    assert paths.band_percentiles == (5, 50, 95)
    # float32 in the band buffer, so exact equality is not on offer; the report
    # renders two decimal places of a percentage, which is 1e-4.
    assert terminal == pytest.approx(
        [result.return_p05, result.median_return, result.return_p95], abs=1e-6
    )
    assert paths.paths.shape[0] == min(keep_paths, 2000)


def test_the_fan_chart_draws_its_bands_even_when_no_paths_are_retained():
    equity, (_, paths) = _fan_paths(0)

    html = tearsheet._fan_chart_html(equity, paths)

    assert "Simulated p05" in html
    assert "Simulated median" in html
    assert "Simulated p95" in html
    assert "Realized run" in html
    assert "simulated paths" not in html


def test_the_fan_chart_size_stays_bounded_as_bars_grow():
    """The fan is a texture, so its cost must not scale with the bar count.

    One trace per drawn bar put 4.4 MB of equity levels into a 10-year daily
    report, on top of the plotly bundle, on every run.
    """
    sizes = []
    for bars in (600, 2520):
        equity, (_, paths) = _fan_paths(1000, bars=bars, iterations=1000)
        sizes.append(len(tearsheet._fan_chart_html(equity, paths)))

    short, long = sizes
    assert long < 1_200_000, f"fan chart grew to {long:,} bytes at 2520 bars"
    # 4.2x the bars must not cost 4.2x the bytes.
    assert long < short * 1.5


def test_the_fan_ends_on_the_same_bar_as_the_realized_run():
    """Finding 5's misalignment must not come back through the downsampling."""
    equity, (_, paths) = _fan_paths(1000, bars=2520, iterations=200)
    drawn = paths.paths[:: max(1, paths.paths.shape[0] // tearsheet._FAN_LINES)]
    bars = drawn.shape[1] + 1
    points = min(tearsheet._FAN_BAR_POINTS, bars)
    columns = np.round(np.linspace(0, bars - 1, points)).astype(int)

    assert columns[0] == 0
    assert columns[-1] == bars - 1
    assert bars == len(equity)


def test_the_realized_return_row_reads_the_metric_the_overview_tab_prints():
    """One definition of the number, so the two tabs cannot drift apart."""
    equity, (mc, paths) = _fan_paths(50)
    result = Mock()
    result.equity_curve = equity
    result.metrics = {"total_return": 0.1234, "max_drawdown": -0.0567}

    html = tearsheet._monte_carlo_sections(result, mc, paths)

    assert "<th>Realized return</th><td>+12.34%</td>" in html
    assert "<th>Realized max drawdown</th><td>-5.67%</td>" in html
