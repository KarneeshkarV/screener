"""Hand-computed tests for cash-hurdle and calendar risk metrics.

Expected values are re-derived with plain arithmetic in the comments.
A mismatch is a real defect, not a snapshot drift.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from screener.backtester.metrics import (
    _calendar_cagr,
    _cagr,
    _expected_shortfall,
    _max_drawdown_duration_days,
    _psr,
    _sortino,
    compute_metrics,
    equity_curve_sharpe,
    result_view,
    sharpe_moments_from_returns,
)
from screener.backtester.models import BacktestConfig
from screener.cli import cli


def _flat_config(**updates: object) -> BacktestConfig:
    payload = {
        "market": "india",
        "as_of": date(2024, 1, 31),
        "benchmark": "^NSEI",
        "tickers": ("RELIANCE",),
        "entry_expr": "close > 0",
        "exit_expr": None,
        "hold": 21,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "top": 10,
        "initial_capital": 100_000.0,
    }
    payload.update(updates)
    return BacktestConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_risk_free_rate_defaults_to_explicit_zero():
    assert _flat_config().risk_free_rate == 0.0


def test_risk_free_rate_rejects_negative_and_non_finite():
    with pytest.raises(ValidationError):
        _flat_config(risk_free_rate=-0.01)
    with pytest.raises(ValidationError):
        _flat_config(risk_free_rate=float("nan"))
    with pytest.raises(ValidationError):
        _flat_config(risk_free_rate=float("inf"))


# ---------------------------------------------------------------------------
# Excess-return Sharpe / Sortino (rf hurdle)
# ---------------------------------------------------------------------------


def test_equity_curve_sharpe_with_six_percent_hurdle_is_hand_computed():
    """rf=0.06 annual -> per-day hurdle 0.06/252.

    returns = [0.01, 0.02, -0.01, 0.00, 0.03]
    excess  = r - 0.06/252
    mean_e  = 0.01 - 0.06/252
    std is unchanged under a constant shift, so
    sharpe = mean_e / std_pop * sqrt(252)
    """
    returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03])
    rf = 0.06
    mean_e = 0.01 - rf / 252
    std_pop = math.sqrt(2e-4)  # same as rf=0 golden: sqrt(1e-3/5)
    expected = mean_e / std_pop * math.sqrt(252)

    assert abs(equity_curve_sharpe(returns, rf=rf) - expected) < 1e-9
    # Zero hurdle stays the legacy path.
    assert abs(equity_curve_sharpe(returns) - (0.01 / std_pop * math.sqrt(252))) < 1e-9


def test_sortino_with_hurdle_uses_excess_target_downside():
    """rf=0.06; target downside is RMS of min(excess, 0) over all N."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    rf = 0.06
    per = rf / 252
    excess = returns - per
    mean_e = float(excess.mean())
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    expected = mean_e / downside * math.sqrt(252)

    assert abs(_sortino(returns, rf=rf) - expected) < 1e-9


def test_compute_metrics_zero_hurdle_matches_legacy_sharpe_sortino():
    """Default rf=0 must keep the established zero-hurdle Sharpe/Sortino."""
    idx = pd.date_range("2022-01-03", periods=6, freq="B")
    equity = pd.Series([100.0, 101.0, 103.0, 102.0, 102.0, 105.0], index=idx)
    daily = equity.pct_change().dropna()
    metrics = compute_metrics(equity, pd.Series(dtype=float), [], slot_count=1)

    assert metrics["risk_free_rate"] == 0.0
    assert metrics["sharpe"] == equity_curve_sharpe(daily)
    assert metrics["sortino"] == _sortino(daily)
    assert metrics["psr"] == _psr(daily)


def test_compute_metrics_explicit_hurdle_lowers_smooth_low_return_sharpe():
    """A smooth ~5% path looks fine at rf=0 and weaker at rf=0.06."""
    # Constant daily growth that compounds to about 5% over 252 bars.
    daily_r = (1.05) ** (1 / 252) - 1
    values = 100_000.0 * np.cumprod(
        np.concatenate([[1.0], np.full(252, 1.0 + daily_r)])
    )
    equity = pd.Series(
        values, index=pd.date_range("2022-01-03", periods=len(values), freq="B")
    )
    zero = compute_metrics(equity, pd.Series(dtype=float), [], slot_count=1)
    hurdle = compute_metrics(
        equity,
        pd.Series(dtype=float),
        [],
        slot_count=1,
        risk_free_rate=0.06,
    )

    assert zero["sharpe"] > 1.0
    assert hurdle["sharpe"] < zero["sharpe"]
    assert hurdle["sortino"] < zero["sortino"]
    assert hurdle["risk_free_rate"] == 0.06
    # Moments used by PSR/DSR follow the same excess Sharpe.
    moments = sharpe_moments_from_returns(equity.pct_change().dropna(), rf=0.06)
    assert moments is not None
    assert abs(moments.sharpe_annual - hurdle["sharpe"]) < 1e-12


# ---------------------------------------------------------------------------
# Calendar CAGR, drawdown duration, expected shortfall
# ---------------------------------------------------------------------------


def test_calendar_cagr_uses_wall_clock_years_not_252_bars():
    """Two points spanning 365.25 days: (110/100)^1 - 1 = 0.10 exactly.

    Bar-count CAGR with periods_per_year=252 on a 2-point curve treats the
    horizon as 1/252 years and therefore explodes; calendar CAGR stays 10%.
    """
    t0 = pd.Timestamp("2021-01-01")
    t1 = t0 + pd.Timedelta(days=365.25)
    equity = pd.Series([100.0, 110.0], index=pd.DatetimeIndex([t0, t1]))

    assert abs(_calendar_cagr(equity) - 0.10) < 1e-12
    # Established 252-bar CAGR is unchanged and intentionally different.
    assert _cagr(equity, periods_per_year=252) != pytest.approx(0.10)


def test_calendar_cagr_without_datetime_index_is_zero():
    """Unit fixtures often use a RangeIndex; do not invent wall-clock years."""
    equity = pd.Series([100.0, 110.0, 120.0])
    assert _calendar_cagr(equity) == 0.0


def test_max_drawdown_duration_days_peak_to_recovery():
    """Peak on day 0, trough day 2, recovery on day 5 -> 5 calendar days.

    Dates: Mon 2024-01-01 = 100 (peak), Tue 99, Wed 95, Thu 97, Fri 98,
    Mon 2024-01-08 = 100 (recovery). Duration = 7 calendar days.
    """
    idx = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ]
    )
    equity = pd.Series([100.0, 99.0, 95.0, 97.0, 98.0, 100.0], index=idx)
    assert _max_drawdown_duration_days(equity) == 7.0


def test_max_drawdown_duration_open_at_end_runs_to_last_bar():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05"])
    equity = pd.Series([100.0, 90.0, 95.0], index=idx)
    assert _max_drawdown_duration_days(equity) == 4.0


def test_expected_shortfall_95_mean_of_left_tail():
    """Ten sorted daily returns; 5% tail is the lowest observation.

    For N=10 and alpha=0.95, quantile(0.05) is between the 1st and 2nd order
    stats under linear interpolation; values <= that cutoff form the ES set.
    Hand check with a uniform ladder:
        r = [-0.05, -0.04, ..., 0.04]
        q05 = -0.05 + 0.05*(0.05/0.111...) via pandas; assert against direct mean
        of the filtered tail.
    """
    returns = pd.Series([-0.05 + 0.01 * i for i in range(10)])
    cutoff = float(returns.quantile(0.05))
    expected = float(returns[returns <= cutoff].mean())
    assert abs(_expected_shortfall(returns, alpha=0.95) - expected) < 1e-12


def test_compute_metrics_emits_new_keys_and_slot_occupancy_label():
    idx = pd.date_range("2022-01-03", periods=40, freq="B")
    equity = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    metrics = compute_metrics(equity, pd.Series(dtype=float), [], slot_count=1)

    assert "calendar_cagr" in metrics
    assert "max_drawdown_duration_days" in metrics
    assert "expected_shortfall_95" in metrics
    assert "exposure" in metrics  # serialized key preserved
    labels = {row.key: row.label for row in result_view(metrics)}
    assert labels["exposure"] == "Avg Slot Occupancy"
    assert labels["sharpe"] == "Sharpe (rf hurdle)"
    assert labels["sortino"] == "Sortino (rf hurdle)"
    assert labels["risk_free_rate"] == "Risk-Free Hurdle (ann.)"
    assert labels["calendar_cagr"] == "Calendar CAGR"


# ---------------------------------------------------------------------------
# CLI / config propagation
# ---------------------------------------------------------------------------


def test_help_lists_risk_free_rate_on_backtest_and_optimize():
    runner = CliRunner()
    for argv in (
        ["backtest-rolling", "--help"],
        ["backtest-historical", "--help"],
        ["optimize", "grid", "--help"],
        ["optimize", "walk-forward", "--help"],
    ):
        result = runner.invoke(cli, argv)
        assert result.exit_code == 0, result.output
        assert "--risk-free-rate" in result.output
        assert "not cash interest" in result.output.lower() or "hurdle" in result.output


def test_yaml_config_and_flag_propagate_risk_free_rate(tmp_path, monkeypatch):
    from screener.backtester import historical as historical_cli
    from screener.backtester.models import BacktestResult

    captured: dict[str, BacktestConfig] = {}

    def fake_run_backtest(cfg, fetcher):
        captured["cfg"] = cfg
        equity = pd.Series([cfg.initial_capital], index=pd.to_datetime([cfg.as_of]))
        metrics = compute_metrics(
            equity,
            equity,
            [],
            slot_count=max(cfg.top, 1),
            risk_free_rate=float(cfg.risk_free_rate),
        )
        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=equity,
            metrics=metrics,
        )

    monkeypatch.setattr(historical_cli, "run_backtest", fake_run_backtest)

    path = tmp_path / "screener.yaml"
    path.write_text(
        """
backtest-historical:
  market: us
  as_of: "2024-01-31"
  tickers: AAA
  entry_expr: close > 0
  hold: 5
  top: 1
  risk_free_rate: 0.04
"""
    )
    result = CliRunner().invoke(cli, ["--config", str(path), "backtest-historical"])
    assert result.exit_code == 0, result.output
    assert captured["cfg"].risk_free_rate == 0.04

    result = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--as-of",
            "2024-01-31",
            "--tickers",
            "AAA",
            "--entry",
            "close > 0",
            "--hold",
            "5",
            "--top",
            "1",
            "--risk-free-rate",
            "0.06",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["cfg"].risk_free_rate == 0.06
    assert "Risk-Free Hurdle" in result.output


def test_optimize_grid_propagates_default_and_explicit_risk_free_rate(
    monkeypatch,
):
    from screener.backtester.optimization import cli as optimize_cli

    captured: dict[str, object] = {}

    def fake_grid_search(cfg, fetcher, parameter_grid, **kwargs):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(optimize_cli, "grid_search", fake_grid_search)
    argv = [
        "optimize",
        "grid",
        "--tickers",
        "AAA,BBB",
        "--entry",
        "close > sma(close, 3)",
        "--hold",
        "5",
    ]
    assert CliRunner().invoke(cli, argv).exit_code == 0
    assert captured["cfg"].risk_free_rate == 0.0  # type: ignore[union-attr]

    assert CliRunner().invoke(cli, [*argv, "--risk-free-rate", "0.06"]).exit_code == 0
    assert captured["cfg"].risk_free_rate == 0.06  # type: ignore[union-attr]
