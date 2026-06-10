"""Offline tests for vbt-sweep walk-forward mode (no vectorbt required)."""

from __future__ import annotations

import io
from typing import Any

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from main import cli
from screener.backtester.optimization.walk_forward import (
    generate_walk_forward_windows,
)
from screener.backtester.vbt_sweep import (
    _single_combo_sweep_kwargs,
    parse_walk_forward,
    run_walk_forward_sweep,
)
from tests.conftest import StubPriceFetcher, make_bars


def _make_close(periods: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(
        {
            "AAA": 100.0 + np.cumsum(rng.normal(0.05, 1.0, periods)),
            "BBB": 50.0 + np.cumsum(rng.normal(0.02, 0.8, periods)),
        },
        index=idx,
    )


def _stub_sweep(oos_factor: float = 1.0) -> Any:
    """Deterministic stand-in for run_parameter_sweep.

    Scores (fast + slow) / 100 so the (20, 100) combo always wins in-sample.
    Slices shorter than 150 bars (the OOS test windows) are scaled by
    ``oos_factor`` so walk-forward efficiency is predictable.
    """

    def sweep(
        close: pd.DataFrame,
        *,
        fast_values: list[int],
        slow_values: list[int],
        hold_values: list[int],
        indicators: list[str] | None = None,
        open_: pd.DataFrame | None = None,
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        initial_capital: float = 0.0,
        **_: Any,
    ) -> pd.DataFrame:
        factor = 1.0 if len(close) > 150 else oos_factor
        rows: list[dict[str, Any]] = []
        for fast in fast_values:
            for slow in slow_values:
                if slow <= fast:
                    continue
                for hold in hold_values:
                    score = (fast + slow) / 100.0 * factor
                    rows.append(
                        {
                            "indicator": (indicators or ["sma"])[0],
                            "fast": fast,
                            "slow": slow,
                            "hold": hold,
                            "sharpe": score,
                            "total_return": score / 10.0,
                            "calmar": score,
                            "max_drawdown": -0.1,
                            "win_rate": 0.5,
                            "trades": 4,
                        }
                    )
        return pd.DataFrame(rows)

    return sweep


def test_parse_walk_forward_valid():
    assert parse_walk_forward("12:3") == (12, 3)
    assert parse_walk_forward(" 6 : 2 ") == (6, 2)


@pytest.mark.parametrize("raw", ["12", "12:3:1", "a:b", "0:3", "12:-1"])
def test_parse_walk_forward_invalid(raw):
    with pytest.raises(click.UsageError):
        parse_walk_forward(raw)


def test_single_combo_sweep_kwargs_mappings():
    sma = _single_combo_sweep_kwargs("sma", 20, 100, 5)
    assert sma == {
        "indicators": ["sma"],
        "fast_values": [20],
        "slow_values": [100],
        "hold_values": [5],
    }
    breakout = _single_combo_sweep_kwargs("breakout", 55, 0, 0)
    assert breakout["indicators"] == ["breakout"]
    assert breakout["breakout_windows"] == [55]
    rsi = _single_combo_sweep_kwargs("rsi", 60, 0, 10)
    assert rsi["rsi_thresholds"] == [60]
    with pytest.raises(ValueError):
        _single_combo_sweep_kwargs("nope", 1, 2, 3)


def test_run_walk_forward_sweep_picks_winner_and_aggregates():
    close = _make_close()
    windows = generate_walk_forward_windows(
        close.index[0].date(),
        close.index[-1].date(),
        train_days=360,
        test_days=90,
    )
    assert len(windows) >= 2
    summary = run_walk_forward_sweep(
        close,
        windows=windows,
        metric="sharpe",
        grid={
            "fast_values": [10, 20],
            "slow_values": [50, 100],
            "hold_values": [0],
            "indicators": ["sma"],
        },
        sweep_fn=_stub_sweep(oos_factor=0.5),
    )
    df = summary.windows
    assert len(df) == len(windows)
    # In-sample winner is always (fast=20, slow=100) -> IS 1.2, OOS 0.6.
    assert (df["fast"] == 20).all()
    assert (df["slow"] == 100).all()
    np.testing.assert_allclose(df["is_score"].to_numpy(), 1.2)
    np.testing.assert_allclose(df["oos_score"].to_numpy(), 0.6)
    assert summary.aggregate_is_score == pytest.approx(1.2)
    assert summary.aggregate_oos_score == pytest.approx(0.6)
    assert summary.efficiency == pytest.approx(0.5)
    # Same params chosen in every window -> perfectly stable.
    assert summary.parameter_stability == pytest.approx(1.0)
    assert summary.aggregate_oos_metrics["sharpe"] == pytest.approx(0.6)
    assert summary.aggregate_oos_metrics["trades"] == pytest.approx(4 * len(windows))
    # Window dates line up with the generated schedule.
    assert list(df["train_start"]) == [w.train_start for w in windows]
    assert list(df["test_end"]) == [w.test_end for w in windows]


def _stub_env() -> StubPriceFetcher:
    bars_a = make_bars(start="2022-01-03", n=600, seed=1, open_base=100.0)
    bars_b = make_bars(start="2022-01-03", n=600, seed=2, open_base=50.0)
    spy = make_bars(start="2022-01-03", n=600, seed=3, open_base=400.0)
    return StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})


def test_cli_walk_forward_prints_summary(monkeypatch):
    monkeypatch.setattr(
        "screener.backtester.vbt_sweep.run_parameter_sweep", _stub_sweep(0.5)
    )
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-01-03",
            "--end",
            "2024-04-01",
            "--walk-forward",
            "12:3",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code == 0, res.output
    assert "Walk-Forward Sweep" in res.output
    assert "WF efficiency" in res.output
    assert "Parameter stability" in res.output
    assert "Aggregate OOS" in res.output


def test_cli_walk_forward_csv_emits_per_window_rows(monkeypatch):
    monkeypatch.setattr(
        "screener.backtester.vbt_sweep.run_parameter_sweep", _stub_sweep(0.5)
    )
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-01-03",
            "--end",
            "2024-04-01",
            "--walk-forward",
            "12:3",
            "--csv",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(io.StringIO(res.output))
    expected_cols = {
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "indicator",
        "fast",
        "slow",
        "hold",
        "is_score",
        "oos_score",
        "oos_total_return",
        "oos_trades",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 2
    # CLI defaults: --fast 10,20,50 / --slow 50,100,200 -> winner (50, 200).
    assert (df["fast"] == 50).all()
    assert (df["slow"] == 200).all()


def test_cli_walk_forward_rejects_bad_spec():
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA",
            "--walk-forward",
            "12",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "TRAIN_MONTHS:TEST_MONTHS" in res.output


def test_cli_walk_forward_rejects_window_overflow():
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-01-03",
            "--end",
            "2022-06-01",
            "--walk-forward",
            "12:3",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "do not fit" in res.output
