"""Regression tests for vectorized vbt parameter sweep (requires vectorbt extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from screener.backtester.vbt_sweep import (
    iter_param_combos,
    rank_results,
    run_combo_backtest,
    run_parameter_sweep,
    _require_vectorbt,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "vbt_sweep_aapl_msft_nvda.csv"


def _synthetic_close_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    idx = pd.bdate_range("2019-01-01", periods=260)
    symbols = ["AAPL", "MSFT", "NVDA"]
    close = pd.DataFrame(
        {
            sym: 100.0 + i * 10 + np.cumsum(np.random.randn(len(idx)))
            for i, sym in enumerate(symbols)
        },
        index=idx,
    )
    return close, close * 0.998


def test_run_parameter_sweep_matches_frozen_csv() -> None:
    close, open_ = _synthetic_close_panel()
    actual = run_parameter_sweep(
        close,
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[0],
        open_=open_,
    )
    expected = pd.read_csv(FIXTURE_CSV)
    metric_cols = [
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
    ]
    for col in metric_cols:
        np.testing.assert_allclose(
            actual[col].to_numpy(),
            expected[col].to_numpy(),
            rtol=0,
            atol=1e-9,
        )
    pd.testing.assert_frame_equal(
        actual[["fast", "slow", "hold", "trades"]].astype(int).reset_index(drop=True),
        expected[["fast", "slow", "hold", "trades"]].astype(int).reset_index(drop=True),
    )


def test_run_parameter_sweep_matches_per_combo_loop() -> None:
    close, open_ = _synthetic_close_panel()
    fast_values, slow_values, hold_values = [10, 20], [50, 100], [0, 10]
    vbt = _require_vectorbt()
    loop_df = pd.DataFrame(
        [
            run_combo_backtest(
                close, fast, slow, hold, vbt=vbt, open_=open_, initial_capital=100_000
            )
            for fast, slow, hold in iter_param_combos(
                fast_values, slow_values, hold_values
            )
        ]
    )
    vec_df = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        open_=open_,
    )
    metric_cols = [
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    ]
    np.testing.assert_allclose(
        loop_df[metric_cols].to_numpy(),
        vec_df[metric_cols].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def test_rank_results_unchanged_on_vectorized_output() -> None:
    close, open_ = _synthetic_close_panel()
    results = run_parameter_sweep(
        close,
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[0],
        open_=open_,
    )
    ranked = rank_results(results, "sharpe")
    assert ranked.iloc[0]["fast"] == 20
    assert ranked.iloc[0]["slow"] == 100
