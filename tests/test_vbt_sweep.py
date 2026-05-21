"""Offline tests for vbt-sweep helpers (no vectorbt required)."""

from __future__ import annotations

import math

import pandas as pd
from click.testing import CliRunner

from main import cli
from screener.backtester.vbt_sweep import (
    iter_param_combos,
    parse_int_list,
    rank_results,
)


def test_iter_param_combos_skips_invalid_slow():
    combos = iter_param_combos([10, 20, 50], [50, 100, 200], [10, 20])
    assert len(combos) == 16
    assert all(slow > fast for fast, slow, _hold in combos)


def test_parse_int_list():
    assert parse_int_list("10, 20", name="fast") == [10, 20]


def test_rank_results_deprioritizes_non_finite():
    df = pd.DataFrame(
        {
            "fast": [1, 2],
            "slow": [10, 20],
            "hold": [0, 0],
            "sharpe": [float("inf"), 0.5],
            "total_return": [0.0, 0.1],
            "calmar": [0.0, 0.1],
            "max_drawdown": [0.0, -0.1],
            "win_rate": [0.0, 0.5],
            "trades": [0, 3],
        }
    )
    ranked = rank_results(df, "sharpe")
    assert math.isfinite(ranked.iloc[0]["sharpe"])


def test_vbt_sweep_help_documents_exploration_only():
    res = CliRunner().invoke(cli, ["vbt-sweep", "--help"])
    assert res.exit_code == 0
    assert "exploration" in res.output.lower()
    assert "backtest-rolling" in res.output
