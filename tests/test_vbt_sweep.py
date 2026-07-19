"""Offline tests for vbt-sweep helpers (no vectorbt required)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.backtester.costs import IndiaDeliveryCosts, build_cost_model
from screener.backtester.vbt.sweep import vbt_fee_fraction
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
    assert "--cost-model" in res.output


def test_vbt_fee_fraction_flat_default_is_zero():
    # Default flat + commission_bps=0 → fees=0 (legacy hard-coded behaviour).
    assert vbt_fee_fraction("flat") == 0.0
    assert vbt_fee_fraction("flat", commission_bps=0.0) == 0.0


def test_vbt_fee_fraction_flat_is_per_side_bps():
    # Flat is symmetric: average(buy, sell) == side fraction == bps/10_000.
    assert vbt_fee_fraction("flat", commission_bps=10.0) == pytest.approx(0.001)
    assert vbt_fee_fraction("flat", commission_bps=25.0) == pytest.approx(0.0025)


def test_vbt_fee_fraction_india_averages_buy_and_sell():
    model = IndiaDeliveryCosts()
    notional = 100_000.0
    expected = 0.5 * (
        model.side_cost_fraction("buy", notional)
        + model.side_cost_fraction("sell", notional)
    )
    assert vbt_fee_fraction("india", notional=notional) == pytest.approx(expected)
    # India statutory stack is well above a flat-zero fee.
    assert vbt_fee_fraction("india") > vbt_fee_fraction("flat", commission_bps=0.0)


def test_vbt_fee_fraction_accepts_cost_model_instance():
    model = build_cost_model("us_vested")
    frac = vbt_fee_fraction(model, notional=100_000.0)
    buy = model.side_cost_fraction("buy", 100_000.0)
    sell = model.side_cost_fraction("sell", 100_000.0)
    assert frac == pytest.approx(0.5 * (buy + sell))
    assert frac > 0.0
