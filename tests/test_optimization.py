from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.metrics import compute_metrics, deflated_sharpe
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization import grid as grid_module
from screener.backtester.optimization.grid import (
    _cache_key,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.metrics import optimization_metrics
from screener.backtester.optimization.monte_carlo import simulate_monte_carlo
from screener.backtester.optimization.walk_forward import generate_walk_forward_windows
from screener.backtester.slippage import FixedBpsSlippage, HalfSpreadSlippage
from screener.cli import cli
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


def test_walk_forward_rejects_zero_step_and_overlap():
    with pytest.raises(ValueError, match="step_days must be positive"):
        generate_walk_forward_windows(
            date(2024, 1, 1),
            date(2024, 6, 1),
            train_days=30,
            test_days=20,
            step_days=0,
        )
    with pytest.raises(ValueError, match="overlapping"):
        generate_walk_forward_windows(
            date(2024, 1, 1),
            date(2024, 8, 1),
            train_days=30,
            test_days=60,
            step_days=30,
        )


def test_monte_carlo_reproducible_with_same_seed():
    trades = [_trade(10.0, 0.10), _trade(-5.0, -0.05), _trade(3.0, 0.03)]
    a = simulate_monte_carlo(trades, iterations=100, seed=7)
    b = simulate_monte_carlo(trades, iterations=100, seed=7)
    assert a == b


@pytest.mark.parametrize("ruin_threshold", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_trade_monte_carlo_rejects_invalid_ruin_threshold(ruin_threshold):
    with pytest.raises(ValueError, match="ruin_threshold must be a fraction"):
        simulate_monte_carlo(
            [_trade(10.0, 0.10)], ruin_threshold=ruin_threshold, iterations=10
        )


def test_trade_monte_carlo_accepts_ruin_threshold_one():
    loss_res = simulate_monte_carlo(
        [_trade(-10.0, -0.10)], ruin_threshold=1.0, iterations=20, seed=1
    )
    assert loss_res.risk_of_ruin == 1.0

    gain_res = simulate_monte_carlo(
        [_trade(10.0, 0.10)], ruin_threshold=1.0, iterations=20, seed=1
    )
    assert gain_res.risk_of_ruin == 0.0


def test_monte_carlo_bootstrap_has_terminal_return_distribution():
    trades = [
        _trade(10.0, 0.10),
        _trade(-5.0, -0.05),
        _trade(3.0, 0.03),
        _trade(-2.0, -0.02),
    ]
    result = simulate_monte_carlo(trades, iterations=1000, seed=7)
    assert result.return_p05 < result.return_p95


@pytest.mark.parametrize("chunk_cells", [1, 7, 1_000_000])
def test_monte_carlo_chunked_draw_matches_per_iteration_draw(monkeypatch, chunk_cells):
    from screener.backtester.optimization import monte_carlo

    monkeypatch.setattr(monte_carlo, "_CHUNK_CELLS", chunk_cells)
    returns = [0.10, -0.05, 0.03, -0.02, -0.30]
    trades = [_trade(100.0 * r, r) for r in returns]

    result = simulate_monte_carlo(trades, iterations=23, seed=4, ruin_threshold=0.8)

    # The loop this replaced: one ``choice`` per iteration from one generator.
    rng = np.random.default_rng(4)
    terminal, drawdowns, ruined = [], [], 0
    for _ in range(23):
        path = 100_000.0 * np.cumprod(1.0 + rng.choice(returns, size=len(returns)))
        levels = np.concatenate(([100_000.0], path))
        peak = np.maximum.accumulate(levels)
        terminal.append(path[-1] / 100_000.0 - 1.0)
        drawdowns.append(((levels - peak) / peak).min())
        ruined += bool(path.min() <= 80_000.0)
    assert result.median_return == float(np.median(terminal))
    assert result.return_p05 == float(np.percentile(terminal, 5))
    assert result.worst_drawdown == float(np.min(drawdowns))
    assert result.drawdown_p05 == float(np.percentile(drawdowns, 5))
    assert result.risk_of_ruin == ruined / 23


@pytest.mark.parametrize("ruin_threshold", ["0", "-0.1", "1.5", "nan", "inf"])
def test_optimize_validate_rejects_ruin_threshold_before_reading_trades(
    tmp_path, monkeypatch, ruin_threshold
):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("not,a,valid,ledger\n")

    def unexpected_load(_path):
        raise AssertionError("trade file must not be loaded for an invalid threshold")

    monkeypatch.setattr(
        "screener.backtester.optimization.cli._load_trades", unexpected_load
    )
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "validate",
            "--trades",
            str(ledger),
            "--ruin-threshold",
            ruin_threshold,
        ],
    )

    assert res.exit_code == 2
    assert "--ruin-threshold must be a fraction" in res.output


def test_optimize_validate_reads_an_intraday_ledger(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
        "AAA,1,2024-03-04 14:30:00,2024-03-04 14:45:00,100,2024-03-04 15:45:00,"
        "101,time,1,100,101,1,0.01\n"
        "BBB,2,2024-03-04,2024-03-05,50,2024-03-06,49,stop,2,100,98,-2,-0.02\n"
    )

    res = CliRunner().invoke(
        cli, ["optimize", "validate", "--trades", str(ledger), "--iterations", "20"]
    )

    assert res.exit_code == 0, res.output
    assert "Monte Carlo Validation" in res.output


def test_monte_carlo_groups_partial_exit_tranches_into_single_position():
    """Partial-exit tranches sharing (ticker, open_seq, entry_date) resample as one position."""
    tranche_1 = Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 4),
        exit_price=110.0,
        exit_reason="target",
        shares=6.0,
        entry_cost=600.0,
        exit_value=660.0,
        pnl=60.0,
        return_pct=0.10,
        open_seq=1,
    )
    tranche_2 = Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 8),
        exit_price=95.0,
        exit_reason="time",
        shares=4.0,
        entry_cost=400.0,
        exit_value=380.0,
        pnl=-20.0,
        return_pct=-0.05,
        open_seq=1,
    )

    # Combined position has pnl = 40.0 on entry_cost = 1000.0 -> +4.0% total return.
    # Because sample_size is 1 position (not 2 tranches), every draw yields +0.04.
    result = simulate_monte_carlo([tranche_1, tranche_2], iterations=200, seed=7)
    assert result.median_return == pytest.approx(0.04)
    assert result.return_p05 == pytest.approx(0.04)
    assert result.return_p95 == pytest.approx(0.04)
    assert result.probability_of_profit == 1.0
    assert result.worst_drawdown == 0.0

    # A separate lot (distinct open_seq) stays a separate position.
    second_lot = Trade(
        ticker="AAA",
        rank=2,
        signal_date=date(2024, 1, 10),
        entry_date=date(2024, 1, 11),
        entry_price=100.0,
        exit_date=date(2024, 1, 15),
        exit_price=90.0,
        exit_reason="stop",
        shares=10.0,
        entry_cost=1000.0,
        exit_value=900.0,
        pnl=-100.0,
        return_pct=-0.10,
        open_seq=2,
    )
    two_pos = simulate_monte_carlo(
        [tranche_1, tranche_2, second_lot], iterations=500, seed=7
    )
    # Two positions (+4%, -10%): best 2-draw path is (1.04)^2 - 1 = 0.0816,
    # worst is (0.90)^2 - 1 = -0.19 (each occurs ~25% of draws, not 3-tranche compounding).
    assert two_pos.return_p95 == pytest.approx(1.04**2 - 1.0)
    assert two_pos.return_p05 == pytest.approx(0.90**2 - 1.0)
    assert two_pos.return_p05 < two_pos.return_p95


def test_monte_carlo_includes_dividend_income_from_total_pnl():
    """In splits_only/none mode, dividend-driven gains are reflected in MC returns."""
    div_winner = Trade(
        ticker="DIV",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 10),
        exit_price=98.0,
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=98.0,
        pnl=4.0,  # -2.0 capital + 6.0 cash dividend
        return_pct=-0.02,  # capital-only return is negative
        dividend_income=6.0,
        open_seq=1,
    )

    result = simulate_monte_carlo([div_winner], iterations=100, seed=3)
    assert result.median_return == pytest.approx(0.04)
    assert result.return_p05 == pytest.approx(0.04)
    assert result.return_p95 == pytest.approx(0.04)
    assert result.probability_of_profit == 1.0
    assert result.risk_of_ruin == 0.0


def test_monte_carlo_falls_back_to_return_pct_for_zero_entry_cost_fixture():
    synthetic = Trade(
        ticker="SYN",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=0.0,
        exit_date=date(2024, 1, 3),
        exit_price=0.0,
        exit_reason="time",
        shares=0.0,
        entry_cost=0.0,
        exit_value=0.0,
        pnl=0.0,
        return_pct=0.05,
    )
    result = simulate_monte_carlo([synthetic], iterations=50, seed=1)
    assert result.median_return == pytest.approx(0.05)
    assert result.probability_of_profit == 1.0


def test_monte_carlo_zero_cost_multi_tranche_uses_share_weighted_return():
    """Zero-cost positions with multiple tranches compute share-weighted return_pct."""
    tranche_1 = Trade(
        ticker="SYN",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=0.0,
        exit_date=date(2024, 1, 4),
        exit_price=0.0,
        exit_reason="target",
        shares=3.0,
        entry_cost=0.0,
        exit_value=0.0,
        pnl=0.0,
        return_pct=0.10,
        open_seq=1,
    )
    tranche_2 = Trade(
        ticker="SYN",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=0.0,
        exit_date=date(2024, 1, 6),
        exit_price=0.0,
        exit_reason="time",
        shares=1.0,
        entry_cost=0.0,
        exit_value=0.0,
        pnl=0.0,
        return_pct=-0.02,
        open_seq=1,
    )
    # Share-weighted return: (3 * 0.10 + 1 * (-0.02)) / 4 = 0.07 (not silent 0.0).
    result = simulate_monte_carlo([tranche_1, tranche_2], iterations=100, seed=5)
    assert result.median_return == pytest.approx(0.07)
    assert result.return_p05 == pytest.approx(0.07)
    assert result.return_p95 == pytest.approx(0.07)
    assert result.probability_of_profit == 1.0

    # When total_shares <= 0, fall back to simple mean of finite tranche return_pct.
    zero_share_1 = tranche_1.model_copy(update={"shares": 0.0})
    zero_share_2 = tranche_2.model_copy(update={"shares": 0.0})
    mean_result = simulate_monte_carlo(
        [zero_share_1, zero_share_2], iterations=100, seed=5
    )
    assert mean_result.median_return == pytest.approx(0.04)

    # Non-finite tranche return_pct on a zero-cost multi-tranche position is rejected.
    bad_tranche = tranche_2.model_copy(update={"return_pct": float("nan")})
    with pytest.raises(ValueError, match="trade returns must be finite"):
        simulate_monte_carlo([tranche_1, bad_tranche], iterations=10)


def test_monte_carlo_rejects_negative_entry_cost():
    bad_trade = _trade(10.0, 0.10).model_copy(update={"entry_cost": -100.0})
    with pytest.raises(ValueError, match="entry_cost must be non-negative"):
        simulate_monte_carlo([bad_trade], iterations=10)

    # Even if another tranche in the same position has a larger positive entry_cost.
    t1 = _trade(10.0, 0.10).model_copy(update={"entry_cost": -50.0, "open_seq": 1})
    t2 = _trade(20.0, 0.10).model_copy(update={"entry_cost": 200.0, "open_seq": 1})
    with pytest.raises(ValueError, match="entry_cost must be non-negative"):
        simulate_monte_carlo([t1, t2], iterations=10)


def test_monte_carlo_separates_walk_forward_folds_reusing_open_seq():
    """Walk-forward folds that reuse open_seq with distinct entry_date stay separate."""
    fold_1 = Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 10),
        exit_price=110.0,
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=110.0,
        pnl=10.0,
        return_pct=0.10,
        open_seq=1,
    )
    fold_2 = Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 4, 1),
        entry_date=date(2024, 4, 2),
        entry_price=100.0,
        exit_date=date(2024, 4, 10),
        exit_price=95.0,
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=95.0,
        pnl=-5.0,
        return_pct=-0.05,
        open_seq=1,
    )
    # Two distinct positions (+10%, -5%): 2-draw bootstrap has best (1.10^2 - 1) and
    # worst (0.95^2 - 1), rather than collapsing into a single +2.5% position.
    result = simulate_monte_carlo([fold_1, fold_2], iterations=500, seed=7)
    assert result.return_p95 == pytest.approx(1.10**2 - 1.0)
    assert result.return_p05 == pytest.approx(0.95**2 - 1.0)
    assert result.return_p05 < result.return_p95


def test_monte_carlo_absorbs_bankruptcy_on_returns_below_negative_one():
    """Returns < -1.0 absorb at 0 equity without sign-flip resurrection."""
    wipeouts = [_trade(-150.0, -1.5), _trade(-150.0, -1.5)]
    result = simulate_monte_carlo(wipeouts, iterations=200, seed=11)

    assert result.median_return == pytest.approx(-1.0)
    assert result.return_p05 == pytest.approx(-1.0)
    assert result.return_p95 == pytest.approx(-1.0)
    assert result.worst_drawdown == pytest.approx(-1.0)
    assert result.worst_drawdown >= -1.0
    assert result.probability_of_profit == 0.0
    assert result.risk_of_ruin == 1.0

    # Even mixed with a big winner, no path can exceed -100% drawdown.
    mixed = [_trade(-150.0, -1.5), _trade(200.0, 2.0)]
    mixed_result = simulate_monte_carlo(mixed, iterations=500, seed=11)
    assert mixed_result.worst_drawdown >= -1.0
    assert mixed_result.return_p05 >= -1.0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_monte_carlo_rejects_non_finite_trade_returns(bad_value):
    with pytest.raises(ValueError, match="trade returns must be finite"):
        simulate_monte_carlo([_trade(bad_value, bad_value)], iterations=10)

    # Also reject non-finite return_pct on a zero-cost synthetic fixture.
    zero_cost = Trade(
        ticker="BAD",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=0.0,
        exit_date=date(2024, 1, 3),
        exit_price=0.0,
        exit_reason="time",
        shares=0.0,
        entry_cost=0.0,
        exit_value=0.0,
        pnl=0.0,
        return_pct=bad_value,
    )
    with pytest.raises(ValueError, match="trade returns must be finite"):
        simulate_monte_carlo([zero_cost], iterations=10)


@pytest.mark.parametrize("bad_capital", [0.0, -1000.0, float("nan"), float("inf")])
def test_monte_carlo_rejects_invalid_initial_capital(bad_capital):
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        simulate_monte_carlo(
            [_trade(10.0, 0.10)], initial_capital=bad_capital, iterations=10
        )


def test_load_trades_preserves_dividend_income_and_open_seq(tmp_path):
    from screener.backtester.optimization.cli import _load_trades

    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct,dividend_income,open_seq\n"
        "AAA,1,2024-03-01,2024-03-04,100,2024-03-06,110,target,6,600,660,63,0.10,3.0,2\n"
        "AAA,1,2024-03-01,2024-03-04,100,2024-03-10,95,time,4,400,380,-18,-0.05,2.0,2\n"
    )
    trades = _load_trades(ledger)
    assert len(trades) == 2
    assert trades[0].dividend_income == pytest.approx(3.0)
    assert trades[0].open_seq == 2
    assert trades[1].dividend_income == pytest.approx(2.0)
    assert trades[1].open_seq == 2

    # Legacy CSV without dividend_income / open_seq defaults to 0.0 / 0.
    legacy = tmp_path / "legacy.csv"
    legacy.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
        "BBB,1,2024-03-01,2024-03-04,100,2024-03-06,105,time,1,100,105,5,0.05\n"
    )
    legacy_trades = _load_trades(legacy)
    assert legacy_trades[0].dividend_income == 0.0
    assert legacy_trades[0].open_seq == 0

    # End-to-end optimize validate groups the two tranches into one +4.5% position.
    out_json = tmp_path / "mc.json"
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "validate",
            "--trades",
            str(ledger),
            "--iterations",
            "50",
            "--json",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, res.output
    import json

    payload = json.loads(out_json.read_text())
    assert payload["median_return"] == pytest.approx(45.0 / 1000.0)
    assert payload["probability_of_profit"] == 1.0


def test_load_trades_handles_empty_cells_float_ints_and_legacy_warning(
    tmp_path, caplog
):
    import logging

    from screener.backtester.optimization.cli import _load_trades

    # Empty strings in dividend_income/open_seq default to 0.0/0, and pandas-formatted
    # float integer strings like "2.0" parse cleanly as integers.
    ledger = tmp_path / "pandas_ledger.csv"
    ledger.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct,dividend_income,open_seq\n"
        "AAA,1.0,2024-03-01,2024-03-04,100,2024-03-06,110,target,6,600,660,60,0.10,,2.0\n"
        "BBB,2,2024-03-01,2024-03-04,100,2024-03-06,105,time,1,100,105,5,0.05,,\n"
    )
    with caplog.at_level(
        logging.WARNING, logger="screener.backtester.optimization.cli"
    ):
        trades = _load_trades(ledger)
    assert len(trades) == 2
    assert trades[0].rank == 1
    assert trades[0].dividend_income == 0.0
    assert trades[0].open_seq == 2
    assert trades[1].dividend_income == 0.0
    assert trades[1].open_seq == 0
    assert not any("open_seq" in rec.message for rec in caplog.records)

    # Non-integer float strings and non-finite strings in open_seq are rejected.
    for bad_seq in ("2.5", "nan", "inf"):
        bad_csv = tmp_path / f"bad_{bad_seq}.csv"
        bad_csv.write_text(
            "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
            "exit_reason,shares,entry_cost,exit_value,pnl,return_pct,open_seq\n"
            f"AAA,1,2024-03-01,2024-03-04,100,2024-03-06,105,time,1,100,105,5,0.05,{bad_seq}\n"
        )
        with pytest.raises(ValueError, match="Expected integer value"):
            _load_trades(bad_csv)

    # Legacy CSV without open_seq column logs a warning when rows are present.
    legacy = tmp_path / "legacy_no_open_seq.csv"
    legacy.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
        "AAA,1,2024-03-01,2024-03-04,100,2024-03-06,105,time,1,100,105,5,0.05\n"
    )
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="screener.backtester.optimization.cli"
    ):
        _load_trades(legacy)
    assert any(
        "open_seq" in rec.message and "partial-exit" in rec.message
        for rec in caplog.records
    )

    # Empty legacy CSV (headers only) does not warn.
    empty_legacy = tmp_path / "empty_legacy.csv"
    empty_legacy.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
    )
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="screener.backtester.optimization.cli"
    ):
        assert _load_trades(empty_legacy) == []
    assert not any("open_seq" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    ("flags", "expected_message"),
    [
        (["--iterations", "0"], "--iterations must be positive"),
        (["--iterations", "-10"], "--iterations must be positive"),
        (["--seed", "-1"], "--seed must not be negative"),
        (["--initial-capital", "0"], "--initial-capital must be positive"),
        (["--initial-capital", "-500"], "--initial-capital must be positive"),
        (["--initial-capital", "nan"], "--initial-capital must be positive"),
        (["--initial-capital", "inf"], "--initial-capital must be positive"),
        (["--ruin-threshold", "0"], "--ruin-threshold must be a fraction"),
        (["--ruin-threshold", "-0.1"], "--ruin-threshold must be a fraction"),
        (["--ruin-threshold", "1.5"], "--ruin-threshold must be a fraction"),
        (["--ruin-threshold", "nan"], "--ruin-threshold must be a fraction"),
        (["--ruin-threshold", "inf"], "--ruin-threshold must be a fraction"),
    ],
)
def test_optimize_validate_rejects_invalid_flags_as_usage_error(
    tmp_path, monkeypatch, flags, expected_message
):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("not,a,valid,ledger\n")

    def unexpected_load(_path):
        raise AssertionError("trade file must not be loaded for invalid flags")

    monkeypatch.setattr(
        "screener.backtester.optimization.cli._load_trades", unexpected_load
    )
    res = CliRunner().invoke(
        cli,
        ["optimize", "validate", "--trades", str(ledger), *flags],
    )

    assert res.exit_code == 2, res.output
    assert expected_message in res.output
    assert "Traceback" not in res.output


def test_optimize_validate_reports_non_finite_csv_return_cleanly(tmp_path):
    ledger = tmp_path / "bad.csv"
    ledger.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
        "AAA,1,2024-03-04,2024-03-05,100,2024-03-06,100,time,1,100,100,nan,nan\n"
    )
    res = CliRunner().invoke(
        cli, ["optimize", "validate", "--trades", str(ledger), "--iterations", "10"]
    )
    assert res.exit_code == 1, res.output
    assert "trade returns must be finite" in res.output
    assert "Traceback" not in res.output


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
    base = _config(slippage_bps=5.0, slippage_model=FixedBpsSlippage(bps=5.0))
    half_spread = _config(
        slippage_bps=5.0, slippage_model=HalfSpreadSlippage(half_spread_bps=5.0)
    )

    base_key = _cache_key(
        base,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
        n_trials=1,
    )
    half_spread_key = _cache_key(
        half_spread,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
        n_trials=1,
    )

    assert base_key != half_spread_key


def test_grid_cache_key_includes_trial_count():
    """A cached Deflated Sharpe was deflated by the grid that produced it."""
    cfg = _config()
    key_args = {
        "runner": "historical",
        "start_date": None,
        "end_date": None,
        "metric": "sharpe",
        "min_trades": 1,
    }

    small = _cache_key(cfg, {"hold": 5}, n_trials=2, **key_args)
    large = _cache_key(cfg, {"hold": 5}, n_trials=40, **key_args)

    assert small != large


def test_rolling_grid_reuses_prepared_data_for_runtime_parameters():
    class CountingFetcher(StubPriceFetcher):
        def __init__(self, data):
            super().__init__(data)
            self.calls = 0

        def fetch(self, tickers, start, end):
            self.calls += 1
            return super().fetch(tickers, start, end)

    bars = make_bars(n=80, drift=0.4)
    fetcher = CountingFetcher({"AAA": bars, "SPY": bars})
    cfg = _config(tickers=("AAA",), min_price=None, min_avg_dollar_volume=None)

    results = grid_search(
        cfg,
        fetcher,
        {"hold": [3, 5, 8], "top": [1, 2]},
        runner="rolling",
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
        top_n=6,
        max_workers=1,
    )

    assert len(results) == 6
    assert fetcher.calls == 1


def test_optimization_metrics_preserve_canonical_values():
    equity = pd.Series([100.0, 110.0, 105.0, 120.0])
    trades = [_trade(10.0, 0.10), _trade(-5.0, -0.05)]
    res = BacktestResult(
        config=_config(),
        trades=trades,
        equity_curve=equity,
        benchmark_curve=equity,
        metrics={
            "total_return": 0.20,
            "sharpe": 9.99,
            "profit_factor": 2.0,
            "max_drawdown": -5 / 110,
            "hit_rate": 0.5,
            "expectancy": 0.025,
            "calmar": 123.45,
        },
    )

    metrics = optimization_metrics(res)

    assert metrics["sharpe"] == pytest.approx(9.99)
    assert metrics["calmar"] == pytest.approx(123.45)
    assert metrics["max_drawdown"] == pytest.approx(-5 / 110)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["risk_adjusted_return"] == pytest.approx(4.4)
    assert metrics["trade_count"] == pytest.approx(2.0)
    assert "dsr" not in metrics


def test_optimization_metrics_deflate_the_sharpe_by_the_trial_count():
    """The optimizer layer owns the deflation: the engines only ever see 1."""
    rng = np.random.default_rng(11)
    equity = pd.Series(100_000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, 260)))
    res = BacktestResult(
        config=_config(),
        trades=[_trade(10.0, 0.10)],
        equity_curve=equity,
        benchmark_curve=equity,
        metrics=compute_metrics(equity, pd.Series(dtype=float), [], slot_count=1),
    )

    single = optimization_metrics(res)
    searched = optimization_metrics(res, n_trials=32)

    assert "dsr" not in single
    assert searched["dsr_trials"] == pytest.approx(32.0)
    assert searched["dsr"] == pytest.approx(deflated_sharpe(equity, n_trials=32))
    assert searched["dsr"] < searched["psr"]
    assert searched["psr"] == pytest.approx(single["psr"])


def test_grid_search_deflates_every_row_by_the_configurations_it_evaluated():
    """n_trials is the size of the grid the search actually ran."""
    # Mild drift: a strongly trending curve saturates PSR at 1.0, which hides
    # the deflation instead of measuring it.
    bars = make_bars(n=120, drift=0.03, seed=4)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars})
    cfg = _config(tickers=("AAA",), min_price=None, min_avg_dollar_volume=None)
    parameter_grid = {"hold": [3, 5, 8], "top": [1, 2]}

    results = grid_search(
        cfg,
        fetcher,
        parameter_grid,
        runner="rolling",
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
        top_n=6,
        max_workers=1,
    )

    expected = len(parameter_combinations(parameter_grid))
    assert expected == 6
    for result in results:
        assert result.metrics["dsr_trials"] == pytest.approx(float(expected))
        assert result.metrics["dsr"] < result.metrics["psr"]


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


def test_optimize_rejects_a_swept_hold_below_one():
    """A bad grid point must fail before the sweep, not run as a one-bar hold.

    ``grid_search`` applies each combination with ``model_copy(update=...)``,
    which pydantic does not re-validate, so the ``hold >= 1`` bound on
    ``BacktestConfig`` cannot catch it mid-sweep.
    """
    runner = CliRunner()
    bars = make_bars(n=80, seed=21, open_base=100.0)
    # Injected so a regression fails on the assertion rather than on the
    # network: without the parse-time check the sweep runs to completion.
    fetcher = StubPriceFetcher({"AAA": bars, "BBB": bars, "SPY": bars})

    res = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "5,0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 2, res.output
    assert "--hold" in res.output
    assert "must be >= 1" in res.output

    res = runner.invoke(
        cli,
        [
            "optimize",
            "research-report",
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--param",
            "hold=-5",
            "--out",
            "unused",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 2, res.output
    assert "--hold" in res.output
    assert "must be >= 1" in res.output


def test_cli_help_exposes_trial_identity_options():
    runner = CliRunner()
    for command in ("grid", "walk-forward", "research-report"):
        res = runner.invoke(cli, ["optimize", command, "--help"])
        assert res.exit_code == 0, res.output
        assert "--experiment-id" in res.output
        assert "--trial-db" in res.output
        assert "--n-trials-effective" in res.output
        if command in {"walk-forward", "research-report"}:
            assert "calendar days" in res.output.lower()


def test_cli_rejects_invalid_trial_identity_options():
    runner = CliRunner()
    bars = make_bars(n=40, seed=3, open_base=100.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars})
    res = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "5",
            "--n-trials-effective",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 2, res.output
    assert "--n-trials-effective" in res.output

    res = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "5",
            "--experiment-id",
            "   ",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 2, res.output
    assert "--experiment-id" in res.output


def test_cli_grid_persists_rejected_trials_under_explicit_family(tmp_path):
    """End-to-end offline CLI: rejected trials remain in the shared family DB."""
    import sqlite3

    from screener.backtester.optimization.trials import load_trial_search_stats

    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    bars_b = make_bars(n=80, seed=22, open_base=50.0)
    spy = make_bars(n=80, seed=99, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    trial_db = tmp_path / "family_trials.db"
    exp = "cli-family-identity"
    start = bars_a.index[20].date().isoformat()
    end = bars_a.index[60].date().isoformat()
    runner = CliRunner()

    # High min-trades forces rejection while still recording the attempt.
    res = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA,BBB",
            "--start",
            start,
            "--end",
            end,
            "--entry",
            "close > sma(close, 3)",
            "--stop-loss",
            "none",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5,8",
            "--min-trades",
            "500",
            "--top-n",
            "1",
            "--workers",
            "1",
            "--experiment-id",
            exp,
            "--trial-db",
            str(trial_db),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert trial_db.exists()

    conn = sqlite3.connect(trial_db)
    try:
        rows = conn.execute(
            "SELECT status, params_json FROM trials WHERE experiment_id = ?",
            (exp,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) >= 2
    assert any(status == "rejected" for status, _ in rows)

    # Second search under the same family accumulates history.
    res2 = runner.invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA,BBB",
            "--start",
            start,
            "--end",
            end,
            "--entry",
            "close > sma(close, 3)",
            "--stop-loss",
            "none",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5,8,13",
            "--min-trades",
            "500",
            "--top-n",
            "1",
            "--workers",
            "1",
            "--experiment-id",
            exp,
            "--trial-db",
            str(trial_db),
            "--n-trials-effective",
            "2",
        ],
        obj=fetcher,
    )
    assert res2.exit_code == 0, res2.output
    stats = load_trial_search_stats(exp, db_path=trial_db, n_trials_effective=2)
    assert stats.n_trials_nominal >= 3
    assert stats.n_trials_effective == 2


def test_holding_period_fold_warnings_for_long_holds():
    from screener.backtester.optimization.walk_forward import (
        HOLDING_PERIOD_FIT_WARNING_PREFIX,
        holding_period_fold_warnings,
    )

    assert holding_period_fold_warnings({"hold": [5, 10]}, test_days=63) == []
    assert holding_period_fold_warnings({"hold": [50]}, test_days=63)
    warnings = holding_period_fold_warnings({"hold": [21, 126]}, test_days=63)
    assert len(warnings) == 1
    assert warnings[0].startswith(HOLDING_PERIOD_FIT_WARNING_PREFIX)
    assert "126" in warnings[0] and "63" in warnings[0]

    selected = holding_period_fold_warnings(
        {"hold": [5]}, test_days=10, selected_holds=[21]
    )
    assert selected and "21" in selected[0]


def test_research_report_isolates_descriptive_and_each_training_period(tmp_path):
    import sqlite3

    from screener.backtester.optimization.research_report import run_research_report

    bars = make_bars(n=95, seed=71, open_base=100.0)
    cfg = BacktestConfig(
        market="india",
        as_of=bars.index[0].date(),
        tickers=("AAA",),
        benchmark="^NSEI",
        entry_expr="close > 0",
        exit_expr=None,
        hold=3,
        top=1,
        initial_capital=100_000,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
    )
    database = tmp_path / "trials.db"
    run_research_report(
        cfg,
        StubPriceFetcher({"AAA": bars, "AAA.NS": bars, "^NSEI": bars}),
        {"hold": [3, 5]},
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
        train_days=35,
        test_days=28,
        min_trades=1,
        max_workers=1,
        mc_iterations=10,
        mc_block=2,
        out_path=tmp_path / "report",
        experiment_id="india-family",
        trial_db_path=database,
    )
    with sqlite3.connect(database) as connection:
        ids = {
            row[0]
            for row in connection.execute("SELECT DISTINCT experiment_id FROM trials")
        }
    assert "india-family::descriptive" in ids
    training_ids = ids - {"india-family::descriptive"}
    assert len(training_ids) >= 2
    assert all(value.startswith("india-family::train:") for value in training_ids)
