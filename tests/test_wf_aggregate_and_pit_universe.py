"""Regressions for walk-forward aggregate metrics and PIT universe capping."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization.cli import optimize, pit_active_membership
from screener.backtester.optimization.grid import GridSearchResult
from screener.backtester.optimization.walk_forward import walk_forward_optimize
from screener.universes import UniverseSelection
from tests.conftest import StubPriceFetcher


def _config(**overrides) -> BacktestConfig:
    values = {
        "market": "us",
        "as_of": date(2024, 1, 1),
        "hold": 5,
        "top": 1,
        "entry_expr": "close > 0",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA",),
        "min_price": None,
        "min_avg_dollar_volume": None,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def _trade(pnl: float, return_pct: float, *, signal_date: date) -> Trade:
    return Trade(
        ticker="AAA",
        rank=1,
        signal_date=signal_date,
        entry_date=signal_date + timedelta(days=1),
        entry_price=100.0,
        exit_date=signal_date + timedelta(days=2),
        exit_price=100.0 * (1.0 + return_pct),
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=100.0 + pnl,
        pnl=pnl,
        return_pct=return_pct,
    )


def _fold_equity(
    cfg: BacktestConfig,
    start: date,
    end: date,
    *,
    pattern: tuple[float, ...] = (0.01, -0.004, 0.006, -0.002, 0.003),
) -> pd.Series:
    index = pd.bdate_range(start, end)
    steps = np.resize(np.asarray(pattern, dtype=float), len(index))
    steps[0] = 0.0
    values = float(cfg.initial_capital) * np.cumprod(1.0 + steps)
    return pd.Series(values, index=index, dtype=float)


def _install_fake_search(monkeypatch, *, metrics: dict[str, float], score: float):
    import screener.backtester.optimization.walk_forward as wf_mod

    def fake_grid(*args, **kwargs):
        return [
            GridSearchResult(
                params={"hold": 5},
                score=score,
                metrics=dict(metrics),
                trade_count=3,
            )
        ]

    def fake_roll(cfg_arg, fetcher, *, start_date, end_date):
        equity = _fold_equity(cfg_arg, start_date, end_date)
        return BacktestResult(
            config=cfg_arg,
            trades=[_trade(1.0, 0.01, signal_date=start_date)],
            equity_curve=equity,
            benchmark_curve=equity,
            metrics={"sharpe": 1.0},
        )

    monkeypatch.setattr(wf_mod, "grid_search", fake_grid)
    monkeypatch.setattr(wf_mod, "run_rolling_backtest", fake_roll)
    return wf_mod


def test_walk_forward_dsr_metric_uses_real_trial_count(monkeypatch, tmp_path):
    """``metric="dsr"`` must not lose its aggregate objective to n_trials=1.

    Fold selection records ``dsr_trials=4``; the combined OOS Deflated Sharpe
    has to reuse that count. With the old single-trial default the aggregate
    dict had no ``"dsr"`` key at all, so the OOS objective was None and the
    run was marked insufficient even with a full OOS path.
    """
    wf_mod = _install_fake_search(
        monkeypatch,
        metrics={"dsr": 1.4, "dsr_trials": 4.0},
        score=1.4,
    )
    monkeypatch.setattr(
        wf_mod,
        "fetch_benchmark",
        lambda symbol, start, end, fetcher: pd.Series(
            400.0, index=pd.bdate_range(start, end), dtype=float
        ),
    )
    summary = walk_forward_optimize(
        _config(),
        StubPriceFetcher({}),
        {"hold": [5, 10]},
        metric="dsr",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 30),
        train_days=20,
        test_days=20,
        step_days=20,
        min_trades=1,
        trial_db_path=tmp_path / "trials.db",
    )

    assert len(summary.windows) >= 2
    assert summary.evidence["missing_eligible_folds"] == 0
    assert "dsr" in summary.aggregate_metrics
    assert summary.aggregate_metrics["dsr_trials"] == pytest.approx(4.0)
    assert math.isfinite(summary.aggregate_metrics["dsr"])
    assert summary.evidence["oos_objective_finite"] is True
    assert summary.evidence["aggregate_dsr_trials"] == 4
    assert summary.insufficient_data is False


def test_walk_forward_aggregate_passes_raw_benchmark_marks(monkeypatch):
    """A benchmark gap must not become a fake 0% bar in the OOS alpha/beta.

    The benchmark drives the session calendar, so the gap survives only because
    the fold equity (the book's own panel) still has that bar. Forward-filling
    it would pair a 0% benchmark return with a real book return, changing the
    regression slope; the raw marks drop that incomplete pair instead.
    """
    wf_mod = _install_fake_search(monkeypatch, metrics={"sharpe": 1.0}, score=1.0)
    start, end = date(2024, 1, 1), date(2024, 4, 15)
    gap_day = pd.Timestamp("2024-02-14")
    full_index = pd.bdate_range(start, end)
    benchmark = pd.Series(
        400.0 + np.arange(len(full_index)) * 0.5,
        index=full_index,
        dtype=float,
    ).drop(index=gap_day)
    monkeypatch.setattr(
        wf_mod,
        "fetch_benchmark",
        lambda symbol, begin, finish, fetcher: benchmark,
    )
    calls: list[dict] = []
    real_compute_metrics = compute_metrics

    def capture(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_compute_metrics(*args, **kwargs)

    monkeypatch.setattr(wf_mod, "compute_metrics", capture)

    summary = walk_forward_optimize(
        _config(),
        StubPriceFetcher({}),
        {"hold": [5]},
        start_date=start,
        end_date=end,
        train_days=20,
        test_days=15,
        step_days=15,
        min_trades=1,
        trial_db_path=None,
    )

    assert len(calls) == 1
    equity, aligned, trades, slots = calls[0]["args"]
    kwargs = calls[0]["kwargs"]
    marks = kwargs["benchmark_marks"]
    assert marks is not None
    assert marks.index.equals(equity.index)
    assert gap_day in equity.index
    assert pd.isna(marks.loc[gap_day])
    assert aligned.notna().all()

    with_marks = real_compute_metrics(equity, aligned, trades, slots, **kwargs)
    without_marks = real_compute_metrics(
        equity,
        aligned,
        trades,
        slots,
        **{**kwargs, "benchmark_marks": None},
    )
    assert summary.aggregate_metrics["beta"] == pytest.approx(with_marks["beta"])
    assert summary.aggregate_metrics["alpha_annual"] == pytest.approx(
        with_marks["alpha_annual"]
    )
    # The regression really did see the gap: the filled convention differs.
    assert with_marks["beta"] != pytest.approx(without_marks["beta"])


def test_walk_forward_aggregate_benchmark_applies_splits_only_adjustment(monkeypatch):
    """Under price_adjustment='splits_only', the aggregate benchmark is split-adjusted."""
    wf_mod = _install_fake_search(monkeypatch, metrics={"sharpe": 1.0}, score=1.0)
    start, end = date(2024, 1, 1), date(2024, 4, 15)
    full_index = pd.bdate_range(start, end)
    split_day = pd.Timestamp("2024-02-15")
    gap_day = pd.Timestamp("2024-02-20")
    before_split = full_index < split_day
    raw_close = np.where(before_split, 200.0, 100.0)
    split_factor = np.where(before_split, 2.0, 1.0)
    spy_frame = pd.DataFrame(
        {
            "open": raw_close,
            "high": raw_close + 1.0,
            "low": raw_close - 1.0,
            "close": raw_close,
            "volume": 1_000_000.0,
            "split_factor": split_factor,
        },
        index=full_index,
    ).drop(index=gap_day)

    calls: list[dict] = []
    real_compute_metrics = compute_metrics

    def capture(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_compute_metrics(*args, **kwargs)

    monkeypatch.setattr(wf_mod, "compute_metrics", capture)

    summary = walk_forward_optimize(
        _config(price_adjustment="splits_only"),
        StubPriceFetcher({"SPY": spy_frame}),
        {"hold": [5]},
        start_date=start,
        end_date=end,
        train_days=20,
        test_days=15,
        step_days=15,
        min_trades=1,
        trial_db_path=None,
    )

    # After 2:1 split adjustment, SPY is flat at 100.0 (0% return, not -50%).
    assert summary.aggregate_metrics["benchmark_return"] == pytest.approx(0.0)
    assert len(calls) == 1
    marks = calls[0]["kwargs"]["benchmark_marks"]
    assert marks is not None
    assert pd.isna(marks.loc[gap_day])
    assert marks.dropna().eq(100.0).all()


def test_pit_helper_keeps_window_overlap_before_max_universe_slots():
    symbols = ("OLD1", "OLD2", "OLD3", "NEW1", "NEW2", "LATER")
    windows = (
        ("OLD1", date(2000, 1, 1), date(2005, 1, 1)),
        ("OLD2", date(2000, 1, 1), date(2008, 1, 1)),
        ("OLD3", date(2000, 1, 1), date(2023, 12, 31)),
        ("NEW1", date(2024, 2, 1), None),
        ("NEW2", date(2024, 3, 1), None),
        ("LATER", date(2025, 1, 1), None),
    )

    active, active_windows = pit_active_membership(
        symbols,
        windows,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    # The three dead listings precede every live one in selection order; they
    # must not be the names a --max-universe slice keeps.
    assert active == ("NEW1", "NEW2")
    assert {symbol for symbol, _from, _until in active_windows} == {"NEW1", "NEW2"}

    # Half-open windows: ending exactly at the start date is not eligible.
    half_open = (
        ("END_AT_START", date(2000, 1, 1), date(2024, 1, 1)),
        ("STILL_OPEN", date(2000, 1, 1), date(2024, 1, 2)),
    )
    active, _ = pit_active_membership(
        ("END_AT_START", "STILL_OPEN"),
        half_open,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )
    assert active == ("STILL_OPEN",)


def test_research_report_pit_cap_keeps_late_additions(tmp_path, monkeypatch):
    """``--max-universe`` must not spend its slots on exited-before-start names."""
    import screener.backtester.optimization.research_report as report_module
    import screener.universes as universes

    selection = UniverseSelection(
        name="sp500",
        market="us",
        benchmark="SPY",
        symbols=("OLD1", "OLD2", "NEW1", "NEW2"),
        source="test point-in-time snapshots",
        membership_windows=(
            ("OLD1", date(2000, 1, 1), date(2005, 1, 1)),
            ("OLD2", date(2000, 1, 1), date(2010, 1, 1)),
            ("NEW1", date(2024, 2, 1), None),
            ("NEW2", date(2024, 3, 1), None),
        ),
    )
    monkeypatch.setattr(
        universes, "load_universe_selection", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher",
        lambda: StubPriceFetcher({}),
    )
    captured: dict[str, object] = {}

    def fake_report(cfg, *args, **kwargs):
        captured["cfg"] = cfg
        captured["warnings"] = kwargs.get("universe_warnings")

    monkeypatch.setattr(report_module, "run_research_report", fake_report)

    result = CliRunner().invoke(
        optimize,
        [
            "research-report",
            "--entry",
            "close > 0",
            "--start",
            "2024-01-01",
            "--end",
            "2024-04-01",
            "--hold",
            "3",
            "--top",
            "1",
            "--train-days",
            "30",
            "--test-days",
            "15",
            "--step-days",
            "15",
            "--mc-iterations",
            "10",
            "--mc-block",
            "2",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--metric",
            "total_return",
            "--trial-db",
            str(tmp_path / "trials.db"),
            "--out",
            str(tmp_path / "report"),
            "--universe",
            "sp500",
            "--max-universe",
            "1",
            "--point-in-time",
            "--param",
            "hold=3",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    assert cfg.tickers == ("NEW1", "NEW2")
    assert {window[0] for window in cfg.membership_windows} == {"NEW1", "NEW2"}
    warnings = captured["warnings"]
    assert any("--max-universe 1 ignored for point-in-time" in w for w in warnings)
