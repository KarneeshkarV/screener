"""Offline, deterministic coverage tests for ``screener.backtester.optimization``.

Targets 100% line coverage across cli, walk_forward, grid, reporting,
monte_carlo, and metrics. All tests are network-free: any provider / fetcher is
stubbed (``StubPriceFetcher``) or the backtest runner is monkeypatched.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from rich.console import Console

from main import cli
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization import cli as opt_cli
from screener.backtester.optimization import grid as grid_module
from screener.backtester.optimization import walk_forward as wf_module
from screener.backtester.optimization.grid import (
    GridSearchResult,
    _config_fingerprint,
    _from_cache,
    _json_default,
    _load_cache,
    _run_one,
    _run_one_safe,
    _save_cache,
    _stable_fingerprint,
    grid_search,
)
from screener.backtester.optimization.metrics import (
    calmar_ratio,
    optimization_metrics,
    profit_factor,
    score_result,
    win_rate,
)
from screener.backtester.optimization.monte_carlo import (
    MonteCarloResult,
    _drawdown,
    simulate_monte_carlo,
)
from screener.backtester.optimization.reporting import (
    print_walk_forward_table,
    write_html_report,
    write_json_report,
)
from screener.backtester.optimization.walk_forward import (
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindow,
    _parameter_stability,
    generate_walk_forward_windows,
    walk_forward_optimize,
)
from tests.conftest import StubPriceFetcher, make_bars

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
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


def _result(cfg, trades=None, equity=None):
    eq = equity if equity is not None else pd.Series([100_000.0, 101_000.0])
    return BacktestResult(
        config=cfg,
        trades=trades if trades is not None else [_trade(10.0, 0.10)],
        equity_curve=eq,
        benchmark_curve=eq,
        metrics={"total_return": 0.01},
    )


# --------------------------------------------------------------------------- #
# metrics.py                                                                   #
# --------------------------------------------------------------------------- #
def test_profit_factor_all_profit_returns_inf():
    # gross_loss == 0 and gross_profit > 0  -> inf  (line 37/43 branch)
    assert profit_factor([_trade(5.0, 0.05)]) == float("inf")


def test_win_rate_empty_returns_zero():
    assert win_rate([]) == 0.0


def test_calmar_ratio_short_series_returns_zero():
    assert calmar_ratio(pd.Series([100.0])) == 0.0  # len < 2
    assert calmar_ratio(pd.Series(dtype=float)) == 0.0  # empty


def test_calmar_ratio_no_drawdown_inf_and_zero():
    rising = pd.Series([100.0, 110.0, 121.0, 133.0])
    assert calmar_ratio(rising) == float("inf")  # cagr > 0, dd == 0
    flat = pd.Series([100.0, 100.0, 100.0])
    assert calmar_ratio(flat) == 0.0  # cagr <= 0, dd == 0


def test_score_result_nan_metric_returns_neg_inf():
    cfg = _config()
    eq = pd.Series([100.0, 110.0])
    res = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=eq,
        benchmark_curve=eq,
        metrics={"weird": float("nan")},
    )
    assert score_result(res, "weird") == float("-inf")


def test_maximum_drawdown_empty_series_returns_zero():
    from screener.backtester.optimization.metrics import maximum_drawdown

    assert maximum_drawdown(pd.Series(dtype=float)) == 0.0


def test_optimization_metrics_includes_expected_keys():
    cfg = _config()
    res = _result(cfg)
    metrics = optimization_metrics(res)
    for key in [
        "sharpe",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "expectancy",
        "calmar",
        "risk_adjusted_return",
        "trade_count",
    ]:
        assert key in metrics


# --------------------------------------------------------------------------- #
# monte_carlo.py                                                               #
# --------------------------------------------------------------------------- #
def test_drawdown_empty_array_returns_zero():
    assert _drawdown(np.array([])) == 0.0


def test_monte_carlo_rejects_non_positive_iterations():
    with pytest.raises(ValueError, match="iterations must be positive"):
        simulate_monte_carlo([_trade(1.0, 0.01)], iterations=0)


def test_monte_carlo_rejects_non_positive_capital():
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        simulate_monte_carlo([_trade(1.0, 0.01)], initial_capital=0.0)


def test_monte_carlo_empty_trades_returns_zeroed_result():
    res = simulate_monte_carlo([], iterations=10)
    assert isinstance(res, MonteCarloResult)
    assert res.median_return == 0.0
    assert res.risk_of_ruin == 0.0


def test_monte_carlo_ruin_counted_with_catastrophic_losses():
    # A big negative return drives equity below the ruin level on every path.
    trades = [_trade(-90.0, -0.9), _trade(-90.0, -0.9)]
    res = simulate_monte_carlo(trades, iterations=50, seed=1, ruin_threshold=0.5)
    assert res.risk_of_ruin > 0.0


# --------------------------------------------------------------------------- #
# reporting.py                                                                 #
# --------------------------------------------------------------------------- #
def test_json_default_handles_date_model_and_infinities(tmp_path):
    from decimal import Decimal

    # Native float inf serialises to JSON ``Infinity`` without hitting our
    # ``default`` hook; ``Decimal`` infinities are non-serialisable so they
    # reach ``_json_default`` and exercise the +/-inf branches.
    payload = {
        "d": date(2024, 1, 1),
        "model": GridSearchResult(
            params={"hold": 5}, score=1.0, metrics={}, trade_count=1
        ),
        "pos_inf": Decimal("Infinity"),
        "neg_inf": Decimal("-Infinity"),
        "obj": object(),
    }
    path = tmp_path / "out.json"
    write_json_report(payload, path)
    loaded = json.loads(path.read_text())
    assert loaded["d"] == "2024-01-01"
    assert loaded["model"]["params"] == {"hold": 5}
    assert loaded["pos_inf"] == "inf"
    assert loaded["neg_inf"] == "-inf"
    assert isinstance(loaded["obj"], str)


def test_print_walk_forward_table_renders_windows():
    window = WalkForwardWindow(
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 31),
        test_start=date(2024, 2, 1),
        test_end=date(2024, 2, 28),
    )
    best = GridSearchResult(
        params={"hold": 10}, score=1.5, metrics={"sharpe": 1.2}, trade_count=4
    )
    summary = WalkForwardSummary(
        windows=[
            WalkForwardResult(
                window=window,
                best_train=best,
                test_metrics={"sharpe": 0.8},
                test_trade_count=3,
            )
        ],
        stability_score=0.9,
        aggregate_metrics={"sharpe": 0.8},
        overfit_flag=False,
        train_test_score_ratio=1.4,
    )
    console = Console(record=True, width=200)
    print_walk_forward_table(summary, console=console)
    text = console.export_text()
    assert "Walk-Forward Results" in text
    assert "Stability" in text
    assert "Overfit flag" in text


def test_write_html_report_without_disclaimer(tmp_path):
    path = tmp_path / "wf.html"
    write_html_report([{"sharpe": 1.0}], path, "Walk-Forward Report")
    html = path.read_text()
    assert "Walk-Forward Report" in html
    assert "background:#fff3cd" not in html  # banner omitted


# --------------------------------------------------------------------------- #
# grid.py                                                                      #
# --------------------------------------------------------------------------- #
def test_grid_json_default_branches():
    assert _json_default(date(2024, 1, 1)) == "2024-01-01"
    assert _json_default((1, 2)) == [1, 2]
    assert _json_default(float("inf")) == "inf"
    assert _json_default("plain") == "plain"


def test_stable_fingerprint_branches():
    assert _stable_fingerprint(None) is None
    assert _stable_fingerprint(5) == 5
    assert _stable_fingerprint(date(2024, 1, 1)) == "2024-01-01"
    assert _stable_fingerprint((1, 2)) == [1, 2]
    assert _stable_fingerprint([1, 2]) == [1, 2]
    assert _stable_fingerprint({"b": 1, "a": 2}) == {"a": 2, "b": 1}

    class _Plain:
        def __repr__(self) -> str:
            return "<plain>"

    fp = _stable_fingerprint(_Plain())
    assert fp["repr"] == "<plain>"
    assert "__class__" in fp


def test_stable_fingerprint_basemodel():
    fp = _stable_fingerprint(
        GridSearchResult(params={"hold": 1}, score=1.0, metrics={}, trade_count=0)
    )
    assert fp["fields"]["params"] == {"hold": 1}
    assert "__class__" in fp


def test_config_fingerprint_serialises_slippage_model():
    cfg = _config()
    fp = _config_fingerprint(cfg)
    assert "slippage_model" in fp


def test_load_cache_missing_and_corrupt(tmp_path):
    assert _load_cache(None) == {}
    missing = tmp_path / "nope.json"
    assert _load_cache(missing) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json")
    assert _load_cache(corrupt) == {}  # JSONDecodeError path


def test_save_cache_none_path_is_noop():
    _save_cache(None, {"k": {"v": 1}})  # must not raise


def test_run_one_rolling_requires_dates():
    cfg = _config()
    with pytest.raises(ValueError, match="rolling grid search requires"):
        _run_one(
            cfg,
            {"hold": 5},
            StubPriceFetcher({}),
            "rolling",
            None,
            None,
            "sharpe",
            1,
        )


def test_run_one_historical(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(grid_module, "run_backtest", lambda c, f: _result(c))
    out = _run_one(
        cfg,
        {"hold": 5},
        StubPriceFetcher({}),
        "historical",
        None,
        None,
        "total_return",
        1,
    )
    assert out.trade_count == 1
    assert out.score == pytest.approx(0.01)


def test_run_one_below_min_trades_scores_neg_inf(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(
        grid_module,
        "run_backtest",
        lambda c, f: _result(c, trades=[]),
    )
    out = _run_one(
        cfg, {"hold": 5}, StubPriceFetcher({}), "historical", None, None, "sharpe", 1
    )
    assert out.score == float("-inf")


def test_run_one_safe_catches_exception(monkeypatch):
    cfg = _config()

    def boom(c, f):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(grid_module, "run_backtest", boom)
    args = (
        cfg,
        {"hold": 5},
        StubPriceFetcher({}),
        "historical",
        None,
        None,
        "sharpe",
        1,
    )
    out = _run_one_safe(args)
    assert out.score == float("-inf")
    assert out.error == "kaboom"
    assert out.trade_count == 0


def test_run_one_safe_reraises_keyboard_interrupt(monkeypatch):
    cfg = _config()

    def interrupt(c, f):
        raise KeyboardInterrupt

    monkeypatch.setattr(grid_module, "run_backtest", interrupt)
    args = (
        cfg,
        {"hold": 5},
        StubPriceFetcher({}),
        "historical",
        None,
        None,
        "sharpe",
        1,
    )
    with pytest.raises(KeyboardInterrupt):
        _run_one_safe(args)


def test_from_cache_rehydrates_record():
    record = {
        "params": {"hold": 5},
        "score": 1.5,
        "metrics": {"sharpe": "1.2"},
        "trade_count": 3,
        "error": None,
    }
    out = _from_cache(record)
    assert out.cached is True
    assert out.score == 1.5
    assert out.metrics["sharpe"] == 1.2
    assert out.trade_count == 3


def test_grid_search_cache_hit_short_circuits(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_run_backtest(c, f):
        calls["n"] += 1
        return _result(c)

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
    # Second call with identical key must read from cache (no new run).
    second = grid_search(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5]},
        metric="total_return",
        min_trades=1,
        cache_path=cache_path,
    )
    assert calls["n"] == 1
    assert second[0].cached is True
    assert second[0].score == pytest.approx(first[0].score)


def test_grid_search_process_pool_path(monkeypatch):
    # Exercise the max_workers != 1 ProcessPoolExecutor branch deterministically
    # by replacing the executor with an in-process synchronous double.
    cfg = _config()

    monkeypatch.setattr(grid_module, "run_backtest", lambda c, f: _result(c))

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _SyncPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, arg):
            return _ImmediateFuture(fn(arg))

    monkeypatch.setattr(grid_module, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(grid_module, "as_completed", lambda futures: list(futures))

    results = grid_search(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5, 10]},
        metric="total_return",
        max_workers=2,
    )
    assert len(results) == 2
    assert all(not r.cached for r in results)


def test_grid_search_keyboard_interrupt_saves_cache(monkeypatch, tmp_path):
    cfg = _config()

    def interrupt(arg):
        raise KeyboardInterrupt

    monkeypatch.setattr(grid_module, "_run_one_safe", interrupt)
    cache_path = tmp_path / "grid.json"
    with pytest.raises(KeyboardInterrupt):
        grid_search(
            cfg,
            StubPriceFetcher({}),
            {"hold": [5]},
            metric="total_return",
            cache_path=cache_path,
        )
    # cache file is flushed (empty dict) on interrupt
    assert cache_path.exists()


# --------------------------------------------------------------------------- #
# walk_forward.py                                                              #
# --------------------------------------------------------------------------- #
def test_generate_windows_rejects_bad_durations():
    with pytest.raises(ValueError, match="train_days and test_days must be positive"):
        generate_walk_forward_windows(
            date(2024, 1, 1), date(2024, 4, 1), train_days=0, test_days=10
        )


def test_generate_windows_rejects_bad_step():
    with pytest.raises(ValueError, match="step_days must be positive"):
        generate_walk_forward_windows(
            date(2024, 1, 1),
            date(2024, 4, 1),
            train_days=30,
            test_days=10,
            step_days=-1,
        )


def test_parameter_stability_single_set_returns_one():
    assert _parameter_stability([{"hold": 5}]) == 1.0
    assert _parameter_stability([]) == 1.0


def test_parameter_stability_numeric_and_categorical():
    # All-numeric values -> variance-based stability.
    numeric = _parameter_stability([{"hold": 10}, {"hold": 10}, {"hold": 10}])
    assert numeric == pytest.approx(1.0)
    # Mixed/categorical (None present) -> unique-count branch.
    categorical = _parameter_stability([{"stop_loss": None}, {"stop_loss": 0.05}])
    assert 0.0 <= categorical <= 1.0


def _patch_wf(monkeypatch, *, ranked_factory, test_result_factory):
    monkeypatch.setattr(wf_module, "grid_search", ranked_factory)
    monkeypatch.setattr(wf_module, "run_rolling_backtest", test_result_factory)


def test_walk_forward_optimize_full_path(monkeypatch):
    cfg = _config()

    best = GridSearchResult(
        params={"hold": 10}, score=2.0, metrics={"sharpe": 2.0}, trade_count=5
    )

    def fake_grid_search(c, f, grid, **kwargs):
        return [best]

    def fake_rolling(c, f, *, start_date, end_date):
        eq = pd.Series([100_000.0, 101_000.0, 102_000.0])
        return BacktestResult(
            config=c,
            trades=[_trade(5.0, 0.05), _trade(3.0, 0.03)],
            equity_curve=eq,
            benchmark_curve=eq,
            metrics={"total_return": 0.02},
        )

    _patch_wf(
        monkeypatch, ranked_factory=fake_grid_search, test_result_factory=fake_rolling
    )

    summary = walk_forward_optimize(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5, 10]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        train_days=30,
        test_days=20,
        step_days=20,
        metric="sharpe",
        cache_path="grid.json",
    )
    assert isinstance(summary, WalkForwardSummary)
    assert summary.windows
    assert summary.aggregate_metrics  # test_trade_counts > 0 branch
    # overfit detection: train avg (2.0) >> test avg sharpe -> ratio large.
    assert summary.train_test_score_ratio > 0


def test_walk_forward_optimize_empty_ranked_skips(monkeypatch):
    cfg = _config()

    def empty_grid(c, f, grid, **kwargs):
        return []  # `if not ranked: continue` path

    monkeypatch.setattr(wf_module, "grid_search", empty_grid)
    monkeypatch.setattr(
        wf_module,
        "run_rolling_backtest",
        lambda *a, **k: pytest.fail("should not run"),
    )

    summary = walk_forward_optimize(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        train_days=30,
        test_days=20,
        step_days=20,
    )
    assert summary.windows == []
    assert summary.aggregate_metrics == {}  # no trades aggregate
    assert summary.stability_score == 1.0
    assert summary.train_test_score_ratio == 0.0


def test_walk_forward_optimize_no_trades_aggregate_empty(monkeypatch):
    cfg = _config()
    best = GridSearchResult(
        params={"hold": 5}, score=-0.5, metrics={"sharpe": -0.5}, trade_count=0
    )

    def fake_grid_search(c, f, grid, **kwargs):
        return [best]

    def no_trade_rolling(c, f, *, start_date, end_date):
        eq = pd.Series([100_000.0, 100_000.0])
        return BacktestResult(
            config=c,
            trades=[],  # zero test trades -> test_trade_counts stays 0
            equity_curve=eq,
            benchmark_curve=eq,
            metrics={"total_return": 0.0},
        )

    _patch_wf(
        monkeypatch,
        ranked_factory=fake_grid_search,
        test_result_factory=no_trade_rolling,
    )
    summary = walk_forward_optimize(
        cfg,
        StubPriceFetcher({}),
        {"hold": [5]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        train_days=30,
        test_days=20,
        step_days=20,
    )
    assert summary.windows  # a window was recorded
    assert summary.aggregate_metrics == {}  # but no aggregate (no trades)
    assert summary.overfit_flag is False  # train_avg <= 0 -> ratio 0.0


# --------------------------------------------------------------------------- #
# cli.py                                                                       #
# --------------------------------------------------------------------------- #
def test_parse_values_none_and_disabled():
    assert opt_cli._parse_values(None) == [None]
    assert opt_cli._parse_values(None, allow_none=False) == []
    # whitespace/empty items are skipped; none keywords -> None
    assert opt_cli._parse_values(" none , 0.05 ,, off ") == [None, 0.05, None]


def test_parse_values_range_two_and_three_parts():
    assert opt_cli._parse_values("1:3", int) == [1, 2, 3]
    assert opt_cli._parse_values("0:1:0.5") == pytest.approx([0.0, 0.5, 1.0])


def test_parse_values_invalid_range_shape():
    with pytest.raises(Exception) as exc:
        opt_cli._parse_values("1:2:3:4")
    assert "Invalid range" in str(exc.value)


def test_parse_values_non_positive_step():
    with pytest.raises(Exception) as exc:
        opt_cli._parse_values("1:5:0")
    assert "step must be positive" in str(exc.value)


def test_base_config_requires_entry_or_strategy():
    with pytest.raises(Exception) as exc:
        opt_cli._base_config(
            market="us",
            end_date=date(2024, 1, 1),
            hold=5,
            top=1,
            entry_expr=None,
            exit_expr=None,
            strategy_name=None,
            tickers="AAA",
            universe_file=None,
            max_universe=10,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=1000.0,
            benchmark=None,
            min_price=None,
            min_avg_dollar_volume=None,
            adv_window=20,
        )
    assert "--entry or --strategy" in str(exc.value)


def test_base_config_requires_tickers_or_universe():
    with pytest.raises(Exception) as exc:
        opt_cli._base_config(
            market="us",
            end_date=date(2024, 1, 1),
            hold=5,
            top=1,
            entry_expr="close > 0",
            exit_expr=None,
            strategy_name=None,
            tickers=None,
            universe_file=None,
            max_universe=10,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=1000.0,
            benchmark=None,
            min_price=None,
            min_avg_dollar_volume=None,
            adv_window=20,
        )
    assert "--tickers or --universe-file" in str(exc.value)


def test_base_config_strategy_and_zero_filters():
    # strategy_name resolves entry; min_price/min_adv == 0 -> None
    cfg = opt_cli._base_config(
        market="us",
        end_date=date(2024, 1, 1),
        hold=5,
        top=1,
        entry_expr=None,
        exit_expr=None,
        strategy_name="breakout",
        tickers="AAA,BBB",
        universe_file=None,
        max_universe=10,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=1000.0,
        benchmark=None,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
        adv_window=20,
    )
    assert cfg.entry_expr  # populated from the strategy
    assert cfg.min_price is None
    assert cfg.min_avg_dollar_volume is None
    assert cfg.benchmark == "SPY"  # default for us market
    assert cfg.tickers == ("AAA", "BBB")


def test_optimize_grid_with_json_and_html_exports(tmp_path):
    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    bars_b = make_bars(n=80, seed=22, open_base=50.0)
    spy = make_bars(n=80, seed=99, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    json_path = tmp_path / "grid.json"
    html_path = tmp_path / "grid.html"
    res = CliRunner().invoke(
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
            "none",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5",
            "--top-n",
            "1",
            "--workers",
            "1",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert "warning" in payload
    assert "selection bias" in payload["warning"].lower()
    html = html_path.read_text().lower()
    assert "grid search report" in html
    assert "selection bias" in html


def test_optimize_walk_forward_command(monkeypatch, tmp_path):
    # Patch grid_search + rolling runner referenced from walk_forward so the
    # command path runs deterministically without a real backtest.
    best = GridSearchResult(
        params={"hold": 5}, score=1.0, metrics={"sharpe": 1.0}, trade_count=3
    )
    monkeypatch.setattr(wf_module, "grid_search", lambda c, f, g, **k: [best])

    def fake_rolling(c, f, *, start_date, end_date):
        eq = pd.Series([100_000.0, 101_000.0])
        return BacktestResult(
            config=c,
            trades=[_trade(2.0, 0.02)],
            equity_curve=eq,
            benchmark_curve=eq,
            metrics={"total_return": 0.01},
        )

    monkeypatch.setattr(wf_module, "run_rolling_backtest", fake_rolling)

    json_path = tmp_path / "wf.json"
    html_path = tmp_path / "wf.html"
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "walk-forward",
            "--tickers",
            "AAA",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-30",
            "--entry",
            "close > sma(close, 3)",
            "--stop-loss",
            "none",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5",
            "--train-days",
            "30",
            "--test-days",
            "20",
            "--step-days",
            "20",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
        ],
        obj=StubPriceFetcher({}),
    )
    assert res.exit_code == 0, res.output
    assert "Walk-Forward Results" in res.output
    assert json_path.exists()
    assert html_path.exists()


def test_optimize_grid_uses_default_dates_and_fetcher(monkeypatch):
    # No --start/--end -> _resolve_dates derives from years; obj=None forces the
    # build_price_fetcher() fallback in _fetcher (which we stub out).
    monkeypatch.setattr(opt_cli, "grid_search", lambda c, f, g, **k: [])

    import screener.backtester.data as data_mod

    monkeypatch.setattr(data_mod, "build_price_fetcher", lambda: StubPriceFetcher({}))
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "grid",
            "--tickers",
            "AAA",
            "--entry",
            "close > 0",
            "--stop-loss",
            "none",
            "--take-profit",
            "none",
            "--trailing-stop",
            "none",
            "--hold",
            "5",
        ],
        obj=None,
    )
    assert res.exit_code == 0, res.output


def _write_trades_csv(path: Path) -> None:
    path.write_text(
        "ticker,rank,signal_date,entry_date,entry_price,exit_date,exit_price,"
        "exit_reason,shares,entry_cost,exit_value,pnl,return_pct\n"
        "AAA,1,2024-01-01,2024-01-02,100,2024-01-05,110,time,1,100,110,10,0.1\n"
        "BBB,2,2024-02-01,2024-02-02,50,2024-02-05,45,stop,1,50,45,-5,-0.1\n"
    )


def test_load_trades_csv(tmp_path):
    path = tmp_path / "trades.csv"
    _write_trades_csv(path)
    trades = opt_cli._load_trades(path)
    assert len(trades) == 2
    assert trades[0].ticker == "AAA"
    assert trades[1].exit_reason == "stop"


def test_load_trades_json_dict_and_list(tmp_path):
    rows = [
        {
            "ticker": "AAA",
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-05",
            "entry_price": 100,
            "exit_price": 110,
            "pnl": 10,
            "return_pct": 0.1,
        }
    ]
    # dict-with-"trades" form
    dict_path = tmp_path / "d.json"
    dict_path.write_text(json.dumps({"trades": rows}))
    assert len(opt_cli._load_trades(dict_path)) == 1
    # bare list form (defaults applied: rank from idx, signal_date from entry)
    list_path = tmp_path / "l.json"
    list_path.write_text(json.dumps(rows))
    loaded = opt_cli._load_trades(list_path)
    assert loaded[0].rank == 1
    assert loaded[0].signal_date == date(2024, 1, 2)


def test_optimize_validate_command(tmp_path):
    csv_path = tmp_path / "trades.csv"
    _write_trades_csv(csv_path)
    json_path = tmp_path / "mc.json"
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "validate",
            "--trades",
            str(csv_path),
            "--iterations",
            "100",
            "--seed",
            "1",
            "--json",
            str(json_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Monte Carlo Validation" in res.output
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["iterations"] == 100
