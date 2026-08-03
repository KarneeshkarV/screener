"""Offline tests for the one-command research report orchestration."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.optimization.grid import GridSearchResult
from screener.backtester.optimization.monte_carlo import simulate_monte_carlo
from screener.backtester.optimization.research_report import (
    compute_parameter_stability,
    run_research_report,
)
from screener.backtester.optimization.walk_forward import WalkForwardSummary
from screener.cli import cli
from tests.conftest import StubPriceFetcher, make_bars

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


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
        "as_of": date(2024, 6, 1),
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
        "tickers": ("AAA", "BBB"),
    }
    values.update(overrides)
    return BacktestConfig(**values)


def test_parameter_stability_plateau_vs_spike():
    # hold=5 is clearly best (spike); top scores nearly identical (plateau).
    results = [
        GridSearchResult(
            params={"hold": 5, "top": 1},
            score=2.0,
            metrics={"sharpe": 2.0},
            trade_count=10,
        ),
        GridSearchResult(
            params={"hold": 5, "top": 2},
            score=1.95,
            metrics={"sharpe": 1.95},
            trade_count=10,
        ),
        GridSearchResult(
            params={"hold": 20, "top": 1},
            score=0.4,
            metrics={"sharpe": 0.4},
            trade_count=10,
        ),
        GridSearchResult(
            params={"hold": 20, "top": 2},
            score=0.35,
            metrics={"sharpe": 0.35},
            trade_count=10,
        ),
    ]
    stability = compute_parameter_stability(
        results,
        {"hold": [5, 20], "top": [1, 2]},
        metric="sharpe",
    )
    by_name = {row["parameter"]: row for row in stability}
    assert by_name["hold"]["shape"] == "spike"
    assert by_name["hold"]["best_value"] == 5
    assert by_name["hold"]["score_range"] == pytest.approx(1.6, abs=0.05)
    assert by_name["top"]["shape"] == "plateau"
    assert by_name["top"]["values_evaluated"] == 2


def test_parameter_stability_empty_and_errors_ignored():
    results = [
        GridSearchResult(
            params={"hold": 5},
            score=float("-inf"),
            metrics={},
            trade_count=0,
            error="boom",
        )
    ]
    stability = compute_parameter_stability(results, {"hold": [5, 10]})
    assert len(stability) == 1
    assert stability[0]["shape"] == "empty"
    assert stability[0]["values_evaluated"] == 0


def test_run_research_report_end_to_end_stub(tmp_path, monkeypatch):
    """Tiny synthetic universe + 2-point grid; asserts real module wiring."""
    bars_a = make_bars(n=120, seed=11, open_base=100.0, drift=0.05)
    bars_b = make_bars(n=120, seed=12, open_base=50.0, drift=0.03)
    spy = make_bars(n=120, seed=99, open_base=400.0, drift=0.02)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})

    start = bars_a.index[10].date()
    end = bars_a.index[110].date()
    cfg = _config(as_of=end)
    out = tmp_path / "reports" / "rs_report"

    import screener.backtester.optimization.grid as grid_mod
    import screener.backtester.optimization.monte_carlo as mc_mod
    import screener.backtester.optimization.research_report as rr
    import screener.backtester.optimization.walk_forward as wf_mod

    calls = {"grid": 0, "wf": 0, "mc": 0}
    real_grid = grid_mod.grid_search
    real_wf = wf_mod.walk_forward_optimize
    real_mc = mc_mod.simulate_monte_carlo

    def wrapped_grid(*args, **kwargs):
        calls["grid"] += 1
        return real_grid(*args, **kwargs)

    def wrapped_wf(*args, **kwargs):
        calls["wf"] += 1
        return real_wf(*args, **kwargs)

    def wrapped_mc(*args, **kwargs):
        calls["mc"] += 1
        return real_mc(*args, **kwargs)

    monkeypatch.setattr(rr, "grid_search", wrapped_grid)
    monkeypatch.setattr(rr, "walk_forward_optimize", wrapped_wf)
    monkeypatch.setattr(rr, "simulate_monte_carlo", wrapped_mc)

    payload = run_research_report(
        cfg,
        fetcher,
        {"hold": [5, 10]},
        start_date=start,
        end_date=end,
        train_days=40,
        test_days=15,
        step_days=15,
        metric="total_return",
        min_trades=1,
        max_workers=1,
        mc_iterations=50,
        mc_seed=7,
        top_n=2,
        out_path=out,
    )

    assert calls["grid"] == 1
    assert calls["wf"] == 1
    assert calls["mc"] == 1

    json_path = Path(str(out) + ".json")
    html_path = Path(str(out) + ".html")
    assert json_path.exists()
    assert html_path.exists()

    loaded = json.loads(json_path.read_text())
    for key in ("config", "grid", "walk_forward", "monte_carlo", "summary", "timings"):
        assert key in loaded
        assert key in payload

    assert "stability" in loaded["grid"]
    assert "results" in loaded["grid"]
    assert "best_params" in loaded["grid"]
    assert "warning" in loaded["grid"]
    assert "verdict" in loaded["summary"]
    assert "best_params" in loaded["summary"]
    assert "degradation" in loaded["summary"]
    assert "mc_return_p05" in loaded["summary"]
    assert "return_p05" in loaded["monte_carlo"]

    html = html_path.read_text().lower()
    assert "research report" in html
    assert "parameter stability" in html
    assert "walk-forward" in html
    assert "monte carlo" in html


def test_cli_research_report_offline(tmp_path):
    bars_a = make_bars(n=100, seed=21, open_base=100.0, drift=0.04)
    bars_b = make_bars(n=100, seed=22, open_base=50.0, drift=0.02)
    spy = make_bars(n=100, seed=99, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    out = tmp_path / "cli_report"
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "research-report",
            "--tickers",
            "AAA,BBB",
            "--start",
            bars_a.index[8].date().isoformat(),
            "--end",
            bars_a.index[90].date().isoformat(),
            "--entry",
            "close > sma(close, 3)",
            "--param",
            "hold=5,10",
            "--train-days",
            "30",
            "--test-days",
            "12",
            "--step-days",
            "12",
            "--mc-iterations",
            "40",
            "--workers",
            "1",
            "--metric",
            "total_return",
            "--out",
            str(out),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    plain = _plain(res.output)
    assert "Stage 1/3" in plain
    assert "Stage 2/3" in plain
    assert "Stage 3/3" in plain
    assert "Research Report Summary" in plain
    assert Path(str(out) + ".json").exists()
    assert Path(str(out) + ".html").exists()


def test_cli_optimize_research_report_alias(tmp_path):
    bars_a = make_bars(n=80, seed=3, open_base=100.0)
    spy = make_bars(n=80, seed=9, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "SPY": spy})
    out = tmp_path / "alias_report"
    res = CliRunner().invoke(
        cli,
        [
            "optimize",
            "research-report",
            "--tickers",
            "AAA",
            "--start",
            bars_a.index[5].date().isoformat(),
            "--end",
            bars_a.index[70].date().isoformat(),
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
            "--train-days",
            "25",
            "--test-days",
            "10",
            "--step-days",
            "10",
            "--mc-iterations",
            "20",
            "--workers",
            "1",
            "--out",
            str(out),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert Path(str(out) + ".json").exists()


def test_cli_help_lists_research_report():
    res = CliRunner().invoke(cli, ["optimize", "--help"])
    assert res.exit_code == 0
    assert "research-report" in res.output


def test_research_report_helper_edge_cases():
    import screener.backtester.optimization.research_report as rr
    from screener.backtester.metrics import format_result_value

    assert not rr._param_equal(None, 1)
    assert not rr._param_equal("not-a-number", 1.0)
    assert rr._value_key(1.25) == "1.25"
    assert rr._parse_value_key("1.25") == 1.25
    assert rr._parse_value_key("2") == 2
    assert rr._parse_value_key("value") == "value"
    assert rr._degradation_ratio(0.0, 0.0) == 0.0
    assert rr._degradation_ratio(0.0, -1.0) == 1.0
    assert rr._verdict(
        overfit_flag=True,
        degradation=0.0,
        mc_return_p05=1.0,
        oos_metric=1.0,
    ).startswith("FAIL: severe")
    assert rr._verdict(
        overfit_flag=False,
        degradation=0.0,
        mc_return_p05=-1.0,
        oos_metric=0.0,
    ).startswith("FAIL: weak")
    assert rr._verdict(
        overfit_flag=False,
        degradation=0.5,
        mc_return_p05=1.0,
        oos_metric=1.0,
    ).startswith("CAUTION")
    assert rr._verdict(
        overfit_flag=False,
        degradation=0.0,
        mc_return_p05=1.0,
        oos_metric=1.0,
    ).startswith("PASS")
    assert format_result_value(float("nan"), "ratio") == "-"


def test_run_research_report_fallback_ledger_and_empty_grid(tmp_path, monkeypatch):
    import screener.backtester.optimization.research_report as rr

    cfg = _config()
    best = GridSearchResult(
        params={"hold": 8},
        score=1.5,
        metrics={"sharpe": 1.5},
        trade_count=1,
    )
    empty_wf = WalkForwardSummary(
        windows=[],
        stability_score=1.0,
        aggregate_metrics={"sharpe": 0.5},
        overfit_flag=False,
        train_test_score_ratio=3.0,
    )
    wf_calls = []

    monkeypatch.setattr(rr, "grid_search", lambda *args, **kwargs: [best])

    def fake_wf(*args, **kwargs):
        wf_calls.append(kwargs)
        return empty_wf

    monkeypatch.setattr(rr, "walk_forward_optimize", fake_wf)
    monkeypatch.setattr(
        rr,
        "run_rolling_backtest",
        lambda *args, **kwargs: SimpleNamespace(trades=[_trade(10.0, 0.1)]),
    )
    monkeypatch.setattr(rr, "simulate_monte_carlo", simulate_monte_carlo)
    progress = []
    payload = run_research_report(
        cfg,
        StubPriceFetcher({}),
        {"hold": [8]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        cache_path=tmp_path / "grid.json",
        mc_iterations=2,
        out_path=tmp_path / "fallback.json",
        progress=progress.append,
    )

    assert progress == ["grid", "walk_forward", "monte_carlo"]
    assert wf_calls[0]["cache_path"].name == "grid_research_wf.json"
    assert payload["summary"]["is_metric"] == 1.5
    assert payload["monte_carlo"]["trade_source"] == "full_period"

    monkeypatch.setattr(rr, "grid_search", lambda *args, **kwargs: [])
    empty = run_research_report(
        cfg,
        StubPriceFetcher({}),
        {"hold": [8]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        mc_iterations=2,
        out_path=tmp_path / "empty.html",
    )
    assert empty["grid"]["best_params"] == {}
    assert empty["monte_carlo"]["trade_source"] == "none"


def test_research_report_reuses_walk_forward_oos_trades(tmp_path, monkeypatch):
    import screener.backtester.optimization.research_report as rr

    cfg = _config()
    best = GridSearchResult(
        params={"hold": 8},
        score=1.5,
        metrics={"sharpe": 1.5},
        trade_count=1,
    )
    oos_trade = _trade(10.0, 0.1)
    walk_forward = WalkForwardSummary(
        windows=[
            {
                "window": {
                    "train_start": date(2024, 1, 1),
                    "train_end": date(2024, 1, 31),
                    "test_start": date(2024, 2, 1),
                    "test_end": date(2024, 2, 15),
                },
                "best_train": best,
                "test_metrics": {"sharpe": 0.5},
                "test_trade_count": 1,
            }
        ],
        stability_score=1.0,
        aggregate_metrics={"sharpe": 0.5},
        overfit_flag=False,
        train_test_score_ratio=3.0,
        oos_trades=(oos_trade,),
    )
    captured_trades = []

    monkeypatch.setattr(rr, "grid_search", lambda *args, **kwargs: [best])
    monkeypatch.setattr(
        rr, "walk_forward_optimize", lambda *args, **kwargs: walk_forward
    )
    monkeypatch.setattr(
        rr,
        "run_rolling_backtest",
        lambda *args, **kwargs: pytest.fail("OOS backtest must not be re-run"),
    )

    def fake_monte_carlo(trades, **kwargs):
        captured_trades.extend(trades)
        return simulate_monte_carlo(trades, **kwargs)

    monkeypatch.setattr(rr, "simulate_monte_carlo", fake_monte_carlo)
    payload = run_research_report(
        cfg,
        StubPriceFetcher({}),
        {"hold": [8]},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 1),
        mc_iterations=2,
        out_path=tmp_path / "reuse",
    )

    assert captured_trades == [oos_trade]
    assert payload["monte_carlo"]["trade_count"] == 1
    assert payload["monte_carlo"]["trade_source"] == "walk_forward_oos"
    assert "oos_trades" not in payload["walk_forward"]


def test_resolve_universe_tickers_branches(monkeypatch):
    import screener.backtester.optimization.cli as optimization_cli
    from screener import universes

    assert optimization_cli._resolve_universe_tickers(
        market="us",
        tickers=None,
        universe_file=None,
        universe=None,
        end_date=date(2024, 1, 1),
        max_universe=0,
    ) == (None, None)

    monkeypatch.setattr(
        universes,
        "load_current_universe",
        lambda *args, **kwargs: SimpleNamespace(symbols=("AAA", "BBB")),
    )
    assert optimization_cli._resolve_universe_tickers(
        market="us",
        tickers=None,
        universe_file=None,
        universe="sp500",
        end_date=date(2024, 1, 1),
        max_universe=1,
    ) == ("AAA", None)

    monkeypatch.setattr(
        universes,
        "load_current_universe",
        lambda *args, **kwargs: SimpleNamespace(symbols=()),
    )
    with pytest.raises(click.UsageError, match="zero symbols"):
        optimization_cli._resolve_universe_tickers(
            market="us",
            tickers=None,
            universe_file=None,
            universe="sp500",
            end_date=date(2024, 1, 1),
            max_universe=0,
        )


def test_backtest_config_rejects_unknown_interval():
    with pytest.raises(ValueError, match="unsupported interval"):
        _config(interval="2h")
