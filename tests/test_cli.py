"""CLI smoke tests — offline, no network."""

from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd
from click.testing import CliRunner

from main import cli
from screener.backtester import historical as historical_cli
from screener.backtester.models import BacktestResult
from screener.backtester.optimization import cli as optimize_cli
from screener.cli import cli as package_cli

from tests.conftest import StubPriceFetcher, make_bars


def test_help_includes_backtest_historical():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "--config" in res.output
    assert "backtest-historical" in res.output
    assert "backtest-rolling" in res.output


def test_package_cli_matches_main_wrapper():
    runner = CliRunner()
    main_res = runner.invoke(cli, ["--help"])
    package_res = runner.invoke(package_cli, ["--help"])

    assert main_res.exit_code == 0
    assert package_res.exit_code == 0
    for command in [
        "screen",
        "rs-breakout",
        "promoter-buys",
        "unusual-volume",
        "backtest-historical",
        "backtest-rolling",
        "operator-scan",
        "optimize",
    ]:
        assert command in main_res.output
        assert command in package_res.output


def test_backtest_help_lists_flags():
    runner = CliRunner()
    res = runner.invoke(cli, ["backtest-historical", "--help"])
    assert res.exit_code == 0
    for flag in [
        "--market",
        "--as-of",
        "--hold",
        "--top",
        "--entry",
        "--exit",
        "--stop-loss",
        "--take-profit",
        "--trailing-stop",
        "--slippage-bps",
        "--commission-bps",
        "--initial-capital",
        "--benchmark",
        "--csv",
        "--strategy",
        "--tickers",
    ]:
        assert flag in res.output, f"missing flag in help: {flag}"


def test_rolling_backtest_help_lists_core_flags():
    runner = CliRunner()
    res = runner.invoke(cli, ["backtest-rolling", "--help"])
    assert res.exit_code == 0
    for flag in [
        "--market",
        "--start",
        "--end",
        "--years",
        "--strategy",
        "--entry",
        "--universe",
        "--tickers",
        "--hold",
        "--top",
        "--csv",
    ]:
        assert flag in res.output, f"missing flag in help: {flag}"


def _stub_env():
    bars_a = make_bars(n=60, seed=11, open_base=100.0)
    bars_b = make_bars(n=60, seed=12, open_base=50.0)
    spy = make_bars(n=60, seed=99, open_base=400.0)
    return StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy}), bars_a


def test_offline_run_with_injected_fetcher():
    fetcher, bars_a = _stub_env()
    runner = CliRunner()
    as_of = bars_a.index[39].date()
    res = runner.invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            as_of.isoformat(),
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--initial-capital",
            "10000",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Total Return" in res.output
    assert "Performance" in res.output


def test_csv_flag_emits_trade_ledger():
    fetcher, bars_a = _stub_env()
    runner = CliRunner()
    as_of = bars_a.index[39].date()
    res = runner.invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            as_of.isoformat(),
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--csv",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(io.StringIO(res.output))
    for col in [
        "ticker",
        "rank",
        "signal_date",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "exit_reason",
        "shares",
        "pnl",
        "return_pct",
    ]:
        assert col in df.columns
    # rank must preserve selection ordering (1, 2)
    assert sorted(df["rank"].tolist()) == list(range(1, len(df) + 1))


def _minimal_result(cfg):
    equity = pd.Series([cfg.initial_capital], index=pd.to_datetime([cfg.as_of]))
    return BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=equity,
        benchmark_curve=equity,
        metrics={"total_return": 0.0, "trade_count": 0},
    )


def test_yaml_config_supplies_command_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_run_backtest(cfg, fetcher):
        captured["cfg"] = cfg
        return _minimal_result(cfg)

    monkeypatch.setattr(historical_cli, "run_backtest", fake_run_backtest)
    path = tmp_path / "screener.yaml"
    path.write_text(
        """
backtest-historical:
  market: us
  as_of: "2026-04-30"
  tickers: AAA,BBB
  entry_expr: close > sma(close, 3)
  hold: 7
  top: 2
  initial_capital: 12345
"""
    )

    res = CliRunner().invoke(cli, ["--config", str(path), "backtest-historical"])

    assert res.exit_code == 0, res.output
    cfg = captured["cfg"]
    assert cfg.as_of == date(2026, 4, 30)
    assert cfg.tickers == ("AAA", "BBB")
    assert cfg.hold == 7
    assert cfg.top == 2
    assert cfg.initial_capital == 12345


def test_json_config_supplies_command_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_run_backtest(cfg, fetcher):
        captured["cfg"] = cfg
        return _minimal_result(cfg)

    monkeypatch.setattr(historical_cli, "run_backtest", fake_run_backtest)
    path = tmp_path / "screener.json"
    path.write_text(
        json.dumps(
            {
                "backtest-historical": {
                    "market": "india",
                    "as_of": "2026-04-30",
                    "tickers": "AAA,BBB",
                    "entry_expr": "close > sma(close, 3)",
                    "hold": 9,
                    "top": 3,
                }
            }
        )
    )

    res = CliRunner().invoke(cli, ["--config", str(path), "backtest-historical"])

    assert res.exit_code == 0, res.output
    cfg = captured["cfg"]
    assert cfg.market == "india"
    assert cfg.hold == 9
    assert cfg.top == 3


def test_cli_options_override_config_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_run_backtest(cfg, fetcher):
        captured["cfg"] = cfg
        return _minimal_result(cfg)

    monkeypatch.setattr(historical_cli, "run_backtest", fake_run_backtest)
    path = tmp_path / "screener.yaml"
    path.write_text(
        """
backtest-historical:
  market: us
  as_of: "2026-04-30"
  tickers: AAA
  entry_expr: close > sma(close, 3)
  hold: 7
"""
    )

    res = CliRunner().invoke(
        cli,
        ["--config", str(path), "backtest-historical", "--hold", "5"],
    )

    assert res.exit_code == 0, res.output
    assert captured["cfg"].hold == 5


def test_config_rejects_missing_file():
    res = CliRunner().invoke(
        cli,
        ["--config", "missing.yaml", "backtest-historical"],
    )

    assert res.exit_code != 0
    assert "Config file not found" in res.output


def test_config_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "screener.txt"
    path.write_text("{}")

    res = CliRunner().invoke(
        cli,
        ["--config", str(path), "backtest-historical"],
    )

    assert res.exit_code != 0
    assert "Unsupported config file extension" in res.output


def test_nested_optimize_config_supplies_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_grid_search(cfg, fetcher, parameter_grid, **kwargs):
        captured["cfg"] = cfg
        captured["parameter_grid"] = parameter_grid
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(optimize_cli, "grid_search", fake_grid_search)
    path = tmp_path / "screener.yaml"
    path.write_text(
        """
optimize:
  grid:
    market: us
    end_arg: "2026-04-30"
    tickers: AAA,BBB
    entry_expr: close > sma(close, 3)
    hold: "5,10"
    top: 2
    metric: total_return
    top_n: 4
"""
    )

    res = CliRunner().invoke(cli, ["--config", str(path), "optimize", "grid"])

    assert res.exit_code == 0, res.output
    assert captured["cfg"].tickers == ("AAA", "BBB")
    assert captured["parameter_grid"]["hold"] == [5, 10]
    assert captured["kwargs"]["metric"] == "total_return"
    assert captured["kwargs"]["top_n"] == 4
