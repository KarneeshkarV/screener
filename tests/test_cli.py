"""CLI smoke tests — offline, no network."""
from __future__ import annotations

import io

import pandas as pd
from click.testing import CliRunner

from screener.cli import cli
from tests.conftest import StubPriceFetcher, make_bars


def test_help_includes_backtest_historical():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "backtest-historical" in res.output


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
    for col in ["ticker", "rank", "signal_date", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_reason", "shares", "pnl", "return_pct"]:
        assert col in df.columns
    # rank must preserve selection ordering (1, 2)
    assert sorted(df["rank"].tolist()) == list(range(1, len(df) + 1))
