"""HTML tear-sheet rendering tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
from click.testing import CliRunner

from screener.backtester.historical import backtest_historical
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import backtest_rolling, run_rolling_backtest
from screener.backtester.tearsheet import render_tearsheet

SECTION_MARKERS = [
    'id="metrics-summary"',
    'id="equity-vs-benchmark"',
    'id="drawdown-curve"',
    'id="monthly-heatmap"',
    'id="trade-timeline"',
    'id="tab-ledger"',
    'id="trade-ledger"',
    'id="trade-ledger-table"',
    'id="trade-histogram"',
    'id="winners-losers"',
    'id="top-winners-table"',
    'id="top-losers-table"',
    'id="config"',
    'id="warnings"',
]


def _trend_bars(start: str = "2024-01-01", n: int = 65) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    close = pd.Series([100.0 + i for i in range(n)], index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _stub_data() -> dict[str, pd.DataFrame]:
    return {"AAA": _trend_bars(), "BBB": _trend_bars(), "SPY": _trend_bars()}


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 28),
        hold=3,
        top=2,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB"),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_render_tearsheet_from_engine_result(tmp_path, stub_fetcher_factory):
    fetcher = stub_fetcher_factory(_stub_data())
    result = run_rolling_backtest(
        _cfg(),
        fetcher,
        start_date=date(2024, 1, 15),
        end_date=date(2024, 3, 28),
    )
    assert result.trades, "engine should produce trades on trending stub data"

    out = tmp_path / "reports" / "tearsheet.html"
    path = render_tearsheet(
        result, out, extra_notes=["survivorship bias: today's members applied"]
    )

    assert path == out
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    for marker in SECTION_MARKERS:
        assert marker in html, f"missing section marker {marker}"
    assert "survivorship bias: today&#x27;s members applied" in html
    assert "close &gt; sma(close, 3)" in html
    assert "Median Trade" in html
    assert "--paper: #07090d" in html
    assert '"plot_bgcolor":"#0d1117"' in html


def test_backtest_rolling_report_option(tmp_path, stub_fetcher_factory):
    fetcher = stub_fetcher_factory(_stub_data())
    report = tmp_path / "rolling.html"
    runner = CliRunner()
    result = runner.invoke(
        backtest_rolling,
        [
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "3",
            "--top",
            "2",
            "--start",
            "2024-01-15",
            "--end",
            "2024-03-28",
            "--report",
            str(report),
        ],
        obj=fetcher,
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    for marker in SECTION_MARKERS:
        assert marker in html, f"missing section marker {marker}"


def test_backtest_historical_report_option(tmp_path, stub_fetcher_factory):
    fetcher = stub_fetcher_factory(_stub_data())
    report = tmp_path / "historical.html"
    runner = CliRunner()
    result = runner.invoke(
        backtest_historical,
        [
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "3",
            "--top",
            "2",
            "--as-of",
            "2024-02-15",
            "--report",
            str(report),
        ],
        obj=fetcher,
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    for marker in SECTION_MARKERS:
        assert marker in html, f"missing section marker {marker}"
    assert "survivorship bias" in html


def test_backtest_historical_auto_temp_report(
    tmp_path, monkeypatch, stub_fetcher_factory
):
    fetcher = stub_fetcher_factory(_stub_data())
    report = tmp_path / "auto-historical.html"
    monkeypatch.setattr("screener.reporting.temp_report_path", lambda prefix: report)
    runner = CliRunner()
    result = runner.invoke(
        backtest_historical,
        [
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "3",
            "--top",
            "2",
            "--as-of",
            "2024-02-15",
        ],
        obj=fetcher,
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert f"Report: {report}" in result.output
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert 'id="trade-timeline"' in html
    assert "Median Trade" in html


def test_backtest_historical_csv_skips_implicit_temp_report(
    tmp_path, monkeypatch, stub_fetcher_factory
):
    fetcher = stub_fetcher_factory(_stub_data())
    report = tmp_path / "should-not-exist.html"
    monkeypatch.setattr("screener.reporting.temp_report_path", lambda prefix: report)
    runner = CliRunner()
    result = runner.invoke(
        backtest_historical,
        [
            "--tickers",
            "AAA,BBB",
            "--entry",
            "close > sma(close, 3)",
            "--hold",
            "3",
            "--top",
            "2",
            "--as-of",
            "2024-02-15",
            "--csv",
        ],
        obj=fetcher,
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Report:" not in result.output
    assert not report.exists()
