"""Dashboard rendering tests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.dashboard import dashboard_frames, render_dashboard
from screener.backtester.models import BacktestConfig, BacktestResult, Trade


def _cfg() -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
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
        strategy_name="ema_trend",
        tickers=("AAA", "BBB"),
    )


def _result() -> BacktestResult:
    idx = pd.bdate_range("2024-01-01", periods=45)
    equity = pd.Series([100_000 + i * 400 for i in range(len(idx))], index=idx)
    benchmark = pd.Series([100 + i * 0.2 for i in range(len(idx))], index=idx)
    trades = [
        Trade(
            ticker="AAA",
            rank=1,
            signal_date=date(2024, 1, 5),
            entry_date=date(2024, 1, 8),
            entry_price=100.0,
            exit_date=date(2024, 1, 12),
            exit_price=110.0,
            exit_reason="target",
            shares=10.0,
            entry_cost=1000.0,
            exit_value=1100.0,
            pnl=100.0,
            return_pct=0.10,
        ),
        Trade(
            ticker="BBB",
            rank=2,
            signal_date=date(2024, 1, 9),
            entry_date=date(2024, 1, 10),
            entry_price=50.0,
            exit_date=date(2024, 1, 18),
            exit_price=48.0,
            exit_reason="time",
            shares=20.0,
            entry_cost=1000.0,
            exit_value=960.0,
            pnl=-40.0,
            return_pct=-0.04,
        ),
    ]
    selection = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "signal_date": date(2024, 1, 5),
                "as_of_close": 99.0,
                "as_of_volume": 1_000_000,
                "as_of_dollar_vol": 99_000_000,
                "rank": 1,
                "role": "active",
            }
        ]
    )
    return BacktestResult(
        config=_cfg(),
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark,
        metrics={
            "starting_equity": 100_000.0,
            "final_equity": 117_600.0,
            "total_return": 0.176,
            "benchmark_return": 0.088,
            "max_drawdown": -0.02,
            "sharpe": 1.2,
            "trade_count": 2,
            "unique_tickers": 2,
            "hit_rate": 0.5,
        },
        warnings=["sample warning"],
        selection=selection,
    )


def test_dashboard_frames_include_curves_trades_and_selection():
    frames = dashboard_frames(_result())

    assert {"curves", "trades", "monthly", "selection"} == set(frames)
    assert "strategy_return" in frames["curves"].columns
    assert "drawdown" in frames["curves"].columns
    assert "holding_days" in frames["trades"].columns
    assert frames["selection"].iloc[0]["ticker"] == "AAA"


def test_render_dashboard_writes_expected_sections(tmp_path):
    path = render_dashboard(_result(), tmp_path)

    html = path.read_text(encoding="utf-8")
    assert path.exists()
    assert 'id="summary-metrics"' in html
    assert "Starting Capital" in html
    assert "Final Equity" in html
    assert 'id="performance-chart"' in html
    assert 'id="drawdown-chart"' in html
    assert 'id="monthly-returns"' in html
    assert 'id="trade-diagnostics"' in html
    assert 'id="selection-diagnostics"' in html
    assert 'id="trade-ledger-table"' in html
    assert 'id="equity-vs-benchmark" data-echart="equity-vs-benchmark-spec"' in html
    assert 'data-echart-group="perf"' in html
    assert 'id="drawdown-curve"' in html
    assert 'id="monthly-return-bars"' in html
    assert 'id="return-distribution"' in html
    assert 'id="exit-reason-breakdown"' in html
    assert 'id="holding-period-distribution"' in html
    assert 'id="ticker-contribution"' in html
    assert 'id="signal-count-by-day"' in html
    assert 'id="rank-distribution"' in html
    assert "Rolling backtest</div>" in html
    assert "plotly" not in html.lower()


def test_render_dashboard_chart_specs_use_theme_tokens(tmp_path):
    html = render_dashboard(_result(), tmp_path).read_text(encoding="utf-8")

    assert '"token:series-1"' in html
    assert '"token:series-2"' in html
    assert '"token:neg"' in html
    assert '"token:pos"' in html
    assert '"fmt:pct0"' in html
    assert "--paper" not in html
    assert "#0f766e" not in html
