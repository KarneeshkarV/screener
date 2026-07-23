"""Interval-aware cost defaults.

Spread is a much larger fraction of a fine bar's range than of a daily bar's,
so the CLI resolves unset ``--slippage-bps``/``--commission-bps`` through
per-interval default tables. Daily defaults stay 0.0 (byte-identical legacy
runs); intraday slippage rises as bars get finer. The Corwin-Schultz spread
estimator's rolling window scales by bars-per-session so it always averages
~21 sessions.
"""

from __future__ import annotations

from click.testing import CliRunner

from screener.backtester import rolling as rolling_cli
from screener.backtester.cli_common import resolve_interval_cost_defaults
from screener.backtester.costs import (
    DEFAULT_COMMISSION_BPS_BY_INTERVAL,
    corwin_schultz_half_spread,
    default_commission_bps,
    default_spread_window,
)
from screener.backtester.slippage import (
    DEFAULT_SLIPPAGE_BPS_BY_INTERVAL,
    default_slippage_bps,
)
from screener.cli import cli
from tests.conftest import StubPriceFetcher


def test_default_slippage_bps_table():
    assert DEFAULT_SLIPPAGE_BPS_BY_INTERVAL == {
        "1d": 0.0,
        "1h": 2.0,
        "30m": 3.0,
        "15m": 5.0,
        "5m": 7.0,
        "1m": 10.0,
    }
    assert default_slippage_bps("1d") == 0.0  # legacy default preserved
    assert default_slippage_bps("1m") == 10.0
    assert default_slippage_bps("2h") == 0.0  # unknown interval falls back to 0


def test_default_commission_bps_is_zero_at_every_interval():
    assert set(DEFAULT_COMMISSION_BPS_BY_INTERVAL.values()) == {0.0}
    assert default_commission_bps("1m") == 0.0
    assert default_commission_bps("1d") == 0.0
    assert default_commission_bps("2h") == 0.0


def test_default_spread_window_scales_by_bars_per_session():
    assert default_spread_window("1d") == 21  # legacy window preserved
    assert default_spread_window("1h") == 21 * 7
    assert default_spread_window("15m") == 21 * 26
    assert default_spread_window("1m") == 21 * 390
    assert default_spread_window("2h") == 21  # unknown interval → daily window


def test_corwin_schultz_accepts_scaled_window():
    import numpy as np
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=40, freq="D")
    high = pd.Series(np.linspace(10.0, 12.0, 40) + 0.1, index=index)
    low = pd.Series(np.linspace(10.0, 12.0, 40) - 0.1, index=index)
    daily = corwin_schultz_half_spread(high, low, window=default_spread_window("1d"))
    legacy = corwin_schultz_half_spread(high, low)
    assert daily.equals(legacy)  # 1d default reproduces the legacy call exactly


def test_resolve_interval_cost_defaults():
    # Unset flags resolve through the interval tables.
    assert resolve_interval_cost_defaults("1d", None, None) == (0.0, 0.0)
    assert resolve_interval_cost_defaults("15m", None, None) == (5.0, 0.0)
    assert resolve_interval_cost_defaults("1m", None, None) == (10.0, 0.0)
    # Explicit flags always win.
    assert resolve_interval_cost_defaults("1m", 3.5, 1.25) == (3.5, 1.25)
    assert resolve_interval_cost_defaults("1d", 0.0, 0.0) == (0.0, 0.0)


def _capture_rolling_config(monkeypatch) -> list:
    captured = []

    def fake_run(cfg, fetcher, *, start_date, end_date, earnings_blackout=None):
        captured.append(cfg)
        from screener.backtester.models import BacktestResult

        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=__import__("pandas").Series(dtype=float),
            benchmark_curve=__import__("pandas").Series(dtype=float),
            metrics={},
        )

    monkeypatch.setattr(rolling_cli, "run_rolling_backtest", fake_run)
    return captured


def _rolling_args(interval: str, *extra: str) -> list[str]:
    return [
        "backtest-rolling",
        "-m",
        "us",
        "--tickers",
        "AAPL",
        "--entry",
        "close > 0",
        "--interval",
        interval,
        "--start",
        "2024-03-04",
        "--end",
        "2024-03-05",
        "--hold",
        "2",
        *extra,
    ]


def test_rolling_cli_intraday_default_slippage(monkeypatch):
    captured = _capture_rolling_config(monkeypatch)
    result = CliRunner().invoke(
        cli, _rolling_args("15m"), obj=StubPriceFetcher({}), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert captured[0].slippage_bps == 5.0
    assert captured[0].commission_bps == 0.0


def test_rolling_cli_daily_default_slippage_stays_zero(monkeypatch):
    captured = _capture_rolling_config(monkeypatch)
    result = CliRunner().invoke(
        cli, _rolling_args("1d"), obj=StubPriceFetcher({}), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert captured[0].slippage_bps == 0.0


def test_rolling_cli_explicit_slippage_wins(monkeypatch):
    captured = _capture_rolling_config(monkeypatch)
    result = CliRunner().invoke(
        cli,
        _rolling_args("1m", "--slippage-bps", "1.5", "--commission-bps", "2.5"),
        obj=StubPriceFetcher({}),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert captured[0].slippage_bps == 1.5
    assert captured[0].commission_bps == 2.5
