"""Offline coverage tests for the backtester core/rolling/historical/data/pine modules.

These tests are deterministic and never touch the network: every price fetch goes
through ``StubPriceFetcher`` or a monkeypatched seam, and CLI paths use
``click.testing.CliRunner`` with an injected fetcher (``obj=...``).

They are written to drive the remaining uncovered lines in:
  - screener/backtester/core.py
  - screener/backtester/rolling.py
  - screener/backtester/historical.py
  - screener/backtester/data.py
  - screener/backtester/pine.py
"""

from __future__ import annotations


from datetime import date


import numpy as np


import pandas as pd


from click.testing import CliRunner


from main import cli


from screener.backtester.models import BacktestConfig


from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
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
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _stub_env(n=60):
    return StubPriceFetcher(
        {
            "AAA": make_bars(n=n, seed=11, open_base=100.0),
            "BBB": make_bars(n=n, seed=12, open_base=50.0),
            "SPY": make_bars(n=n, seed=99, open_base=400.0),
        }
    )


def _universe_file(tmp_path):
    f = tmp_path / "univ.txt"
    f.write_text("AAA\nBBB\n")
    return f


from screener.backtester.core import (  # noqa: E402
    _SlotState,
)


from screener.backtester.portfolio import Portfolio  # noqa: E402


def _open_slot(bars, *, entry_idx=1, ticker="AAA", **state_kw):
    """Build a portfolio with an open position + matching slot state."""
    cfg = _cfg(initial_capital=10_000.0)
    portfolio = Portfolio(cfg.initial_capital, 1)
    entry_fill = float(bars.iloc[entry_idx]["open"])
    portfolio.assign(ticker, 1, bars.index[0].date())
    portfolio.open(
        ticker=ticker,
        entry_date=bars.index[entry_idx].date(),
        entry_price=entry_fill,
        commission_bps=0.0,
    )
    defaults = dict(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=bars.index[entry_idx].date(),
        entry_fill=entry_fill,
        signal_date=bars.index[entry_idx - 1].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=entry_idx + 5,
        peak=entry_fill,
        exit_signal=None,
    )
    defaults.update(state_kw)
    state = _SlotState(**defaults)
    return cfg, portfolio, state


def _flat_then_trending(start, n, base, *, dip_at=None):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(base, base + n, n), index=idx, dtype=float)
    if dip_at is not None:
        close.iloc[dip_at] = base * 0.5  # crash to trip stop / free slot
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_rolling_cli_open_report(monkeypatch, tmp_path):
    import screener.reporting as reporting

    opened: list = []
    monkeypatch.setattr(reporting, "open_report", lambda p: opened.append(p))
    report = tmp_path / "openroll.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-20",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--report",
            str(report),
            "--open-report",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code == 0, res.output
    assert opened  # open_report was invoked
