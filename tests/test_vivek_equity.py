from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.engine import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.strategies import resolve_strategy
from screener.backtester.vivek_equity import (
    prepare_vivek_equity_tool_frame,
    required_history_bars,
)

from tests.conftest import StubPriceFetcher


def _vivek_bars() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=120)
    close = np.r_[
        np.full(45, 100.0),
        np.linspace(101.0, 140.0, 35),
        np.linspace(139.0, 90.0, 40),
    ]
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) + 0.5
    low = np.minimum(openp, close) - 0.5
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100_000.0,
        },
        index=idx,
    )


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 5),
        hold=80,
        top=1,
        entry_expr="vivek_equity_entry > 0",
        exit_expr="vivek_equity_exit > 0 or vivek_equity_close > 0",
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name="vivek_equity_tool",
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_vivek_equity_tool_generates_stateful_entry_and_exit():
    out = prepare_vivek_equity_tool_frame(_vivek_bars())

    entries = out.index[out["vivek_equity_entry"] > 0]
    exits = out.index[out["vivek_equity_exit"] > 0]

    assert len(entries) == 1
    assert len(exits) == 1
    assert entries[0] < exits[0]
    assert out.loc[entries[0], "vivek_equity_condition"] == 1.0
    assert out.loc[exits[0], "vivek_equity_condition"] == -1.0
    assert out.loc[entries[0], "vivek_equity_direction"] == 1.0


def test_vivek_equity_tool_is_causal_before_changed_future():
    bars = _vivek_bars()
    out = prepare_vivek_equity_tool_frame(bars)
    changed = bars.copy()
    changed.iloc[90:, changed.columns.get_loc("close")] += 500.0
    changed.iloc[90:, changed.columns.get_loc("open")] += 500.0
    changed.iloc[90:, changed.columns.get_loc("high")] += 500.0
    changed.iloc[90:, changed.columns.get_loc("low")] += 500.0
    out_changed = prepare_vivek_equity_tool_frame(changed)

    pd.testing.assert_series_equal(
        out["vivek_equity_entry"].iloc[:90],
        out_changed["vivek_equity_entry"].iloc[:90],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out["vivek_equity_exit"].iloc[:90],
        out_changed["vivek_equity_exit"].iloc[:90],
        check_names=False,
    )


def test_vivek_equity_named_strategy_resolves():
    strategy = resolve_strategy("vivek_equity_tool")

    assert strategy.entry == "vivek_equity_entry > 0"
    assert strategy.exit == "vivek_equity_exit > 0 or vivek_equity_close > 0"


def test_run_backtest_with_vivek_equity_tool_strategy():
    aaa = _vivek_bars()
    prepared = prepare_vivek_equity_tool_frame(aaa)
    signal_day = prepared.index[prepared["vivek_equity_entry"] > 0][0].date()
    spy = aaa.copy()
    fetcher = StubPriceFetcher({"AAA": aaa, "SPY": spy})
    strategy = resolve_strategy("vivek_equity_tool")

    result = run_backtest(
        _cfg(as_of=signal_day, entry_expr=strategy.entry, exit_expr=strategy.exit),
        fetcher,
    )

    assert result.selection["ticker"].tolist() == ["AAA"]
    assert result.trades
    assert result.trades[0].ticker == "AAA"
    assert result.trades[0].signal_date == signal_day
    assert result.trades[0].exit_reason == "exit_expr"


def test_vivek_equity_required_history_matches_trend_length():
    assert required_history_bars() == 40
