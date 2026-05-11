from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester import pine_runner
from screener.strategies.pine_ports import (
    strat_ma_cross_st_entry,
)
from screener.strategies.registry import STRATEGIES, get_strategy, iter_strategies


def test_strategy_registry_preserves_pine_runner_names():
    expected = {
        "bb_breakout",
        "ma_cross",
        "ma_cross_regime",
        "ma_cross_st_entry",
        "ma_cross_st_exit",
        "macd_rsi",
        "rsi_ema",
        "supertrend",
        "supertrend_rsi",
    }

    assert set(STRATEGIES) == expected
    assert set(pine_runner.STRATEGIES) == expected
    assert dict(iter_strategies()) == STRATEGIES


def test_strategy_registry_lookup_returns_callable():
    strategy = get_strategy("ma_cross_st_entry")

    assert strategy is STRATEGIES["ma_cross_st_entry"]
    assert callable(strategy)


def test_backtester_pine_runner_reexports_legacy_helpers():
    assert pine_runner._ema is not None
    assert pine_runner._rsi is not None
    assert pine_runner.load_universe is not None


def _ohlcv(n: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    x = np.linspace(0, 18, n)
    close = 100 + np.linspace(0, 80, n) + np.sin(x) * 8
    high = close + 1.5
    low = close - 1.5
    open_ = close + np.sin(x / 2) * 0.5
    volume = np.full(n, 10_000.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


def test_ma_cross_st_entry_smoke():
    trades = strat_ma_cross_st_entry(_ohlcv())

    assert isinstance(trades, list)
    assert all(trade.entry_idx <= trade.exit_idx for trade in trades)
