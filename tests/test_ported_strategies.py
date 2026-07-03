from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.registry import STRATEGIES

PORTED_STRATEGIES = [
    "awesome_oscillator",
    "bb_pattern",
    "heikin_ashi",
    "macd_oscillator",
    "parabolic_sar",
    "rsi_pattern",
    "shooting_star",
]

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

def test_ported_strategies_registered():
    for name in PORTED_STRATEGIES:
        assert name in STRATEGIES

def test_ported_strategies_smoke():
    df = _ohlcv()
    
    for name in PORTED_STRATEGIES:
        strategy_fn = STRATEGIES[name]
        trades = strategy_fn(df)
        
        assert isinstance(trades, list), f"{name} returned {type(trades)}"
        for trade in trades:
            assert trade.entry_idx <= trade.exit_idx, f"{name}: entry > exit"
