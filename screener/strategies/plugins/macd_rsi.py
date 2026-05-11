"""MACD cross + RSI oversold/overbought confirmation (5-bar window)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _ema, _rsi
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("macd_rsi")
def strat_macd_rsi(df: pd.DataFrame) -> list[Trade]:
    """entry: MACD crosses over signal AND RSI was <= 30 in last 5 bars
    exit:  MACD crosses under signal AND RSI was >= 70 in last 5 bars
    """
    close = df["close"].to_numpy(dtype=float)
    macd = _ema(close, 12) - _ema(close, 26)
    sig = _ema(macd, 9)
    rsi = _rsi(close, 14)
    mp = np.concatenate(([macd[0]], macd[:-1]))
    sp = np.concatenate(([sig[0]], sig[:-1]))
    cross_over = (mp <= sp) & (macd > sig)
    cross_under = (mp >= sp) & (macd < sig)
    n = len(close)
    was_down = np.zeros(n, dtype=bool)
    was_up = np.zeros(n, dtype=bool)
    lookback = 5
    for i in range(1, n):
        lo = max(0, i - lookback)
        w = rsi[lo:i]
        if w.size:
            if np.any(w <= 30):
                was_down[i] = True
            if np.any(w >= 70):
                was_up[i] = True
    entries = cross_over & was_down
    exits = cross_under & was_up
    return _walk(entries, exits, close, df["date"].values)
