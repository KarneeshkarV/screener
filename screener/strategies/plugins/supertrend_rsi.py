"""Supertrend bullish AND RSI cross above 50; exit on RSI > 72 or flip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _rsi, _supertrend_dir
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("supertrend_rsi")
def strat_supertrend_rsi(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    rsi = _rsi(close, 14)
    in_long = d < 0
    rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
    entries = in_long & (rsi_prev < 50) & (rsi > 50)
    dp = np.concatenate(([d[0]], d[:-1]))
    flip_down = (d > 0) & (dp <= 0)
    exits = (rsi > 72) | flip_down
    return _walk(entries, exits, close, df["date"].values)
