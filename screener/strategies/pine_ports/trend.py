"""Trend-following Pine strategy ports."""

from __future__ import annotations

import pandas as pd
import numpy as np

from screener.indicators.numpy import _ema, _supertrend_dir
from screener.strategies.trades import Trade, _walk


def strat_supertrend(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (d < 0) & (dp >= 0)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_st_entry(df: pd.DataFrame) -> list[Trade]:
    """Entry = ma_cross AND supertrend bullish; exit = ma_cross bearish."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    entries = (mfp <= msp) & (mf > ms) & (d < 0)
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)
