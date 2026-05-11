"""Entry = ma_cross bullish; exit = supertrend flips bearish."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _ema, _supertrend_dir
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("ma_cross_st_exit")
def strat_ma_cross_st_exit(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (mfp <= msp) & (mf > ms)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)
