"""Supertrend long-only flip strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _supertrend_dir
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("supertrend")
def strat_supertrend(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (d < 0) & (dp >= 0)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)
