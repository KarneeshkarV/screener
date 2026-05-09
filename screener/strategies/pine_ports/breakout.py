"""Breakout Pine strategy ports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _sma, _stdev
from screener.strategies.trades import Trade, _walk


def strat_bb_breakout(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    s = _sma(close, 350)
    sd = _stdev(close, 350)
    upper = s + 2.5 * sd
    cp = np.concatenate(([close[0]], close[:-1]))
    up = np.concatenate(([upper[0]], upper[:-1]))
    sp = np.concatenate(([s[0]], s[:-1]))
    entries = (cp <= up) & (close > upper)
    exits = (cp >= sp) & (close < s)
    valid = ~np.isnan(upper)
    entries &= valid
    exits &= valid
    return _walk(entries, exits, close, df["date"].values)
