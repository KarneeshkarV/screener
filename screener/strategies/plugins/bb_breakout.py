"""Bollinger Band breakout strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.bollinger_bands import bollinger_bands as _bb
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("bb_breakout")
def strat_bb_breakout(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    _, s, upper = _bb(close, 350, 2.5)
    cp = np.concatenate(([close[0]], close[:-1]))
    up = np.concatenate(([upper[0]], upper[:-1]))
    sp = np.concatenate(([s[0]], s[:-1]))
    entries = (cp <= up) & (close > upper)
    exits = (cp >= sp) & (close < s)
    valid = ~np.isnan(upper)
    entries &= valid
    exits &= valid
    return _walk(entries, exits, close, df["date"].values)
