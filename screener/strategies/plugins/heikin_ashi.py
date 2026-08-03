"""Heikin-Ashi momentum strategy based on HA candlestick patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import strategy
from screener.strategies.trades import ResearchTrade, _walk


@strategy("heikin_ashi")
def strat_heikin_ashi(df: pd.DataFrame) -> list[ResearchTrade]:
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)

    n_len = len(cl)
    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.zeros_like(cl)
    if n_len > 0:
        ha_open[0] = op[0]
        for i in range(1, n_len):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([ha_open, ha_close, hi])
    ha_low = np.minimum.reduce([ha_open, ha_close, lo])

    stls = 3
    entries = np.zeros(n_len, dtype=bool)
    exits = np.zeros(n_len, dtype=bool)

    cumsum = 0

    for i in range(1, n_len):
        # Entry
        if (
            ha_open[i] > ha_close[i]
            and ha_open[i] == ha_high[i]
            and np.abs(ha_open[i] - ha_close[i])
            > np.abs(ha_open[i - 1] - ha_close[i - 1])
            and ha_open[i - 1] > ha_close[i - 1]
        ):
            cumsum += 1
            if cumsum > stls:
                cumsum -= 1
            else:
                entries[i] = True

        # Exit
        elif (
            ha_open[i] < ha_close[i]
            and ha_open[i] == ha_low[i]
            and ha_open[i - 1] < ha_close[i - 1]
        ):
            if cumsum > 0:
                exits[i] = True
                cumsum = 0

    return _walk(entries, exits, cl, df["date"].values)
