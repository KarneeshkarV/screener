"""RSI reversion: buy when RSI crosses above 30, sell when it crosses below 70."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.rsi import rsi as _rsi
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("rsi_reversion")
def strat_rsi_reversion(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    r = _rsi(close, 14)
    prev_r = np.concatenate(([r[0]], r[:-1]))

    entries = (r > 30) & (prev_r <= 30)
    exits = (r < 70) & (prev_r >= 70)

    return _walk(entries, exits, close, df["date"].values)
