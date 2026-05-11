"""RSI mean-reversion gated by EMA150 > EMA600 bull regime."""

from __future__ import annotations

import pandas as pd

from screener.indicators.numpy import _ema, _rsi
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("rsi_ema")
def strat_rsi_ema(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close, 14)
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (rsi < 30) & regime
    exits = rsi > 70
    return _walk(entries, exits, close, df["date"].values)
