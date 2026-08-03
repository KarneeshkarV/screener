"""Donchian channel breakout: entry above the 20-bar high, exit below the 10-bar low."""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import strategy
from screener.strategies.trades import ResearchTrade, _walk


@strategy("donchian_breakout")
def strat_donchian_breakout(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    prior_high = df["high"].rolling(20).max().shift(1)
    prior_low = df["low"].rolling(10).min().shift(1)

    entries = (df["close"] > prior_high).to_numpy()
    exits = (df["close"] < prior_low).to_numpy()

    return _walk(entries, exits, close, df["date"].values)
