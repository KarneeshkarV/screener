"""Awesome Oscillator strategy: trades momentum based on 5 and 34 SMA of (High+Low)/2."""

from __future__ import annotations
import numpy as np
import pandas as pd
from screener.indicators.plugins.sma import sma as _sma
from screener.strategies.spec import strategy
from screener.strategies.trades import ResearchTrade, _walk


@strategy("awesome_oscillator")
def strat_awesome_oscillator(df: pd.DataFrame) -> list[ResearchTrade]:
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)

    mid = (hi + lo) / 2.0
    ma1 = _sma(mid, 5)
    ma2 = _sma(mid, 34)
    ao = ma1 - ma2

    def shift1(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        return np.concatenate(([fill_value], arr[:-1]))

    def shift2(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        return np.concatenate(([fill_value, fill_value], arr[:-2]))

    ao_s1 = shift1(ao)
    ao_s2 = shift2(ao)

    red_bar = op > cl
    green_bar = op < cl

    red_s1 = shift1(red_bar, fill_value=False)
    red_s2 = shift2(red_bar, fill_value=False)
    green_s1 = shift1(green_bar, fill_value=False)
    green_s2 = shift2(green_bar, fill_value=False)

    saucer_long = (
        red_bar & green_s1 & green_s2 & (ao_s1 > ao_s2) & (ao_s1 < 0) & (ao < 0)
    )

    saucer_short = (
        green_bar & red_s1 & red_s2 & (ao_s1 < ao_s2) & (ao_s1 > 0) & (ao > 0)
    )

    cross_long = ma1 > ma2
    cross_short = ma1 < ma2

    entries = saucer_long | cross_long
    exits = saucer_short | cross_short

    return _walk(entries, exits, cl, df["date"].values)
