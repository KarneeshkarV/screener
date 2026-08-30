"""Awesome Oscillator strategy: trades momentum based on 5 and 34 SMA of (High+Low)/2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.sma import sma as _sma
from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


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


# The saucer legs need 1- and 2-bar lags of AO and of the bar colour, and Pine
# has no shift operator, so each lag is a declared column. AO itself is a
# column too: inlining it would repeat the same SMA pair four times across the
# two expressions.
register_expression_strategy(
    "awesome_oscillator",
    entry="ao > 0 or (open > close and green_p1 > 0 and green_p2 > 0 and ao_p1 > ao_p2 and ao_p1 < 0 and ao < 0)",
    exit="ao < 0 or (open < close and red_p1 > 0 and red_p2 > 0 and ao_p1 < ao_p2 and ao_p1 > 0 and ao > 0)",
    bar_columns={
        "ao": _cols.awesome_oscillator,
        "ao_p1": _cols.ao_prev1,
        "ao_p2": _cols.ao_prev2,
        "red_p1": _cols.red_prev1,
        "red_p2": _cols.red_prev2,
        "green_p1": _cols.green_prev1,
        "green_p2": _cols.green_prev2,
    },
    profile=DEFAULT_STRATEGY_PROFILE,
)
