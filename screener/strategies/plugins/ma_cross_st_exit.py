"""Entry = ma_cross bullish; exit = supertrend flips bearish."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.ema import ema as _ema
from screener.indicators.plugins.supertrend import supertrend_dir as _supertrend_dir
from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_ma_cross_st_exit(df: pd.DataFrame) -> list[ResearchTrade]:
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


# One definition: the backtester evaluates this expression and the pine_runner
# gets a callable synthesised from it. The function above stays unregistered as
# the reference body tests/test_bucket_b_parity.py compares against.
register_expression_strategy(
    "ma_cross_st_exit",
    entry="crossover(ema(close, 10), ema(close, 20))",
    exit="crossover(st_dir, 0)",
    bar_columns={"st_dir": _cols.supertrend_direction},
    profile=DEFAULT_STRATEGY_PROFILE,
)
