"""Supertrend long-only flip strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.supertrend import supertrend_dir as _supertrend_dir
from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_supertrend(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (d < 0) & (dp >= 0)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)


# One definition: the backtester evaluates this expression and the pine_runner
# gets a callable synthesised from it. The function above stays unregistered as
# the reference body tests/test_bucket_b_parity.py compares against.
register_expression_strategy(
    "supertrend",
    entry="crossunder(st_dir, 0)",
    exit="crossover(st_dir, 0)",
    bar_columns={"st_dir": _cols.supertrend_direction},
    profile=DEFAULT_STRATEGY_PROFILE,
)
