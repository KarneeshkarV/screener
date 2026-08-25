"""Bollinger Band breakout strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.bollinger_bands import bollinger_bands as _bb
from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_bb_breakout(df: pd.DataFrame) -> list[ResearchTrade]:
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


# One definition: the backtester evaluates this expression and the pine_runner
# gets a callable synthesised from it. The function above stays unregistered as
# the reference body tests/test_bucket_b_parity.py compares against.
register_expression_strategy(
    "bb_breakout",
    entry="crossover(close, bb_upper)",
    exit="crossunder(close, bb_mid)",
    bar_columns={
        "bb_upper": _cols.bb_upper_350,
        "bb_mid": _cols.bb_mid_350,
    },
    profile=DEFAULT_STRATEGY_PROFILE,
)
