"""Parabolic SAR strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.sar import sar as _sar
from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_parabolic_sar(df: pd.DataFrame) -> list[ResearchTrade]:
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)

    real_sar = _sar(hi, lo, cl)

    # "positions = np.where(new['real sar']<new['Close'],1,0)"
    # shift condition to avoid lookahead (trade next open or evaluate at close using prior state)
    # The original calculates 'real_sar' with current bar, so comparing current 'real_sar' to current 'close' uses current data.
    # But wait, original code:
    # positions[i] = real_sar[i] < close[i]
    # signals = positions.diff()
    # It trades *based on the cross of current real_sar and current close*.
    # We delay the condition by 1 bar to strictly avoid lookahead in the cross:
    # Entries when real_sar was >= close, but now < close.
    # Exits when real_sar was < close, but now >= close.

    below = real_sar < cl
    below_prev = np.concatenate(([False], below[:-1]))

    entries = (~below_prev) & below
    exits = below_prev & (~below)

    return _walk(entries, exits, cl, df["date"].values)


# One definition: the backtester evaluates this expression and the pine_runner
# gets a callable synthesised from it. The function above stays unregistered as
# the reference body tests/test_bucket_b_parity.py compares against.
register_expression_strategy(
    "parabolic_sar",
    entry="crossunder(sar, close)",
    exit="crossover(sar, close)",
    bar_columns={"sar": _cols.parabolic_sar},
    profile=DEFAULT_STRATEGY_PROFILE,
)
