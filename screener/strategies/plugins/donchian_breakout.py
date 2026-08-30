"""Donchian channel breakout: entry above the 20-bar high, exit below the 10-bar low."""

from __future__ import annotations

import pandas as pd

from screener.strategies import bar_column_recipes as _cols
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_donchian_breakout(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    prior_high = df["high"].rolling(20).max().shift(1)
    prior_low = df["low"].rolling(10).min().shift(1)

    entries = (df["close"] > prior_high).to_numpy()
    exits = (df["close"] < prior_low).to_numpy()

    return _walk(entries, exits, close, df["date"].values)


# One definition: the backtester evaluates this expression and the pine_runner
# gets a callable synthesised from it. The function above stays unregistered as
# the reference body tests/test_bucket_b_parity.py compares against.
register_expression_strategy(
    "donchian_breakout",
    entry="close > dc_prior_high",
    exit="close < dc_prior_low",
    bar_columns={
        "dc_prior_high": _cols.donchian_prior_high_20,
        "dc_prior_low": _cols.donchian_prior_low_10,
    },
    profile=DEFAULT_STRATEGY_PROFILE,
)
