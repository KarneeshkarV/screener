"""MACD Oscillator strategy using 10 and 21 period simple moving averages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.sma import sma as _sma
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_macd_oscillator(df: pd.DataFrame) -> list[ResearchTrade]:
    cl = df["close"].to_numpy(dtype=float)

    ma1 = _sma(cl, 10)
    ma2 = _sma(cl, 21)

    ma1_prev = np.concatenate(([ma1[0]], ma1[:-1]))
    ma2_prev = np.concatenate(([ma2[0]], ma2[:-1]))

    entries = (ma1_prev <= ma2_prev) & (ma1 > ma2)
    exits = (ma1_prev >= ma2_prev) & (ma1 < ma2)

    return _walk(entries, exits, cl, df["date"].values)


# The rule now lives here once, as the expression both the backtester and the
# pine_runner evaluate. The function above is kept unregistered as the
# reference body that tests/test_bucket_a_parity.py compares against.
register_expression_strategy(
    "macd_oscillator",
    entry="crossover(sma(close, 10), sma(close, 21))",
    exit="crossunder(sma(close, 10), sma(close, 21))",
    profile=DEFAULT_STRATEGY_PROFILE,
)
