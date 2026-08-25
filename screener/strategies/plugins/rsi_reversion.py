"""RSI reversion: buy when RSI crosses above 30, sell when it crosses below 70."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.rsi import rsi as _rsi
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_rsi_reversion(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    r = _rsi(close, 14)
    prev_r = np.concatenate(([r[0]], r[:-1]))

    entries = (r > 30) & (prev_r <= 30)
    exits = (r < 70) & (prev_r >= 70)

    return _walk(entries, exits, close, df["date"].values)


# The rule now lives here once, as the expression both the backtester and the
# pine_runner evaluate. The function above is kept unregistered as the
# reference body that tests/test_bucket_a_parity.py compares against.
register_expression_strategy(
    "rsi_reversion",
    entry="crossover(rsi(close, 14), 30)",
    exit="crossunder(rsi(close, 14), 70)",
    profile=DEFAULT_STRATEGY_PROFILE,
)
