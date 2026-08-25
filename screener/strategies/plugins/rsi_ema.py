"""RSI mean-reversion gated by EMA150 > EMA600 bull regime."""

from __future__ import annotations

import pandas as pd

from screener.indicators.plugins.ema import ema as _ema
from screener.indicators.plugins.rsi import rsi as _rsi
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_rsi_ema(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close, 14)
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (rsi < 30) & regime
    exits = rsi > 70
    return _walk(entries, exits, close, df["date"].values)


# The rule now lives here once, as the expression both the backtester and the
# pine_runner evaluate. The function above is kept unregistered as the
# reference body that tests/test_bucket_a_parity.py compares against.
register_expression_strategy(
    "rsi_ema",
    entry="rsi(close, 14) < 30 and ema(close, 150) > ema(close, 600)",
    exit="rsi(close, 14) > 70",
    profile=DEFAULT_STRATEGY_PROFILE,
)
