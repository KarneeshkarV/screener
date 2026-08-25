"""ma_cross entries gated by EMA150 > EMA600 bull regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.ema import ema as _ema
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)
from screener.strategies.trades import ResearchTrade, _walk


def strat_ma_cross_regime(df: pd.DataFrame) -> list[ResearchTrade]:
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (mfp <= msp) & (mf > ms) & regime
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


# The rule now lives here once, as the expression both the backtester and the
# pine_runner evaluate. The function above is kept unregistered as the
# reference body that tests/test_bucket_a_parity.py compares against.
register_expression_strategy(
    "ma_cross_regime",
    entry="crossover(ema(close, 10), ema(close, 20)) and ema(close, 150) > ema(close, 600)",
    exit="crossunder(ema(close, 10), ema(close, 20))",
    profile=DEFAULT_STRATEGY_PROFILE,
)
