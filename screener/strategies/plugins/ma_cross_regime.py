"""ma_cross entries gated by EMA150 > EMA600 bull regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.numpy import _ema
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk


@strategy("ma_cross_regime")
def strat_ma_cross_regime(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (mfp <= msp) & (mf > ms) & regime
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)
