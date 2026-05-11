"""Simple moving average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.registry import indicator


@indicator("sma")
def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()
