"""Rolling population standard deviation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.registry import indicator


@indicator("stdev")
def stdev(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()
