from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.trend_technical import _kama


def test_kama_recovers_after_flat_tape() -> None:
    # A halted / limit-locked tape (10+ identical closes) makes the efficiency
    # ratio 0/0 (NaN); without a zero-efficiency fallback the recursion would
    # stay NaN forever and kama_trend would silently stop generating entries.
    closes = [100.0] * 15 + list(np.linspace(100.0, 200.0, 49))
    kama = _kama(pd.Series(closes, dtype=float))
    assert not np.isnan(kama[-1])
