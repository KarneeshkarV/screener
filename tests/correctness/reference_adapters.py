"""Reconciliation layer between the screener and independent references.

Every convention difference that would otherwise cause a *false* test failure
is encoded here once, so the test bodies stay declarative and the reasoning is
reviewable in a single place. A green test that goes through these adapters
means the screener's *math* agrees with the reference — not that conventions
happen to line up.

Reconciliation rules captured here (confirmed empirically against the installed
``pandas_ta_classic`` 0.6.x and ``talib`` 0.6.x):

* EMA / RSI / ATR / MACD use Wilder/EWM recursion with a different *seed* than
  the references; they only agree on the converged *tail*. Compare past a
  warm-up cutoff with a mask.
* Bollinger Bands: screener uses population std (ddof=0). TA-Lib BBANDS and
  ``pandas_ta_classic.bbands`` also use ddof=0 → exact match (no rescale).
* OBV: TA-Lib seeds OBV[0] = volume[0]; the screener's ``_obv`` starts the
  cumulative sum at 0. They differ by a constant ``volume[0]`` → compare
  first-differences, which removes the constant.
* Supertrend direction: the screener uses the inverted convention
  ``direction < 0 == uptrend``; ``pandas_ta_classic`` uses ``+1 == uptrend``.
  Compare ``screener_dir`` against ``-ref_dir``.
* empyrical Sharpe / annual-volatility use sample std (ddof=1); the screener
  uses population std (ddof=0). Convert with ``sqrt((N-1)/N)``.
"""

from __future__ import annotations

import numpy as np
import pytest


def require_talib():
    """Return the ``talib`` module or skip the test if it is not installed."""
    return pytest.importorskip("talib")


def require_quantstats():
    return pytest.importorskip("quantstats")


def finite_tail_mask(*arrays: np.ndarray, start: int) -> np.ndarray:
    """Boolean mask: index >= ``start`` AND every array finite at that index.

    Used to compare recursive indicators only on their converged tail, ignoring
    warm-up regions where seeding conventions legitimately differ.
    """
    length = len(arrays[0])
    mask = np.arange(length) >= start
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def ddof0_from_ddof1(value: float, n: int) -> float:
    """Convert a sample-std (ddof=1) statistic to a population-std (ddof=0) one.

    empyrical's Sharpe/volatility divide by ``std(ddof=1)``; the screener divides
    by ``std(ddof=0)``. Scaling by ``sqrt((n-1)/n)`` makes them comparable.
    """
    if n < 2:
        return value
    return value * np.sqrt((n - 1) / n)


def equity_to_returns(equity) -> np.ndarray:
    """Simple period returns from an equity curve, matching ``_daily_returns``."""
    import pandas as pd

    return pd.Series(equity).pct_change().dropna().to_numpy()
