"""Average True Range via Wilder smoothing."""

from __future__ import annotations

from typing import cast

import numpy as np

from screener.indicators.plugins.rma import rma
from screener.indicators.registry import indicator


@indicator("atr")
def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14
) -> np.ndarray:
    """True range, Wilder-smoothed. Accepts one series or a ``(bars, symbols)`` panel."""
    close = np.asarray(close, dtype=np.float64)
    # ``close[:1]`` rather than ``[close[0]]``: it repeats bar 0 along axis 0
    # whichever shape came in, so the panel form needs no separate branch.
    prev_close = np.concatenate((close[:1], close[:-1]), axis=0)
    # A bar with no previous close uses its own, which is what bar 0 has always
    # done. In a panel the same case appears again at the first bar of a symbol
    # whose history starts late, and treating it the same way is what makes a
    # padded column give exactly what the symbol's own frame gives.
    prev_close = np.where(np.isnan(prev_close), close, prev_close)
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    # rma is decorated with @indicator -> Callable[..., Any], so cast back.
    return cast(np.ndarray, rma(tr, n))
