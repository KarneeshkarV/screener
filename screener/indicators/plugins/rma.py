"""Wilder's RMA, matching Pine ``ta.rma``."""

from __future__ import annotations

import numpy as np

from screener.indicators.registry import indicator


@indicator("rma")
def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's running average: SMA seed, then ``alpha = 1/n``.

    The seed is the mean of the first ``n`` *observations* and lands at the
    position of the ``n``-th one, which is not index ``n - 1`` when the input
    opens with undefined values. ``ta.change`` is undefined on bar 0, so
    ``rsi`` feeds exactly such an input, and counting positions instead of
    observations would seed off ``n - 1`` real values one bar early.

    Past the seed the recursion propagates NaN forward, as ``ema`` does: a gap
    in the middle of the input is missing information, not a zero.
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    observed = np.flatnonzero(~np.isnan(x))
    if observed.size < n:
        return out
    seed_at = int(observed[n - 1])
    alpha = 1.0 / n
    out[seed_at] = np.mean(x[observed[:n]])
    for i in range(seed_at + 1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out
