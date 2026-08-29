"""Wilder RSI."""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.rma import rma
from screener.indicators.registry import indicator


@indicator("rsi")
def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's RSI, matching Pine ``ta.rsi``.

    Bar 0 has no prior close, so its change is undefined - Pine's ``ta.change``
    returns ``na`` there. Prepending ``close[0]`` instead would manufacture a
    zero change, which is a real observation to ``rma``: it would seed off
    ``n - 1`` genuine changes plus that zero and emit its first value on bar
    ``n - 1`` rather than bar ``n``. The zero drags the seed toward the middle,
    and the error persists - about 0.3 to 0.9 RSI points against TradingView
    for the life of the series.
    """
    diff = np.diff(close, prepend=np.nan)
    undefined = np.isnan(diff)
    up = np.where(undefined, np.nan, np.maximum(diff, 0.0))
    dn = np.where(undefined, np.nan, np.maximum(-diff, 0.0))
    rma_up = rma(up, n)
    rma_dn = rma(dn, n)
    rs = np.where(rma_dn > 0, rma_up / np.maximum(rma_dn, 1e-12), np.inf)
    out = 100 - 100 / (1 + rs)
    out[rma_dn == 0] = 100
    # Warm-up region: RMA is NaN for the first n-1 bars, so rma_dn is NaN and
    # rs=inf would spuriously pin RSI at 100. Match the NaN-warmup convention of
    # RMA/ATR/SMA/stdev.
    out[np.isnan(rma_up)] = np.nan
    return out
