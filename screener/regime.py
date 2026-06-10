"""Market regime classification from a close-price series.

Two independent per-date labelings, both strictly point-in-time (each date's
label uses only data up to and including that date — rolling windows, no
centering, no lookahead):

- :func:`classify_regimes` — trend regime via SMA50/SMA200:
  ``bull`` when close > SMA200 and SMA50 > SMA200, ``bear`` when close < SMA200
  and SMA50 < SMA200, otherwise ``pullback``. Dates without enough history for
  SMA200 are labeled ``unknown``.
- :func:`vol_regime` — volatility regime via the percentile of 20-day realized
  volatility within its own trailing 252-observation distribution:
  ``high_vol`` when at or above the 80th percentile, else ``normal``.
  Warmup dates are labeled ``unknown``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TREND_FAST_WINDOW = 50
TREND_SLOW_WINDOW = 200
VOL_WINDOW = 20
VOL_DIST_WINDOW = 252
VOL_HIGH_PERCENTILE = 0.8

TREND_LABELS = ("bull", "pullback", "bear")


def classify_regimes(close: pd.Series) -> pd.Series:
    """Label each date 'bull' / 'pullback' / 'bear' / 'unknown'.

    Warmup dates (fewer than ``TREND_SLOW_WINDOW`` prior observations) are
    'unknown'. A flat series (close == SMA200) is 'pullback' by construction.
    """
    close = close.astype(float).sort_index()
    out = pd.Series("unknown", index=close.index, dtype=object)
    if close.empty:
        return out
    sma_fast = close.rolling(TREND_FAST_WINDOW, min_periods=TREND_FAST_WINDOW).mean()
    sma_slow = close.rolling(TREND_SLOW_WINDOW, min_periods=TREND_SLOW_WINDOW).mean()
    known = close.notna() & sma_fast.notna() & sma_slow.notna()
    bull = known & (close > sma_slow) & (sma_fast > sma_slow)
    bear = known & (close < sma_slow) & (sma_fast < sma_slow)
    out[known] = "pullback"
    out[bull] = "bull"
    out[bear] = "bear"
    return out


def vol_regime(close: pd.Series) -> pd.Series:
    """Label each date 'high_vol' / 'normal' / 'unknown'.

    20-day realized volatility (std of daily returns) ranked against its own
    trailing ``VOL_DIST_WINDOW`` observations; 'high_vol' when the current
    value sits at or above the ``VOL_HIGH_PERCENTILE`` percentile. Dates
    without a full trailing distribution are 'unknown'.
    """
    close = close.astype(float).sort_index()
    out = pd.Series("unknown", index=close.index, dtype=object)
    if len(close) < 2:
        return out
    returns = close.pct_change()
    realized_vol = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=0)
    pct_rank = realized_vol.rolling(VOL_DIST_WINDOW, min_periods=VOL_DIST_WINDOW).rank(
        pct=True
    )
    known = pct_rank.notna()
    out[known] = np.where(pct_rank[known] >= VOL_HIGH_PERCENTILE, "high_vol", "normal")
    return out
