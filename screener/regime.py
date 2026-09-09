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

:func:`classify_breadth` labels a *breadth* pair rather than a price series:
the share of a universe trading above its 20-day and 200-day EMA. It is the
single source of truth for those bands, shared by the live ``market-condition``
command and the backtester's breadth gate so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_FAST_WINDOW = 50
TREND_SLOW_WINDOW = 200
RISK_ON_SLOPE_WINDOW = 20
VOL_WINDOW = 20
VOL_DIST_WINDOW = 252
VOL_HIGH_PERCENTILE = 0.8

TREND_LABELS = ("bull", "pullback", "bear")
DEFENSIVE_REGIME_LABELS = ("risk_on", "transition", "risk_off")

BREADTH_LABELS = (
    "strong_bull",
    "bullish",
    "long_term_bull_pullback",
    "recovery_attempt",
    "bearish",
    "mixed",
)

# Bands are ordered: the first whose predicate holds wins. They are deliberately
# not exhaustive — a pair that satisfies none (say a firm 200-day against a
# merely soft 20-day) is ``mixed`` rather than being forced into the nearest
# named regime. Comparisons are strict, so exactly 50/50 is ``mixed``.
_BREADTH_BANDS: tuple[tuple[str, float, str, float, str], ...] = (
    ("strong_bull", 60.0, "gt", 60.0, "gt"),
    ("bullish", 50.0, "gt", 50.0, "gt"),
    ("long_term_bull_pullback", 50.0, "gt", 40.0, "lt"),
    ("recovery_attempt", 50.0, "lt", 60.0, "gt"),
    ("bearish", 40.0, "lt", 40.0, "lt"),
)


def _cmp(values: pd.Series, op: str, threshold: float) -> pd.Series:
    return values > threshold if op == "gt" else values < threshold


def classify_breadth(pct_20: float, pct_200: float) -> str:
    """Label one breadth reading; ``unknown`` when either share is missing.

    ``pct_20``/``pct_200`` are percentages (0-100) of a universe trading above
    its 20-day and 200-day EMA respectively.
    """
    if pd.isna(pct_20) or pd.isna(pct_200):
        return "unknown"
    labels = classify_breadth_series(
        pd.Series([float(pct_20)]), pd.Series([float(pct_200)])
    )
    return str(labels.iloc[0])


def classify_breadth_series(pct_20: pd.Series, pct_200: pd.Series) -> pd.Series:
    """Label each date's breadth pair; dates missing either share are 'unknown'.

    Vectorised counterpart of :func:`classify_breadth` over aligned series.
    """
    pct_20 = pd.to_numeric(pct_20, errors="coerce").astype(float)
    pct_200 = pd.to_numeric(pct_200, errors="coerce").astype(float)
    known = pct_20.notna() & pct_200.notna()

    out = pd.Series("unknown", index=pct_20.index, dtype=object)
    # ``mixed`` is the fallthrough for every known date no band claims.
    out[known] = "mixed"
    # Later bands must not overwrite an earlier match, so assign in reverse.
    for label, slow_threshold, slow_op, fast_threshold, fast_op in reversed(
        _BREADTH_BANDS
    ):
        match = (
            known
            & _cmp(pct_200, slow_op, slow_threshold)
            & _cmp(pct_20, fast_op, fast_threshold)
        )
        out[match] = label
    return out


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


def classify_defensive_regimes(close: pd.Series) -> pd.Series:
    """Label a benchmark ``risk_on`` / ``transition`` / ``risk_off``.

    This stricter, causal classifier is intended for defensive long-only
    strategies.  ``risk_on`` requires price above both moving averages, a
    bullish SMA50/SMA200 relationship, and a rising SMA50.  ``risk_off`` reacts
    immediately to either a close below SMA200 or an SMA50 break below SMA200,
    so it does not wait for the legacy ``bear`` label's two conditions to agree.
    Dates without every required trailing value are ``unknown``.
    """
    close = close.astype(float).sort_index()
    out = pd.Series("unknown", index=close.index, dtype=object)
    if close.empty:
        return out
    sma_fast = close.rolling(TREND_FAST_WINDOW, min_periods=TREND_FAST_WINDOW).mean()
    sma_slow = close.rolling(TREND_SLOW_WINDOW, min_periods=TREND_SLOW_WINDOW).mean()
    fast_slope = sma_fast - sma_fast.shift(RISK_ON_SLOPE_WINDOW)
    known = close.notna() & sma_fast.notna() & sma_slow.notna() & fast_slope.notna()
    risk_on = (
        known
        & (close > sma_slow)
        & (close > sma_fast)
        & (sma_fast > sma_slow)
        & (fast_slope > 0)
    )
    risk_off = known & ((close < sma_slow) | (sma_fast < sma_slow))
    out[known] = "transition"
    out[risk_off] = "risk_off"
    out[risk_on] = "risk_on"
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
