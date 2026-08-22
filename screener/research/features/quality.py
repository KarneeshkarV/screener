"""Category 2: trend quality and chop filters.

These separate a clean trend from a noisy one that travels the same net
distance. Direction is deliberately not in scope: most of these are magnitude
or agreement measures, so pair them with a :mod:`.trend` sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.research.features.base import (
    FeatureCtx,
    feature,
    safe_ratio,
)


@feature(
    "efficiency_ratio",
    category="quality",
    doc="Kaufman efficiency ratio: net move divided by total path travelled.",
    params={"window": 20},
    grid=({"window": 10}, {"window": 60}, {"window": 120}),
    min_lookback=125,
)
def efficiency_ratio(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # 1.0 is a straight line, 0.0 is pure round-tripping. The denominator is the
    # sum of absolute daily moves over the same window, so both legs are
    # strictly trailing.
    close = ctx.close
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return safe_ratio(net, path)


@feature(
    "return_autocorr",
    category="quality",
    doc="Trailing lag-k autocorrelation of daily returns. Positive = trending.",
    params={"window": 60, "lag": 1},
    grid=(
        {"window": 120, "lag": 1},
        {"window": 250, "lag": 1},
        {"window": 120, "lag": 5},
    ),
    min_lookback=260,
)
def return_autocorr(ctx: FeatureCtx, *, window: int, lag: int) -> pd.Series:
    # Persistent daily returns are the statistical signature of a trend that a
    # momentum number alone cannot see.
    returns = ctx.log_returns
    lagged = returns.shift(lag)
    both = pd.concat([returns, lagged], axis=1).dropna()
    if both.empty:
        return pd.Series(np.nan, index=ctx.bars.index)
    paired = returns.where(lagged.notna())
    mean_x = paired.rolling(window, min_periods=window).mean()
    mean_y = lagged.rolling(window, min_periods=window).mean()
    cov = (paired * lagged).rolling(window, min_periods=window).mean() - mean_x * mean_y
    std_x = paired.rolling(window, min_periods=window).std(ddof=0)
    std_y = lagged.rolling(window, min_periods=window).std(ddof=0)
    denominator = std_x * std_y
    return (cov / denominator).where(denominator > 0.0)


@feature(
    "variance_ratio",
    category="quality",
    doc="Lo-MacKinlay variance ratio at horizon q. >1 trending, <1 mean-reverting.",
    params={"window": 120, "q": 5},
    grid=(
        {"window": 120, "q": 2},
        {"window": 120, "q": 10},
        {"window": 250, "q": 5},
        {"window": 250, "q": 20},
    ),
    min_lookback=280,
)
def variance_ratio(ctx: FeatureCtx, *, window: int, q: int) -> pd.Series:
    # Var(q-day return) / (q * Var(1-day return)). Under a random walk this is
    # 1; trending series overshoot it because their moves compound.
    log_close = ctx.log_close
    r1 = log_close.diff()
    rq = log_close.diff(q)
    var1 = r1.rolling(window, min_periods=window).var(ddof=1)
    varq = rq.rolling(window, min_periods=window).var(ddof=1)
    return (varq / (float(q) * var1)).where(var1 > 0.0)


@feature(
    "directional_share",
    category="quality",
    doc="Share of the last N days moving with the window's own net direction.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def directional_share(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # A trend carried by many small aligned days is more durable than one
    # carried by two gaps, and this tells the two apart.
    returns = ctx.log_returns
    up_share = (returns > 0.0).astype(float).where(returns.notna())
    up_share = up_share.rolling(window, min_periods=window).mean()
    net_up = ctx.close > ctx.close.shift(window)
    return up_share.where(net_up, 1.0 - up_share)


@feature(
    "ema_crossings",
    category="quality",
    doc="Times price crossed its EMA in N bars, per bar. Low = clean, high = chop.",
    params={"window": 60, "span": 20},
    grid=(
        {"window": 120, "span": 20},
        {"window": 60, "span": 50},
        {"window": 250, "span": 50},
    ),
    min_lookback=280,
    higher_is_stronger=False,
)
def ema_crossings(ctx: FeatureCtx, *, window: int, span: int) -> pd.Series:
    # The most direct chop measure in the set: a clean trend sits on one side of
    # its EMA for weeks, chop saws across it.
    ema = ctx.close.ewm(span=span, adjust=False, min_periods=span).mean()
    above = (ctx.close > ema).astype(float).where(ema.notna())
    crossed = above.diff().abs()
    return crossed.rolling(window, min_periods=window).sum() / float(window)


@feature(
    "trend_persistence",
    category="quality",
    doc="Longest run of same-signed daily returns in the window, per bar.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def trend_persistence(ctx: FeatureCtx, *, window: int) -> pd.Series:
    def longest_run(values: np.ndarray) -> float:
        best = current = 0
        previous = 0.0
        for value in values:
            sign = 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)
            if sign != 0.0 and sign == previous:
                current += 1
            else:
                current = 1 if sign != 0.0 else 0
            previous = sign
            best = max(best, current)
        return float(best) / float(len(values))

    returns = ctx.log_returns.fillna(0.0)
    return returns.rolling(window, min_periods=window).apply(longest_run, raw=True)


@feature(
    "hurst_vr",
    category="quality",
    doc="Variance-ratio Hurst estimate. >0.5 persistent, <0.5 mean-reverting.",
    params={"window": 126, "q": 5},
    grid=({"window": 250, "q": 5}, {"window": 126, "q": 10}, {"window": 250, "q": 20}),
    min_lookback=280,
)
def hurst_vr(ctx: FeatureCtx, *, window: int, q: int) -> pd.Series:
    # H = 0.5 + log(VR) / (2 log q). Kept as a secondary read on the same
    # evidence as `variance_ratio`, on a scale people already have priors about.
    ratio = variance_ratio(ctx, window=window, q=q)
    hurst = 0.5 + np.log(ratio.where(ratio > 0.0)) / (2.0 * np.log(float(q)))
    return pd.Series(hurst, index=ratio.index)
