"""Category 6: volume, liquidity and price-volume confirmation.

Liquidity is the one category that is a hard gate rather than a score: a name
that cannot be traded in size is not a candidate at any signal strength, so
``adv_value`` is meant to be used as an exclusion and the rest as evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.research.features.base import (
    FeatureCtx,
    feature,
    rolling_ols_slope,
    safe_ratio,
)


def _traded_value(ctx: FeatureCtx) -> pd.Series:
    return ctx.close * ctx.bars["volume"].astype(float)


@feature(
    "adv_value",
    category="liquidity",
    doc="Average daily traded value over the window, in currency units.",
    params={"window": 20},
    grid=({"window": 60}, {"window": 120}),
    min_lookback=125,
)
def adv_value(ctx: FeatureCtx, *, window: int) -> pd.Series:
    return _traded_value(ctx).rolling(window, min_periods=window).mean()


@feature(
    "relative_volume",
    category="liquidity",
    doc="Today's traded value over its own trailing average. >1 = unusual activity.",
    params={"window": 20},
    grid=({"window": 60}, {"window": 120}),
    min_lookback=125,
)
def relative_volume(ctx: FeatureCtx, *, window: int) -> pd.Series:
    value = _traded_value(ctx)
    # The baseline is lagged one bar so today's own volume does not inflate the
    # average it is being measured against.
    baseline = value.rolling(window, min_periods=window).mean().shift(1)
    return safe_ratio(value, baseline)


@feature(
    "volume_trend",
    category="liquidity",
    doc="Trailing OLS slope of log traded value: is participation growing?",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}),
    min_lookback=125,
)
def volume_trend(ctx: FeatureCtx, *, window: int) -> pd.Series:
    value = _traded_value(ctx)
    slope, _, _ = rolling_ols_slope(np.log(value.where(value > 0.0)), window)
    return slope


@feature(
    "up_down_volume_ratio",
    category="liquidity",
    doc="Traded value on up days over traded value on down days in the window.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def up_down_volume_ratio(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # Accumulation versus distribution. Above 1 means the buying days are the
    # ones carrying the volume.
    value = _traded_value(ctx)
    returns = ctx.log_returns
    up = value.where(returns > 0.0, 0.0).where(returns.notna())
    down = value.where(returns < 0.0, 0.0).where(returns.notna())
    up_sum = up.rolling(window, min_periods=window).sum()
    down_sum = down.rolling(window, min_periods=window).sum()
    return safe_ratio(up_sum, down_sum)


@feature(
    "price_volume_corr",
    category="liquidity",
    doc="Trailing correlation of daily return and traded-value change.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
)
def price_volume_corr(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # Positive = volume confirms direction. Negative = the move is happening on
    # drying-up participation, which is the divergence case.
    returns = ctx.log_returns
    value = _traded_value(ctx)
    value_change = pd.Series(np.log(value.where(value > 0.0)), index=value.index).diff()
    paired = returns.where(value_change.notna())
    mean_x = paired.rolling(window, min_periods=window).mean()
    mean_y = value_change.rolling(window, min_periods=window).mean()
    cov = (paired * value_change).rolling(window, min_periods=window).mean() - (
        mean_x * mean_y
    )
    std_x = paired.rolling(window, min_periods=window).std(ddof=0)
    std_y = value_change.rolling(window, min_periods=window).std(ddof=0)
    denominator = std_x * std_y
    return (cov / denominator).where(denominator > 0.0)


@feature(
    "amihud_illiquidity",
    category="liquidity",
    doc="Amihud: mean of |return| / traded value. High = price moves on little volume.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
    higher_is_stronger=False,
)
def amihud_illiquidity(ctx: FeatureCtx, *, window: int) -> pd.Series:
    value = _traded_value(ctx)
    impact = safe_ratio(ctx.returns.abs(), value)
    # Scaled to a readable magnitude; the constant is cosmetic and cancels in
    # any ranking or threshold expressed as a percentile.
    return impact.rolling(window, min_periods=window).mean() * 1e9


@feature(
    "turnover_consistency",
    category="liquidity",
    doc="Share of days in the window whose traded value cleared a floor fraction.",
    params={"window": 60, "floor": 0.25},
    grid=({"window": 120, "floor": 0.25}, {"window": 60, "floor": 0.5}),
    min_lookback=190,
)
def turnover_consistency(ctx: FeatureCtx, *, window: int, floor: float) -> pd.Series:
    # Catches the name whose ADV is respectable only because of three frantic
    # days. The floor is a fraction of the name's own trailing median.
    value = _traded_value(ctx)
    median = value.rolling(window, min_periods=window).median().shift(1)
    cleared = (value > floor * median).astype(float).where(median.notna())
    return cleared.rolling(window, min_periods=window).mean()
