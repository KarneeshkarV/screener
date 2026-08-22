"""Category 5: relative strength, against a benchmark and against a sector.

These are the only per-ticker features that read something outside the ticker.
The caller reindexes the benchmark and sector series onto the ticker's own bar
index before building the context, so a feature can never reach a benchmark bar
the ticker itself had not yet printed.

The cross-sectional half of this category (percentile within the screening
universe, rank consistency across horizons) cannot be computed one ticker at a
time. It lives in :func:`cross_sectional_ranks`, which takes the whole panel.
"""

from __future__ import annotations

import pandas as pd

from screener.research.features.base import (
    FeatureCtx,
    feature,
    rolling_ols_slope,
    safe_ratio,
)


def _relative_series(ctx: FeatureCtx, against: str) -> pd.Series | None:
    reference = ctx.benchmark if against == "benchmark" else ctx.sector
    if reference is None:
        return None
    reference = reference.astype(float)
    return safe_ratio(ctx.close, reference)


@feature(
    "relative_momentum",
    category="relative",
    doc="Stock return minus benchmark return over the window.",
    params={"window": 120},
    grid=({"window": 20}, {"window": 60}, {"window": 250}),
    min_lookback=250,
    needs_benchmark=True,
)
def relative_momentum(ctx: FeatureCtx, *, window: int) -> pd.Series:
    if ctx.benchmark is None:
        return pd.Series(float("nan"), index=ctx.bars.index)
    benchmark = ctx.benchmark.astype(float)
    stock_leg = safe_ratio(ctx.close, ctx.close.shift(window)) - 1.0
    bench_leg = safe_ratio(benchmark, benchmark.shift(window)) - 1.0
    return stock_leg - bench_leg


@feature(
    "sector_relative_momentum",
    category="relative",
    doc="Stock return minus its sector's return over the window.",
    params={"window": 120},
    grid=({"window": 20}, {"window": 60}, {"window": 250}),
    min_lookback=250,
    needs_sector=True,
)
def sector_relative_momentum(ctx: FeatureCtx, *, window: int) -> pd.Series:
    if ctx.sector is None:
        return pd.Series(float("nan"), index=ctx.bars.index)
    sector = ctx.sector.astype(float)
    stock_leg = safe_ratio(ctx.close, ctx.close.shift(window)) - 1.0
    sector_leg = safe_ratio(sector, sector.shift(window)) - 1.0
    return stock_leg - sector_leg


@feature(
    "rs_line_slope",
    category="relative",
    doc="Trailing OLS slope of the log relative-strength line versus the benchmark.",
    params={"window": 60},
    grid=({"window": 20}, {"window": 120}, {"window": 250}),
    min_lookback=250,
    needs_benchmark=True,
)
def rs_line_slope(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # The RS line itself, not its endpoints: a name can beat the index over 120
    # days while having stopped beating it two months ago, and the slope of the
    # line is what separates those.
    relative = _relative_series(ctx, "benchmark")
    if relative is None:
        return pd.Series(float("nan"), index=ctx.bars.index)
    import numpy as np

    slope, _, _ = rolling_ols_slope(np.log(relative.where(relative > 0.0)), window)
    return slope


@feature(
    "vol_adjusted_momentum",
    category="relative",
    doc="Return over the window divided by trailing volatility: a Sharpe-like score.",
    params={"window": 120, "vol_window": 120},
    grid=(
        {"window": 60, "vol_window": 60},
        {"window": 250, "vol_window": 250},
        {"window": 250, "vol_window": 60},
    ),
    min_lookback=260,
)
def vol_adjusted_momentum(
    ctx: FeatureCtx, *, window: int, vol_window: int
) -> pd.Series:
    # Ranking on this rather than raw momentum is the standard fix for momentum
    # portfolios filling up with the most violent names in the universe.
    total = safe_ratio(ctx.close, ctx.close.shift(window)) - 1.0
    vol = ctx.log_returns.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    return (total / vol).where(vol > 0.0)


def cross_sectional_ranks(
    feature_panel: pd.DataFrame, *, pct: bool = True
) -> pd.DataFrame:
    """Rank every ticker against the universe, one day at a time.

    ``feature_panel`` is ``date x ticker``. Ranking runs across each row
    independently, so no value from a later date can influence an earlier one:
    the operation is causal by construction as long as the panel itself is.
    Tickers missing on a day are excluded from that day's ranking rather than
    treated as the worst, which is what a naive fillna would do.
    """
    return feature_panel.rank(axis=1, pct=pct, na_option="keep")


def rank_consistency(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean cross-sectional percentile of one name across several horizons.

    ``panels`` maps a horizon label to a ``date x ticker`` feature panel, e.g.
    20/60/120/250-day momentum. A name ranked highly on every horizon scores
    near 1.0; one carried by a single horizon regresses toward 0.5. This is the
    "rank consistency of momentum across multiple horizons" leg of the brief.
    """
    if not panels:
        raise ValueError("rank_consistency needs at least one panel")
    ranked = [cross_sectional_ranks(panel) for panel in panels.values()]
    stacked = pd.concat(ranked, axis=0, keys=range(len(ranked)))
    return stacked.groupby(level=1).mean()


__all__ = [
    "cross_sectional_ranks",
    "rank_consistency",
    "relative_momentum",
    "rs_line_slope",
    "sector_relative_momentum",
    "vol_adjusted_momentum",
]
