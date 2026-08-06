"""Shared building blocks for cross-sectional screen scores.

All helpers return series aligned to the input index. Rank-based helpers
produce values in [0, 1] (``fillna(0)`` for missing). Scale helpers map
raw metrics onto [0, 1] quality curves used by weighted recipes.
"""

from __future__ import annotations

import math

import pandas as pd


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to float, or return all-NaN when the column is absent."""
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def percentile(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]; missing → 0."""
    return pd.to_numeric(series, errors="coerce").rank(pct=True).fillna(0)


def log_percentile(series: pd.Series) -> pd.Series:
    """Percentile of log1p(clip(x, 0)); useful for dollar-volume / market-cap."""
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return percentile(values.add(1).map(math.log))


def inv_percentile(series: pd.Series, *, positive_only: bool = False) -> pd.Series:
    """Higher score for *lower* raw values (e.g. P/E, debt).

    When ``positive_only`` is true, non-positive values are treated as missing
    so they do not earn a top inverse rank (same idea as GARP PEG handling).
    """
    values = pd.to_numeric(series, errors="coerce")
    if positive_only:
        values = values.where(values > 0)
    return (1 - values.rank(pct=True)).fillna(0)


def clip_scale(
    series: pd.Series,
    *,
    low: float,
    high: float,
) -> pd.Series:
    """Map ``series`` into [0, 1] by clipping to [low, high] and linear scaling."""
    values = pd.to_numeric(series, errors="coerce")
    span = high - low
    if span <= 0:
        raise ValueError("clip_scale requires high > low")
    return ((values.clip(lower=low, upper=high) - low) / span).fillna(0)


def rsi_quality(
    rsi: pd.Series, *, center: float = 60.0, half_width: float = 40.0
) -> pd.Series:
    """Peak at ``center`` (default 60); falls to 0 at center ± half_width."""
    values = pd.to_numeric(rsi, errors="coerce")
    return (1 - ((values - center).abs() / half_width)).clip(lower=0, upper=1).fillna(0)


def momentum_change(change: pd.Series) -> pd.Series:
    """Map day change % into [0, 1] with soft clips at -5 / +10."""
    values = pd.to_numeric(change, errors="coerce")
    return ((values.clip(lower=-5, upper=10) + 5) / 15).fillna(0)


def overextension_penalty(
    close: pd.Series,
    ema20: pd.Series,
    *,
    start: float = 0.12,
    span: float = 0.25,
) -> pd.Series:
    """Penalize price stretched above EMA20; 0 below ``start``, 1 at start+span."""
    extension = ((close - ema20) / ema20).fillna(0)
    return ((extension - start).clip(lower=0) / span).clip(upper=1)


def liquidity_from_dollar_volume(volume: pd.Series, close: pd.Series) -> pd.Series:
    """Log-percentile of dollar volume."""
    return log_percentile(pd.to_numeric(volume, errors="coerce") * close)


def trend_stack_strength(
    close: pd.Series,
    ema5: pd.Series,
    ema20: pd.Series,
    ema100: pd.Series,
    ema200: pd.Series,
) -> pd.Series:
    """Percentile of non-negative EMA stack spread (capped)."""
    trend_spread = (
        ((ema5 - ema20) / close)
        + ((ema20 - ema100) / close)
        + ((ema100 - ema200) / close)
    ).clip(lower=0, upper=0.35)
    return percentile(trend_spread)


def proximity_to_high(close: pd.Series, high_52w: pd.Series) -> pd.Series:
    """close / 52w high clipped to [0, 1]; missing high → 0."""
    ratio = (close / high_52w.replace(0, pd.NA)).clip(lower=0, upper=1)
    return ratio.fillna(0)


def above_flag(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """1.0 when numerator > denominator, else 0.0 (NaN-safe)."""
    a = pd.to_numeric(numerator, errors="coerce")
    b = pd.to_numeric(denominator, errors="coerce")
    return (a > b).astype(float).fillna(0.0)


__all__ = [
    "above_flag",
    "clip_scale",
    "inv_percentile",
    "liquidity_from_dollar_volume",
    "log_percentile",
    "momentum_change",
    "numeric",
    "overextension_penalty",
    "percentile",
    "proximity_to_high",
    "rsi_quality",
    "trend_stack_strength",
]
