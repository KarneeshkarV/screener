"""Canonical pandas indicator primitives for Series and multi-symbol panels."""

from __future__ import annotations

from typing import Literal, cast, overload

import numpy as np
import pandas as pd


@overload
def _restore(
    template: pd.Series, values: np.ndarray, *, name: str | None = None
) -> pd.Series: ...


@overload
def _restore(
    template: pd.DataFrame, values: np.ndarray, *, name: str | None = None
) -> pd.DataFrame: ...


def _restore(
    template: pd.Series | pd.DataFrame,
    values: np.ndarray,
    *,
    name: str | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=name or template.name)
    return pd.DataFrame(values, index=template.index, columns=template.columns)


@overload
def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.Series: ...


@overload
def true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.DataFrame: ...


def true_range(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    *,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.Series | pd.DataFrame:
    """Return true range with an explicit first-bar convention.

    ``high_low`` treats the first bar's range as high minus low, matching Pine
    and the event scanners. ``nan`` requires a prior close, matching the
    vectorized sweep's historical warm-up behavior.
    """
    prev_close = close.shift(1)
    ranges = np.stack(
        [
            (high - low).abs().to_numpy(dtype=float),
            (high - prev_close).abs().to_numpy(dtype=float),
            (low - prev_close).abs().to_numpy(dtype=float),
        ]
    )
    values = (
        np.fmax.reduce(ranges) if first_bar == "high_low" else np.maximum.reduce(ranges)
    )
    return _restore(close, values, name="true_range")


@overload
def wilder_rsi(
    close: pd.Series, period: int = 14, *, min_periods: int = 0
) -> pd.Series: ...


@overload
def wilder_rsi(
    close: pd.DataFrame, period: int = 14, *, min_periods: int = 0
) -> pd.DataFrame: ...


def wilder_rsi(
    close: pd.Series | pd.DataFrame,
    period: int = 14,
    *,
    min_periods: int = 0,
) -> pd.Series | pd.DataFrame:
    """Return RSI using Wilder's exponentially smoothed gains and losses."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(
        alpha=1.0 / period, adjust=False, min_periods=min_periods
    ).mean()
    avg_loss = losses.ewm(
        alpha=1.0 / period, adjust=False, min_periods=min_periods
    ).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    condition = ~((avg_loss == 0.0) & (avg_gain > 0.0))
    if isinstance(result, pd.Series):
        return result.where(cast(pd.Series, condition), 100.0)
    return result.where(cast(pd.DataFrame, condition), 100.0)


@overload
def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    *,
    min_periods: int = 0,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.Series: ...


@overload
def wilder_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 14,
    *,
    min_periods: int = 0,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.DataFrame: ...


def wilder_atr(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    period: int = 14,
    *,
    min_periods: int = 0,
    first_bar: Literal["high_low", "nan"] = "high_low",
) -> pd.Series | pd.DataFrame:
    """Return Average True Range using Wilder smoothing."""
    ranges: pd.Series | pd.DataFrame
    if (
        isinstance(high, pd.Series)
        and isinstance(low, pd.Series)
        and isinstance(close, pd.Series)
    ):
        ranges = true_range(high, low, close, first_bar=first_bar)
    else:
        ranges = true_range(
            cast(pd.DataFrame, high),
            cast(pd.DataFrame, low),
            cast(pd.DataFrame, close),
            first_bar=first_bar,
        )
    return ranges.ewm(alpha=1.0 / period, adjust=False, min_periods=min_periods).mean()


def on_balance_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Return a multi-symbol On-Balance Volume panel."""
    differences = close.diff().to_numpy(dtype=float)
    volumes = volume.to_numpy(dtype=float)
    direction = np.where(differences > 0.0, 1.0, np.where(differences < 0.0, -1.0, 0.0))
    flow = direction * volumes
    flow[~np.isfinite(flow)] = 0.0
    return pd.DataFrame(
        np.cumsum(flow, axis=0), index=close.index, columns=close.columns
    )


__all__ = ["on_balance_volume", "true_range", "wilder_atr", "wilder_rsi"]
