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
def _wilder_rma(values: pd.Series, period: int) -> pd.Series: ...


@overload
def _wilder_rma(values: pd.DataFrame, period: int) -> pd.DataFrame: ...


def _wilder_rma(
    values: pd.Series | pd.DataFrame, period: int
) -> pd.Series | pd.DataFrame:
    """Wilder's running average, seeded the way Pine ``ta.rma`` seeds it.

    ``ta.rma`` starts from the *simple mean of the first ``period``
    observations* and only then applies the ``alpha = 1/period`` recursion.
    ``ewm(alpha=..., adjust=False)`` instead starts from the first observation
    alone, which is a different number: the resulting error decays by
    ``(1 - 1/period)`` per bar, so at ``period = 14`` it is still worth more
    than one RSI point around bar 50. ``min_periods`` cannot repair that - it
    hides early bars, it does not change what the recursion started from.

    The seed is placed at the position of the ``period``-th observation and
    everything before it is blanked, so a plain ``adjust=False`` pass picks the
    seed up as its first value and carries Wilder's recursion from there.
    """
    observed = values.notna()
    count = observed.cumsum()
    # A cumulative sum, read at the seed row, *is* the sum of the first
    # ``period`` observations: every earlier gap contributed an exact zero. It
    # costs a fraction of a full ``rolling(period).mean()`` and, unlike one,
    # keeps seeding across an interior gap instead of returning NaN there.
    seed = values.fillna(0.0).cumsum() / period
    settled = (count > period).to_numpy()
    at_seed = (observed & (count == period)).to_numpy()
    seeded = values.where(settled).mask(at_seed, seed)
    return seeded.ewm(alpha=1.0 / period, adjust=False).mean()


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
    """Return RSI using Wilder's exponentially smoothed gains and losses.

    Matches Pine ``ta.rsi``: ``close.diff()`` is undefined on the first bar, so
    the smoothers seed off the first ``period`` *real* changes and the first
    value lands on bar ``period``. ``min_periods`` now only blanks output that
    rests on fewer than that many observations; the seed no longer depends on
    it.
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = _wilder_rma(gains, period)
    avg_loss = _wilder_rma(losses, period)
    if min_periods > 0:
        enough = (delta.notna().cumsum() >= min_periods).to_numpy()
        avg_gain = avg_gain.where(enough)
        avg_loss = avg_loss.where(enough)
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
