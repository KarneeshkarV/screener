"""Per-position stop distance: the arithmetic the exit and the sizer share.

``BacktestConfig.stop_loss`` is one fraction applied to every ticker, so a
quiet large cap and a volatile microcap get the same stop. ``stop_mode="atr"``
replaces it with ``stop_atr_multiple * ATR(stop_atr_window)`` measured at the
signal bar, so each position's stop is set by its own recent range.

This module sits below both consumers on purpose. ``core._make_slot_state``
turns the distance into the slot's ``stop_ref`` price, and the ``atr_risk``
sizing rule divides equity risk by the same fraction. They read one function so
the size of a position and the stop that closes it cannot disagree: before this
existed ``atr_risk`` sized as if the stop were ``sizing_atr_multiple * ATR``
while the exit fired at the flat ``stop_loss``, or never.

ATR is read at ``signal_idx``, never at the entry bar, for the same reason the
sizing rules are: the level has to be knowable when the order is placed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from screener.indicators.frames import wilder_atr

if TYPE_CHECKING:
    from screener.backtester.models import BacktestConfig

# Shared with ``screener.backtester.sizing``: both memo into
# ``_FrameCache.sizing_series`` under this key, so one frame computes Wilder
# ATR once however many slots and sizing calls read it.
_ATR_CACHE_KIND = "atr"


def atr_series(
    bars: pd.DataFrame,
    window: int,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Wilder ATR over ``bars``, computed at most once per frame and window."""
    if series_cache is not None:
        cached = series_cache.get((_ATR_CACHE_KIND, window))
        if cached is not None:
            return cached
    values = wilder_atr(
        bars["high"],
        bars["low"],
        bars["close"],
        window,
        min_periods=window,
    ).to_numpy(dtype=float)
    if series_cache is not None:
        series_cache[(_ATR_CACHE_KIND, window)] = values
    return values


def atr_stop_fraction(
    bars: pd.DataFrame,
    signal_idx: int,
    *,
    window: int,
    multiple: float,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> float:
    """``multiple * ATR / close`` at ``signal_idx``, or ``nan`` when undefined.

    ``nan`` means the frame has no usable ATR at that bar (short warmup, a flat
    or non-positive close). Callers fall back rather than inventing a level.
    """
    atr_value = float(atr_series(bars, window, series_cache)[signal_idx])
    close = float(bars["close"].iloc[signal_idx])
    if not math.isfinite(atr_value) or atr_value <= 0 or close <= 0:
        return math.nan
    return multiple * atr_value / close


def atr_stop_params(cfg: BacktestConfig) -> tuple[int, float] | None:
    """``(window, multiple)`` of the configured ATR stop, or ``None`` in pct mode."""
    if cfg.stop_mode != "atr" or cfg.stop_atr_multiple is None:
        return None
    return cfg.stop_atr_window, cfg.stop_atr_multiple


def entry_stop_fraction(
    cfg: BacktestConfig,
    bars: pd.DataFrame,
    signal_idx: int,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> float | None:
    """Fractional stop distance below the entry fill, or ``None`` for no stop.

    In ``atr`` mode a frame whose ATR is undefined at the signal bar falls back
    to ``stop_loss``; leaving that slot unprotected would make the stop silently
    optional on exactly the shortest-history names.
    """
    params = atr_stop_params(cfg)
    if params is not None:
        window, multiple = params
        fraction = atr_stop_fraction(
            bars,
            signal_idx,
            window=window,
            multiple=multiple,
            series_cache=series_cache,
        )
        if math.isfinite(fraction) and fraction > 0:
            # A stop wider than the entry price would sit at or below zero and
            # could never fire; clamp it to the whole position instead.
            return min(fraction, 1.0)
    return cfg.stop_loss if cfg.stop_loss else None


def entry_stop_price(
    cfg: BacktestConfig,
    entry_fill: float,
    bars: pd.DataFrame,
    signal_idx: int,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> float | None:
    """The slot's initial ``stop_ref`` price, or ``None`` when no stop applies."""
    fraction = entry_stop_fraction(cfg, bars, signal_idx, series_cache)
    if fraction is None:
        return None
    return entry_fill * (1.0 - fraction)
