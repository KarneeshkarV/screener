"""Trading-session helpers for intraday bars.

Intraday bars are stored with naive-UTC timestamps (see
``screener.backtester.price_frames.naive_normalized_index``), so grouping bars
into exchange sessions requires converting back to the market's timezone.
Session boundaries are detected positionally — a bar is session-last when the
next bar's local date differs — which needs no close-time table and tolerates
half-days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_timezone(market: str) -> str:
    """Return the IANA timezone for a market name (e.g. ``us`` -> New York)."""
    from screener.markets import get_market

    return get_market(market).timezone


def session_dates(index: pd.DatetimeIndex, market_tz: str) -> np.ndarray:
    """Local session-date label for each naive-UTC bar timestamp."""
    localized = index.tz_localize("UTC").tz_convert(market_tz)
    return np.asarray(localized.date)


def is_session_last(index: pd.DatetimeIndex, market_tz: str) -> np.ndarray:
    """Boolean mask: True where a bar is the last bar of its trading session."""
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    labels = session_dates(index, market_tz)
    mask = np.empty(len(labels), dtype=bool)
    mask[:-1] = labels[:-1] != labels[1:]
    mask[-1] = True
    return mask
