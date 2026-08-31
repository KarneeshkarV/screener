"""Trading-session helpers.

Two questions live here. Intraday bars are stored with naive-UTC timestamps (see
``screener.backtester.price_frames.naive_normalized_index``), so grouping bars
into exchange sessions requires converting back to the market's timezone.
Session boundaries are detected positionally — a bar is session-last when the
next bar's local date differs — which needs no close-time table and tolerates
half-days.

The second question is the daily one: has a session finished? A daily bar for
a session that is still open is not a daily bar - it is a snapshot of the tape
at the moment it was requested, and it changes every time it is asked for. The
vendor serves one anyway, so anything that stores or ranks on daily bars has to
know when a venue has closed. That table is keyed on the symbol suffix because
the price fetchers deal in vendor symbols and never see a market name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import overload
from zoneinfo import ZoneInfo

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


@dataclass(frozen=True)
class ExchangeSession:
    """The wall clock a venue's regular session ends on."""

    timezone: str
    close: time

    def last_complete_date(self, now: datetime | None = None) -> date:
        """The newest calendar date whose session has finished.

        Weekends and holidays need no special case: this is an upper bound on
        which bars may be trusted, and no bar exists for a day the venue did
        not trade.
        """
        zone = ZoneInfo(self.timezone)
        local = now.astimezone(zone) if now is not None else datetime.now(zone)
        if local.time() >= self.close:
            return local.date()
        return local.date() - timedelta(days=1)


INDIA_SESSION = ExchangeSession(timezone="Asia/Kolkata", close=time(15, 30))
US_SESSION = ExchangeSession(timezone="America/New_York", close=time(16, 0))

# Indian symbols carry their exchange suffix; every other symbol yfinance
# serves is a US listing, which is the whole set of markets this project
# screens. A venue added later needs a row here, not a new mechanism.
_SESSION_BY_SUFFIX: dict[str, ExchangeSession] = {
    ".NS": INDIA_SESSION,
    ".BO": INDIA_SESSION,
}


def session_for(symbol: str) -> ExchangeSession:
    """The trading session ``symbol`` settles on."""
    upper = symbol.upper()
    for suffix, session in _SESSION_BY_SUFFIX.items():
        if upper.endswith(suffix):
            return session
    return US_SESSION


@overload
def drop_incomplete_sessions(
    frame: pd.DataFrame,
    symbol: str,
    *,
    interval: str = ...,
    now: datetime | None = ...,
) -> pd.DataFrame: ...


@overload
def drop_incomplete_sessions(
    frame: None,
    symbol: str,
    *,
    interval: str = ...,
    now: datetime | None = ...,
) -> None: ...


def drop_incomplete_sessions(
    frame: pd.DataFrame | None,
    symbol: str,
    *,
    interval: str = "1d",
    now: datetime | None = None,
) -> pd.DataFrame | None:
    """Drop daily rows for sessions that have not closed yet.

    Intraday frames are returned untouched: their stamps are instants, and an
    instant inside an open session is exactly what an intraday caller asked
    for. Only a daily bar claims to summarise a whole session.

    A missing frame passes straight through, so a caller holding a cache read
    that may be ``None`` does not have to unwrap it first; the return type
    follows the argument.
    """
    if interval != "1d" or frame is None or frame.empty:
        return frame
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return frame
    cutoff = pd.Timestamp(session_for(symbol).last_complete_date(now))
    if index.max() <= cutoff:
        return frame
    return frame.loc[index.normalize() <= cutoff]
