"""Noise filters for unusual-volume detection.

Drops illiquid names, sub-floor market caps, and India F&O ban-list tickers.
The F&O ban-list is fetched from NSE archives via a primed requests session
(jugaad-data does not expose this endpoint at the time of writing).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from screener.unusual_volume.nse_client import fetch_nse_text


FNO_BAN_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"


def fetch_fno_ban_list(timeout: float = 8.0) -> set[str]:
    """Return the symbols currently in the NSE F&O ban list.

    Fetched through the shared primed/repriming NSE session. Returns an empty
    set on any failure — callers should treat the filter as a soft guard, not
    a load-bearing check.
    """
    text = fetch_nse_text(FNO_BAN_URL, "fno ban list", timeout=timeout)
    if text is None:
        return set()
    return _parse_ban_csv(text)


def _parse_ban_csv(text: str) -> set[str]:
    """The CSV looks like:

        Securities in Ban For Trade Date 27-APR-2026:
        1,SAIL
        2,FOO

    First line is a header sentence; subsequent lines are ``rank,symbol``.
    """
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("securities in ban"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            out.add(parts[1].upper())
        elif len(parts) == 1 and parts[0].isalpha():
            out.add(parts[0].upper())
    return out


def passes_volume_floor(bars: pd.DataFrame, min_avg_volume: float, as_of: date) -> bool:
    """Reject tickers whose 20-day average daily volume is below the floor."""
    if bars is None or bars.empty:
        return False
    df = bars
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
        else:
            return False
    as_of_ts = pd.Timestamp(as_of).normalize()
    df = df[df.index <= as_of_ts]
    if len(df) < 21:
        return False
    avg20 = df["volume"].rolling(20, min_periods=20).mean().shift(1).iloc[-1]
    if pd.isna(avg20):
        # NaN volume inside the window leaves the rolling mean undefined;
        # treat the ticker as ineligible instead of comparing against NaN.
        return False
    return float(avg20) >= min_avg_volume


def passes_market_cap(market_cap: Optional[float], min_market_cap: float) -> bool:
    """Reject tickers below the configured market-cap floor.

    Returns True when ``market_cap`` is unknown — we'd rather emit an event
    with a missing-cap note than silently drop it.
    """
    if min_market_cap <= 0:
        return True
    if market_cap is None or pd.isna(market_cap):
        return True
    return float(market_cap) >= min_market_cap
