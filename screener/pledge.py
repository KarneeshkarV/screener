"""Promoter share-pledge percentage — dual-source (NSE filings + openscreener).

FMP has no promoter-pledge endpoint (pledge is an India-specific disclosure),
so ``pledge_pct`` is sourced from the NSE Corporate Filings "Pledged Data"
endpoint (authoritative regulatory primary) with the screener.in shareholding
page as a fallback when NSE has no row. The two providers use distinct
circuit-breaker names so one outage does not suppress the other.
"""

from __future__ import annotations

import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from screener.insiders import _HttpScraper
from screener.providers import CachedProvider, ProviderSpec
from screener.unusual_volume.detector import Event
from screener.unusual_volume.nse_client import nse_cached_json

_NSE_PLEDGE_URL = "https://www.nseindia.com/api/corporate-pledgedata?symbol={sym}"
# screener.in renders e.g. "Pledged percentage</span> ... 12.34%" in the
# shareholding section. Match the number that follows the label.
_OSC_PLEDGE_RE = re.compile(
    r"pledged?\s*percentage[^0-9%]{0,60}?([0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)

# screener.in pledge scrape: 7d cache, "screener-in" circuit breaker. The
# scraper (``_HttpScraper.fetch_page``) is already resilience-wrapped, so the
# provider's breaker is a no-op belt-and-suspenders here.
_OSC_PLEDGE_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="screener-in", namespace="openscreener_pledge", ttl_seconds=7 * 86400
    )
)


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _as_pct(value: object) -> Optional[float]:
    num = _as_float(value)
    if num is None or num < 0.0 or num > 100.0:
        return None
    return num


def fetch_nse_pledge(symbol: str, *, refresh: bool = False) -> Optional[float]:
    """Latest promoter-pledged % from NSE Corporate Filings, or None."""
    url = _NSE_PLEDGE_URL.format(sym=urllib.parse.quote(symbol.upper()))
    raw = nse_cached_json(
        "nse_pledge",
        ("pledge", symbol.upper()),
        url,
        f"pledge data {symbol}",
        refresh=refresh,
        ttl_seconds=6 * 3600,
    )
    rows = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        return None
    latest = rows[0] if isinstance(rows[0], dict) else None
    if latest is None:
        return None
    for key in (
        "per. of Promoter Holding Shares pledge",
        "percentageOfPromoterHoldingPledged",
        "pledgePercentage",
        "perShareEncumbered",
    ):
        val = _as_pct(latest.get(key))
        if val is not None:
            return val
    # Fallback: any percent-like key mentioning "pledge" with a numeric value.
    for key, val in latest.items():
        key_l = str(key).lower()
        if "pledge" in key_l and ("per" in key_l or "%" in key_l or "percent" in key_l):
            num = _as_pct(val)
            if num is not None:
                return num
    return None


def fetch_openscreener_pledge(name: str, *, refresh: bool = False) -> Optional[float]:
    """Promoter-pledged % scraped from the screener.in shareholding section."""

    def _fetch() -> Optional[float]:
        html = _HttpScraper().fetch_page(name)
        if not html:
            return None
        match = _OSC_PLEDGE_RE.search(html)
        return _as_pct(match.group(1)) if match else None

    return _OSC_PLEDGE_PROVIDER.fetch(
        ("pledge", name.upper()),
        _fetch,
        refresh=refresh,
        fallback=None,
        operation=f"pledge scrape {name}",
    )


def resolve_pledge_pct(
    symbol: str, name: str, *, refresh: bool = False
) -> Optional[float]:
    """Preferred = NSE filings; fallback = openscreener when NSE has no row."""
    nse = fetch_nse_pledge(symbol, refresh=refresh)
    if nse is not None:
        return nse
    return fetch_openscreener_pledge(name, refresh=refresh)


def overlay_pledge(
    events: list[Event], *, refresh: bool = False, max_workers: int = 6
) -> None:
    """Mutate each event's ``pledge_pct`` in place."""
    if not events:
        return
    symbols = sorted({ev.symbol.upper() for ev in events})

    def _one(sym: str) -> tuple[str, Optional[float]]:
        return sym, resolve_pledge_pct(sym, sym, refresh=refresh)

    by_symbol: dict[str, Optional[float]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed([pool.submit(_one, s) for s in symbols]):
            sym, val = fut.result()
            by_symbol[sym] = val
    for ev in events:
        val = by_symbol.get(ev.symbol.upper())
        if val is not None and not pd.isna(val):
            ev.pledge_pct = float(val)
