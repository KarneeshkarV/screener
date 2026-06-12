"""FMP institutional ownership summaries (US only).

Uses FMP's ``/api/v3/institutional-holder/{symbol}`` endpoint, which returns
one row per 13F institutional holder with ``holder``, ``shares``,
``dateReported`` and ``change`` (share delta vs. the holder's previous
quarterly filing). We aggregate per ticker: number of institutional holders,
total institutional shares, and the quarter-over-quarter share change
(absolute and percent) where the API exposes the ``change`` field.

Request/caching/resilience follow the FMP patterns in ``screener.insiders``.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from screener.insiders import _SCREENER_HEADERS
from screener.providers import CachedProvider, ProviderSpec


logger = logging.getLogger(__name__)

_FMP_INSTITUTIONAL_URL = (
    "https://financialmodelingprep.com/api/v3/institutional-holder/{symbol}"
)

# FMP institutional ownership: 24h cache, "fmp" circuit breaker.
_FMP_INSTITUTIONAL_PROVIDER = CachedProvider(
    ProviderSpec(provider="fmp", namespace="fmp_institutional", ttl_seconds=86400)
)


def _aggregate_institutional_holders(rows: list[dict]) -> Optional[dict]:
    """Aggregate per-holder FMP rows into a per-ticker ownership summary.

    ``shares`` rows that are not numeric are skipped entirely; a missing or
    non-numeric ``change`` only drops that row from the QoQ delta (the API
    does not always expose it). When no row carries a usable ``change`` the
    QoQ fields are ``None`` rather than a misleading zero.
    """
    if not rows:
        return None
    holders = 0
    total_shares = 0.0
    change_shares = 0.0
    has_change = False
    for row in rows:
        try:
            shares = float(row.get("shares") or 0.0)
        except (TypeError, ValueError):
            continue
        holders += 1
        total_shares += shares
        try:
            change = float(row["change"])
        except (KeyError, TypeError, ValueError):
            continue
        change_shares += change
        has_change = True
    if holders == 0:
        return None
    qoq_change: Optional[float] = change_shares if has_change else None
    qoq_change_pct: Optional[float] = None
    if has_change:
        prev_total = total_shares - change_shares
        if prev_total > 0:
            qoq_change_pct = change_shares / prev_total * 100.0
    return {
        "holders": holders,
        "total_shares": total_shares,
        "qoq_change_shares": qoq_change,
        "qoq_change_pct": qoq_change_pct,
    }


def _fetch_fmp_institutional_one(
    symbol: str,
    *,
    api_key: str,
    cache_ttl: float | None,
    refresh: bool,
) -> Optional[dict]:
    def _fetch() -> Optional[dict]:
        query = urllib.parse.urlencode({"apikey": api_key})
        url = _FMP_INSTITUTIONAL_URL.format(symbol=urllib.parse.quote(symbol))
        req = urllib.request.Request(f"{url}?{query}", headers=_SCREENER_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8", "ignore"))
        rows = payload if isinstance(payload, list) else None
        if not rows:
            return None
        agg = _aggregate_institutional_holders(rows)
        if agg is None:
            return None
        return {"symbol": symbol, **agg}

    return _FMP_INSTITUTIONAL_PROVIDER.fetch(
        ("institutional_holder", symbol),
        _fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"institutional holders {symbol}",
    )


def fetch_fmp_institutional(
    symbols: list[str],
    *,
    api_key: str,
    max_workers: int = 8,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch institutional ownership summaries from FMP for each US ticker.

    Symbols with no FMP data are simply absent from the returned frame.
    """
    if not symbols:
        return pd.DataFrame()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _fetch_fmp_institutional_one,
                s,
                api_key=api_key,
                cache_ttl=cache_ttl,
                refresh=refresh,
            )
            for s in symbols
        ]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)
