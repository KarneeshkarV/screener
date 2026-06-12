"""Sector + market-cap enrichment for events.

Bulk-fetches sector and market-cap from TradingView in one screener call.
This is much cheaper than scraping per-ticker; the live screener already
exposes these columns. India events get an optional `--deep-india` path that
runs openscreener (Playwright-backed) only on the small set of surviving
events.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from tradingview_screener import Query, col

from screener.providers import CachedProvider, ProviderSpec

from .detector import Event


_TV_MARKETS = {"us": "america", "india": "india"}

# TradingView sector/market-cap enrichment: 24h parquet cache, "tradingview"
# circuit breaker.
_TV_SECTOR_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="tradingview",
        namespace="tradingview_sector",
        ttl_seconds=86400,
        kind="frame",
    )
)


def fetch_sector_map(
    market: str,
    symbols: Iterable[str],
    *,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> dict[str, dict]:
    """Return ``{symbol: {"sector": str, "market_cap": float}}`` for every
    symbol the TradingView screener can resolve."""
    syms = sorted({s.upper() for s in symbols if s})
    if not syms or market not in _TV_MARKETS:
        return {}
    query = (
        Query()
        .set_markets(_TV_MARKETS[market])
        .select("name", "sector", "market_cap_basic")
        .where(col("name").isin(syms))
        .limit(len(syms) + 50)
    )
    df = _TV_SECTOR_PROVIDER.fetch(
        ("sector_enrichment", market, syms),
        lambda: query.get_scanner_data()[1],
        refresh=refresh,
        fallback=pd.DataFrame(),
        ttl_seconds=cache_ttl,
        operation="sector enrichment",
    )
    out: dict[str, dict] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        sym = str(row.get("name") or "").upper().strip()
        if not sym:
            continue
        sector = row.get("sector")
        cap = row.get("market_cap_basic")
        out[sym] = {
            "sector": str(sector) if sector and not pd.isna(sector) else None,
            "market_cap": (
                float(cap) if cap is not None and not pd.isna(cap) else None
            ),
        }
    return out


def attach_sector(events: list[Event], sector_map: dict[str, dict]) -> None:
    for ev in events:
        meta = sector_map.get(ev.symbol.upper())
        if not meta:
            continue
        ev.sector = meta.get("sector") or ev.sector
        ev.market_cap = meta.get("market_cap") or ev.market_cap


def deep_enrich_india(events: list[Event]) -> None:
    """Optional openscreener-based enrichment for India events.

    Pulls promoter-holding from the latest shareholding pattern and appends
    a note if it's notably high (>50%). Fails silently per ticker — the
    screener.in scrape can be flaky.
    """
    try:
        from openscreener import Stock
    except ImportError:
        return
    for ev in events:
        try:
            from screener.insiders import _HttpScraper

            stock = Stock(ev.symbol, scraper=_HttpScraper())
            df = _fetch_shareholding_quarterly(stock)
        except Exception:
            continue
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        try:
            promoter = _extract_promoter_pct(df)
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if promoter is None:
            continue
        tag = f"promoter holding {promoter:.1f}%"
        ev.notes = (ev.notes + "; " + tag).strip("; ") if ev.notes else tag


def _fetch_shareholding_quarterly(stock):
    """Return quarterly shareholding data across openscreener API versions."""
    fetch = getattr(stock, "fetch", None)
    if callable(fetch):
        try:
            payload = fetch("shareholding")
        except TypeError:
            payload = fetch()
        if isinstance(payload, dict) and payload.get("shareholding") is not None:
            return payload["shareholding"]

    shareholding = getattr(stock, "shareholding_quarterly", None)
    if callable(shareholding):
        return shareholding()
    return shareholding


def _extract_promoter_pct(df) -> Optional[float]:
    """Best-effort: pull the most recent 'Promoters' row from a shareholding
    DataFrame and return their percent holding."""
    if df is None:
        return None
    try:
        if isinstance(df, list):
            if not df:
                return None
            latest = df[-1]
            if not isinstance(latest, dict):
                return None
            promoter = latest.get("promoters") or latest.get("Promoters")
            if promoter is None:
                return None
            return float(str(promoter).rstrip("%"))

        # screener.in shareholding tables typically have 'Promoters' as a row
        # label and quarterly columns.
        if hasattr(df, "index"):
            for label in df.index:
                if "promot" in str(label).lower():
                    row = df.loc[label]
                    last = row.iloc[-1] if len(row) else None
                    if last is None or pd.isna(last):
                        return None
                    return float(str(last).rstrip("%"))
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return None
