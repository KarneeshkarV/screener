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

from .detector import Event


_TV_MARKETS = {"us": "america", "india": "india"}


def fetch_sector_map(market: str, symbols: Iterable[str]) -> dict[str, dict]:
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
    try:
        _count, df = query.get_scanner_data()
    except Exception:
        return {}
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
    except Exception:
        return
    for ev in events:
        try:
            stock = Stock(ev.symbol)
            stock.fetch()
            df = stock.shareholding_quarterly
        except Exception:
            continue
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        try:
            promoter = _extract_promoter_pct(df)
        except Exception:
            continue
        if promoter is None:
            continue
        tag = f"promoter holding {promoter:.1f}%"
        ev.notes = (ev.notes + "; " + tag).strip("; ") if ev.notes else tag


def _extract_promoter_pct(df) -> Optional[float]:
    """Best-effort: pull the most recent 'Promoters' row from a shareholding
    DataFrame and return their percent holding."""
    if df is None:
        return None
    try:
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
    except Exception:
        return None
    return None
