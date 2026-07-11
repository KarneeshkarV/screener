"""Sector mapping for cross-sectional neutralization.

Sectors are sourced from yfinance ``Ticker.info["sector"]`` with a long-lived
on-disk cache. Unknown / missing sectors map to the bucket ``"UNKNOWN"``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from screener.cache import is_fresh, read_json, write_json
from screener.symbols import tv_to_yf

LOG = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "sectors"
SECTOR_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
UNKNOWN_SECTOR = "UNKNOWN"

InfoFetcher = Callable[[str], Mapping[str, Any]]


def _cache_path(yf_symbol: str) -> Path:
    # yfinance symbols can contain characters unsafe for filenames (e.g. ``^``).
    safe = yf_symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.json"


def _default_info_fetcher(yf_symbol: str) -> Mapping[str, Any]:
    import yfinance as yf

    ticker = yf.Ticker(yf_symbol)
    info = getattr(ticker, "info", None)
    if not isinstance(info, Mapping):
        return {}
    return info


def _load_cached_sector(yf_symbol: str) -> str | None:
    path = _cache_path(yf_symbol)
    if not is_fresh(path, SECTOR_CACHE_TTL_SECONDS):
        return None
    payload = read_json(path, default=None)
    if not isinstance(payload, dict):
        return None
    sector = payload.get("sector")
    if not isinstance(sector, str) or not sector.strip():
        return None
    return sector.strip()


def _store_cached_sector(yf_symbol: str, sector: str) -> None:
    write_json(_cache_path(yf_symbol), {"symbol": yf_symbol, "sector": sector})


def _resolve_one(
    yf_symbol: str,
    *,
    use_cache: bool,
    info_fetcher: InfoFetcher,
) -> str:
    if use_cache:
        cached = _load_cached_sector(yf_symbol)
        if cached is not None:
            return cached
    try:
        info = info_fetcher(yf_symbol)
    except Exception as exc:  # noqa: BLE001 — provider failures -> UNKNOWN
        LOG.debug("sector lookup failed for %s: %s", yf_symbol, exc)
        info = {}
    raw = info.get("sector") if isinstance(info, Mapping) else None
    sector = raw.strip() if isinstance(raw, str) and raw.strip() else UNKNOWN_SECTOR
    if use_cache:
        _store_cached_sector(yf_symbol, sector)
    return sector


def sector_by_ticker(
    tickers: Iterable[str],
    market: str,
    *,
    use_cache: bool = True,
    info_fetcher: InfoFetcher | None = None,
) -> dict[str, str]:
    """Return ``{input_ticker: sector}`` for each ticker.

    Tickers may be TradingView-style (``NSE:RELIANCE``) or bare symbols; they are
    converted to yfinance form via :func:`screener.symbols.tv_to_yf` for the
    lookup key. The returned map is keyed by the *input* ticker strings so it
    lines up with rolling-backtest matrix columns.

    Missing / unfetchable sectors are bucketed as ``"UNKNOWN"``.
    """
    fetcher = info_fetcher or _default_info_fetcher
    out: dict[str, str] = {}
    # Dedupe yfinance lookups when multiple TV symbols map to the same YF symbol.
    yf_to_sector: dict[str, str] = {}
    for raw in tickers:
        ticker = raw.strip()
        if not ticker:
            continue
        yf_symbol = tv_to_yf(ticker, market)
        if yf_symbol not in yf_to_sector:
            yf_to_sector[yf_symbol] = _resolve_one(
                yf_symbol, use_cache=use_cache, info_fetcher=fetcher
            )
        out[ticker] = yf_to_sector[yf_symbol]
    return out
