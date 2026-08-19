"""Sector mapping for cross-sectional neutralization.

Sectors come from FMP's ``profile`` endpoint, which answers in batches and
covers both markets (``AAPL``, ``RELIANCE.NS``), with yfinance
``Ticker.info["sector"]`` as the per-symbol fallback. A long-lived on-disk
cache sits in front of both. Unknown / missing sectors bucket as ``"UNKNOWN"``.

FMP is primary because yfinance is the wrong shape for this job: ``Ticker.info``
is one HTTP round trip per symbol and rate-limits hard enough that resolving a
500-name universe reliably fails part-way. Those failures are indistinguishable
from "this symbol has no sector", so they used to land in the cache as UNKNOWN
and quietly collapse sector-neutral ranking into a single bucket. One batched
FMP call per hundred symbols removes the rate-limit exposure that created them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from screener.cache import is_fresh, read_json, write_json
from screener.symbols import tv_to_yf

LOG = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "sectors"
SECTOR_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
# A failed lookup must not be cached as confidently as a successful one. Every
# provider error funnels into UNKNOWN, so one bout of yfinance rate limiting
# would otherwise pin a whole universe to a single bucket for a month - which
# is not a visible failure, it silently turns sector-neutral ranking into a
# no-op. A short negative TTL still spares a sweep from re-requesting a
# genuinely sector-less symbol hundreds of times, and clears by the next day.
UNKNOWN_SECTOR_TTL_SECONDS = 24 * 60 * 60  # 1 day
UNKNOWN_SECTOR = "UNKNOWN"
# FMP accepts a comma-separated symbol list; a hundred keeps the URL well under
# any gateway limit while cutting a 900-name universe to nine requests.
FMP_PROFILE_BATCH = 100
FMP_TIMEOUT_SECONDS = 30.0

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


def fetch_fmp_sectors(yf_symbols: Sequence[str]) -> dict[str, str]:
    """Resolve sectors for many symbols at once via FMP's profile endpoint.

    Returns only the symbols FMP answered for; a caller must treat an absent
    symbol as unresolved rather than as UNKNOWN, because "FMP does not cover
    this ticker" and "this request failed" have to stay distinguishable from a
    real sectorless instrument. Returns ``{}`` when no API key is configured,
    which sends every symbol down the yfinance path unchanged.
    """
    from screener import fmp

    api_key = fmp.resolve_api_key()
    if not api_key:
        return {}
    client = fmp.FmpClient(api_key, timeout=FMP_TIMEOUT_SECONDS)
    resolved: dict[str, str] = {}
    unique = list(dict.fromkeys(s for s in yf_symbols if s))
    for i in range(0, len(unique), FMP_PROFILE_BATCH):
        batch = unique[i : i + FMP_PROFILE_BATCH]
        try:
            payload = client.get(f"profile/{','.join(batch)}")
        except Exception as exc:  # noqa: BLE001 — provider failure -> fallback
            LOG.debug("FMP profile batch failed (%d symbols): %s", len(batch), exc)
            continue
        for row in payload if isinstance(payload, list) else []:
            if not isinstance(row, Mapping):
                continue
            symbol = row.get("symbol")
            sector = row.get("sector")
            if isinstance(symbol, str) and isinstance(sector, str) and sector.strip():
                resolved[symbol] = sector.strip()
    return resolved


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
    sector = sector.strip()
    if sector == UNKNOWN_SECTOR and not is_fresh(path, UNKNOWN_SECTOR_TTL_SECONDS):
        return None
    return sector


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
    # Dedupe lookups when multiple TV symbols map to the same YF symbol.
    by_yf: dict[str, list[str]] = {}
    for raw in tickers:
        ticker = raw.strip()
        if not ticker:
            continue
        by_yf.setdefault(tv_to_yf(ticker, market), []).append(ticker)

    # Batch the symbols that are not already cached through FMP in one pass, so
    # the per-symbol path below only handles what FMP could not answer. Skipped
    # entirely when the caller injected a fetcher (tests, and any call site that
    # deliberately chose its own source).
    yf_to_sector: dict[str, str] = {}
    if use_cache and info_fetcher is None:
        missing = [s for s in by_yf if _load_cached_sector(s) is None]
        for symbol, sector in fetch_fmp_sectors(missing).items():
            if symbol in by_yf:
                _store_cached_sector(symbol, sector)
                yf_to_sector[symbol] = sector

    out: dict[str, str] = {}
    for yf_symbol, raw_tickers in by_yf.items():
        if yf_symbol not in yf_to_sector:
            yf_to_sector[yf_symbol] = _resolve_one(
                yf_symbol, use_cache=use_cache, info_fetcher=fetcher
            )
        for ticker in raw_tickers:
            out[ticker] = yf_to_sector[yf_symbol]
    return out
