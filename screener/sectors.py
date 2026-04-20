"""Sector tagging + bull-run detection for universe-wide filtering.

Provides two things used by the historical backtest:

1. ``sector_map(market, tickers)`` — returns a ``{ticker: sector}`` dict,
   looking up each ticker's sector via yfinance ``Ticker.info`` and caching
   the result to ``~/.screener/sectors/{market}.json``.

2. ``bullish_sectors(sector_by_ticker, ohlcv_by_ticker, as_of, lookback)`` —
   returns the set of sectors considered to be in a bull run at ``as_of``,
   defined as sectors whose mean N-bar return across their member tickers
   is **positive AND at or above the median sector return**.  Using the
   universe-derived mean (rather than an external ETF) avoids having to
   fetch / map separate sector-index symbols for both US and India.

The sector filter is a universe-wide gate applied after criterion pass
but before top-N selection — all strategies automatically inherit it.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path.home() / ".screener" / "sectors"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(market: str) -> Path:
    return _CACHE_DIR / f"{market}.json"


def _load_cache(market: str) -> dict[str, str]:
    p = _cache_path(market)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(market: str, data: dict[str, str]) -> None:
    _cache_path(market).write_text(json.dumps(data, sort_keys=True))


def _yf_symbol(market: str, ticker: str) -> str:
    return ticker if market == "us" else f"{ticker}.NS"


def _fetch_sector(market: str, ticker: str) -> str:
    try:
        info = yf.Ticker(_yf_symbol(market, ticker)).info
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def sector_map(
    market: str,
    tickers: list[str],
    max_workers: int = 16,
) -> dict[str, str]:
    """Return ``{ticker: sector}`` for *tickers*, persisting new lookups to disk."""
    cache = _load_cache(market)
    missing = [t for t in tickers if t not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_sector, market, t): t for t in missing}
            for fut in as_completed(futures):
                ticker = futures[fut]
                cache[ticker] = fut.result()
        _save_cache(market, cache)
    return {t: cache.get(t, "Unknown") for t in tickers}


def bullish_sectors(
    sector_by_ticker: dict[str, str],
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback: int = 63,
) -> tuple[set[str], dict[str, float]]:
    """Identify sectors in a bull run as of *as_of*.

    For each sector, compute the equal-weighted mean of its members'
    *lookback*-bar close-to-close returns (using data on or before
    *as_of*).  A sector is "bullish" if its mean return is positive
    **and** at or above the median across all sectors.

    Returns ``(bull_set, sector_mean_returns)``.
    """
    as_of_ts = pd.Timestamp(as_of).normalize()
    per_sector: dict[str, list[float]] = {}
    for ticker, df in ohlcv_by_ticker.items():
        sec = sector_by_ticker.get(ticker)
        if not sec or sec == "Unknown":
            continue
        dates = pd.to_datetime(df["date"]).dt.normalize()
        sliced = df[dates <= as_of_ts]
        if len(sliced) < lookback + 1:
            continue
        close = pd.to_numeric(sliced["close"], errors="coerce").ffill()
        if len(close) < lookback + 1 or close.iloc[-lookback - 1] <= 0:
            continue
        ret = float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)
        if ret != ret:  # NaN
            continue
        per_sector.setdefault(sec, []).append(ret)

    sector_mean = {s: sum(rs) / len(rs) for s, rs in per_sector.items() if rs}
    if not sector_mean:
        return set(), {}

    vals = sorted(sector_mean.values())
    median = vals[len(vals) // 2]
    bull = {s for s, m in sector_mean.items() if m > 0 and m >= median}
    return bull, sector_mean
