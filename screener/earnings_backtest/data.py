"""Stable one-way facade for earnings-backtest data acquisition.

This module is the stable seam for the earnings backtest's data layer. The
implementation now lives in two focused modules:

* :mod:`screener.earnings_backtest.earnings_dates` — earnings/result-date
  acquisition (yfinance, NSE announcements, screener.in) + the batch collector.
* :mod:`screener.earnings_backtest.sentiment` — analyst and options-implied
  (IV) sentiment providers.

Both are eagerly re-exported here for existing imports. Implementations depend
only on canonical helpers and never import this facade back, so import order is
ordinary and there is no lazy ``__getattr__`` cycle.

Designed for batch processing under tight RAM constraints (~2 GB).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf  # re-exported compatibility handle

from screener.backtester.data import (
    YFinancePriceFetcher,
    _configure_yfinance as _configure_yfinance,
)
from screener.cache import cached_json_call as cached_json_call
from screener.earnings_backtest.common import (
    EARNINGS_CACHE_DAYS,
    MAX_WORKERS,
    SENTIMENT_CACHE_DAYS,
    jsonable as _jsonable,
)
from screener.earnings_backtest.earnings_dates import (
    INDIA_EARNINGS_FILING_LAG_DAYS,
    _earnings_from_records,
    _earnings_rows_for_ticker,
    _earnings_to_records,
    _fetch_openscreener_earnings_rows,
    _fetch_yf_earnings_rows,
    _openscreener_earnings_rows_for_ticker,
    collect_earnings_events,
    events_to_dates_map,
    fetch_earnings_dates_nse,
    fetch_earnings_dates_openscreener,
    fetch_earnings_dates_yf,
    fetch_next_earnings_dates,
    load_earnings_dates_map,
    next_earnings_date,
)
from screener.earnings_backtest.sentiment import (
    fetch_analyst_sentiment,
    fetch_iv_sentiment,
    fetch_iv_sentiment_nse,
    fetch_iv_sentiment_yf,
)
from screener.unusual_volume.option_chain import (
    fetch_option_chain as fetch_option_chain,
)

logger = logging.getLogger(__name__)

# ── Universe loaders ────────────────────────────────────────────────────


def load_sp500() -> list[str]:
    """Return current S&P 500 ticker list."""
    from screener.universes import load_current_universe

    univ = load_current_universe("sp500")
    return list(univ.symbols)


def load_nifty500() -> list[str]:
    """Return Nifty 500 ticker list with .NS suffix."""
    from screener.universes import load_current_universe

    univ = load_current_universe("nifty500")
    return list(univ.symbols)


def load_universe(market: str) -> list[str]:
    if market == "us":
        return load_sp500()
    if market == "india":
        return load_nifty500()
    raise ValueError(f"Unknown market: {market!r}")


# ── Price / volume data ─────────────────────────────────────────────────


def fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
    fetcher: Optional[YFinancePriceFetcher] = None,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV bars for *tickers* from *start* to *end*."""
    if fetcher is None:
        fetcher = YFinancePriceFetcher(auto_adjust=True)

    all_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = fetcher.fetch(batch, start, end)
        all_data.update(data)
        for k in list(data.keys()):
            if data[k].empty:
                del data[k]
    return all_data


__all__ = [
    "EARNINGS_CACHE_DAYS",
    "SENTIMENT_CACHE_DAYS",
    "MAX_WORKERS",
    "INDIA_EARNINGS_FILING_LAG_DAYS",
    "YFinancePriceFetcher",
    "_configure_yfinance",
    "cached_json_call",
    "fetch_option_chain",
    "yf",
    "_jsonable",
    "_earnings_to_records",
    "_earnings_from_records",
    "_earnings_rows_for_ticker",
    "_fetch_yf_earnings_rows",
    "_openscreener_earnings_rows_for_ticker",
    "_fetch_openscreener_earnings_rows",
    "load_sp500",
    "load_nifty500",
    "load_universe",
    "fetch_price_data",
    "collect_earnings_events",
    "events_to_dates_map",
    "fetch_earnings_dates_yf",
    "fetch_earnings_dates_nse",
    "fetch_earnings_dates_openscreener",
    "fetch_next_earnings_dates",
    "load_earnings_dates_map",
    "next_earnings_date",
    "fetch_analyst_sentiment",
    "fetch_iv_sentiment",
    "fetch_iv_sentiment_yf",
    "fetch_iv_sentiment_nse",
]
