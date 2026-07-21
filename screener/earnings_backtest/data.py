"""Earnings-backtest universe and price/volume loaders.

This module owns the earnings backtest's universe resolution and OHLCV
fetching. The two other data concerns live in focused modules:

* :mod:`screener.earnings_backtest.earnings_dates` — earnings/result-date
  acquisition (yfinance, NSE announcements, screener.in) + the batch collector.
* :mod:`screener.earnings_backtest.sentiment` — analyst and options-implied
  (IV) sentiment providers.

Import those names from their canonical modules directly.

Designed for batch processing under tight RAM constraints (~2 GB).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from screener.backtester.data import YFinancePriceFetcher

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
    "fetch_price_data",
    "load_nifty500",
    "load_sp500",
    "load_universe",
]
