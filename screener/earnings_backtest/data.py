"""Data acquisition for earnings backtest (back-compat facade).

This module is the stable seam for the earnings backtest's data layer. The
implementation now lives in two focused modules:

* :mod:`screener.earnings_backtest.earnings_dates` — earnings/result-date
  acquisition (yfinance, NSE announcements, screener.in) + the batch collector.
* :mod:`screener.earnings_backtest.sentiment` — analyst and options-implied
  (IV) sentiment providers.

Both are re-exported here so every existing import
(``from screener.earnings_backtest.data import ...``) and test patch target
(``monkeypatch.setattr(data, ...)``) keeps working. The shared seam kept here —
the ``cached_json_call`` / yfinance handles, the JSON coercion helper, the
constants, the universe loaders, and the price fetcher — is looked up as
``data.<name>`` by the split modules so those patches take effect across the
package boundary.

Designed for batch processing under tight RAM constraints (~2 GB).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from screener.backtester.data import (
    YFinancePriceFetcher,
    _configure_yfinance,
)
from screener.cache import cached_json_call
from screener.unusual_volume.option_chain import fetch_option_chain

if TYPE_CHECKING:
    from screener.earnings_backtest.earnings_dates import (
        INDIA_EARNINGS_FILING_LAG_DAYS,
        _earnings_from_records,
        _earnings_rows_for_ticker,
        _earnings_to_records,
        _fetch_openscreener_earnings_rows,
        _fetch_yf_earnings_rows,
        _openscreener_earnings_rows_for_ticker,
        collect_earnings_events,
        fetch_earnings_dates_nse,
        fetch_earnings_dates_openscreener,
        fetch_earnings_dates_yf,
    )
    from screener.earnings_backtest.sentiment import (
        fetch_analyst_sentiment,
        fetch_iv_sentiment,
        fetch_iv_sentiment_nse,
        fetch_iv_sentiment_yf,
    )

logger = logging.getLogger(__name__)

EARNINGS_CACHE_DAYS = 30
SENTIMENT_CACHE_DAYS = 1
MAX_WORKERS = 12


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


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


# ── Back-compat re-exports ──────────────────────────────────────────────
# Resolved lazily (PEP 562): the split modules import THIS module at their top
# and look the shared seam up as ``data.<name>``, so an eager from-import here
# would fail whenever a split module is imported first (circular import). The
# lazy hook keeps both import orders working; resolved names are cached into
# ``globals()`` so subsequent lookups (and monkeypatching) behave like normal
# module attributes.

_EARNINGS_DATES_EXPORTS = frozenset(
    {
        "INDIA_EARNINGS_FILING_LAG_DAYS",
        "_earnings_from_records",
        "_earnings_rows_for_ticker",
        "_earnings_to_records",
        "_fetch_openscreener_earnings_rows",
        "_fetch_yf_earnings_rows",
        "_openscreener_earnings_rows_for_ticker",
        "collect_earnings_events",
        "fetch_earnings_dates_nse",
        "fetch_earnings_dates_openscreener",
        "fetch_earnings_dates_yf",
    }
)
_SENTIMENT_EXPORTS = frozenset(
    {
        "fetch_analyst_sentiment",
        "fetch_iv_sentiment",
        "fetch_iv_sentiment_nse",
        "fetch_iv_sentiment_yf",
    }
)


def __getattr__(name: str) -> Any:
    if name in _EARNINGS_DATES_EXPORTS:
        from screener.earnings_backtest import earnings_dates as _mod
    elif name in _SENTIMENT_EXPORTS:
        from screener.earnings_backtest import sentiment as _mod  # type: ignore[no-redef]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_mod, name)
    globals()[name] = value
    return value


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
    "fetch_earnings_dates_yf",
    "fetch_earnings_dates_nse",
    "fetch_earnings_dates_openscreener",
    "fetch_analyst_sentiment",
    "fetch_iv_sentiment",
    "fetch_iv_sentiment_yf",
    "fetch_iv_sentiment_nse",
]
