"""Data loading helpers for the research Pine runner."""

from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.data import build_price_fetcher, tv_to_yf
from screener.scanner import scan as _tv_scan

_FETCHER = build_price_fetcher()


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    market: str,
    refresh: bool = False,
) -> pd.DataFrame | None:
    yf_sym = ticker if ticker.startswith("^") else tv_to_yf(ticker, market)
    fetcher = build_price_fetcher(refresh=True) if refresh else _FETCHER
    frames = fetcher.fetch([yf_sym], start, end)
    df = frames.get(yf_sym)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    return df


def load_universe(market: str) -> list[str]:
    from tradingview_screener import col

    # Price floor strips OTC sub-penny tickers that volume-rank to the top.
    price_floor = {"us": 5.0, "india": 50.0}.get(market, 5.0)
    filters = [col("type") == "stock", col("close") >= price_floor]
    _total, df = _tv_scan(market=market, filters=filters, limit=500, order_by="volume")
    return [str(t) for t in df["name"].dropna().tolist()]
