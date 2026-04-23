"""Historical OHLCV fetching adapter.

Defines the ``PriceFetcher`` protocol used by the engine, a default
``YFinancePriceFetcher`` with an on-disk parquet cache, and a small symbol
mapper that translates TradingView-style tickers to yfinance tickers.

Tests inject a ``StubPriceFetcher`` that returns pre-built synthetic frames;
the engine never depends directly on yfinance.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Protocol

import pandas as pd


CACHE_DIR = Path.home() / ".screener" / "prices"


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
# Supplementary columns emitted only by the split-only / raw regimes. They
# are always optional on a bars DataFrame — callers should treat a missing
# column the same as a column of zeros.
CORPORATE_ACTION_COLUMNS = ["dividend", "split_factor", "stock_splits"]


class PriceFetcher(Protocol):
    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        """Return dict of yf-style ticker → OHLCV DataFrame indexed by date.

        Frames must have lowercase columns: open, high, low, close, volume.
        ``adj_close`` is optional; absent means ``close`` is already adjusted.
        """


def tv_to_yf(symbol: str, market: str) -> str:
    """Translate a TradingView-style symbol to a yfinance symbol.

    Examples:
      'NSE:RELIANCE' + india → 'RELIANCE.NS'
      'BSE:TCS'     + india → 'TCS.BO'
      'NASDAQ:AAPL' + us    → 'AAPL'
      'AAPL'        + us    → 'AAPL'
      'RELIANCE'    + india → 'RELIANCE.NS'
    """
    sym = symbol.strip().upper()
    if ":" in sym:
        exch, rest = sym.split(":", 1)
        if exch == "NSE":
            return f"{rest}.NS"
        if exch == "BSE":
            return f"{rest}.BO"
        return rest
    if market == "india" and "." not in sym:
        return f"{sym}.NS"
    return sym


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _load_cached(ticker: str) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df
    except Exception:
        return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_cache_path(ticker))
    except Exception:
        # parquet failure is non-fatal; just skip caching
        pass


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    # yfinance returns MultiIndex columns when multiple tickers; callers should
    # split first. For single-ticker frames, columns are plain strings.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)
    rename = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename)
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    out = df[keep].copy()
    if "adj_close" in df.columns:
        out["adj_close"] = df["adj_close"]
    # Preserve explicit corporate-action columns if present (auto_adjust=False
    # path). Split-factor is derived from stock_splits when available.
    if "dividends" in df.columns:
        out["dividend"] = df["dividends"].fillna(0.0).astype(float)
    elif "dividend" in df.columns:
        out["dividend"] = df["dividend"].fillna(0.0).astype(float)
    if "stock_splits" in df.columns:
        splits = df["stock_splits"].fillna(0.0).astype(float)
        # yfinance emits the split ratio (e.g. 2.0 for 2:1). Reverse-cumulative
        # product gives the factor that back-adjusts historical prices so they
        # are comparable to the present.
        factor = splits.replace(0.0, 1.0)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        out["split_factor"] = factor.astype(float)
        out["stock_splits"] = splits
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


class YFinancePriceFetcher:
    """Fetches daily OHLCV from yfinance with a parquet on-disk cache.

    Two regimes are supported:

      * ``auto_adjust=True`` (default, legacy) — yfinance back-propagates
        dividends and splits into the OHLC columns. Volume is left raw so a
        downstream ``close * volume`` screen is biased; dividends are
        silently folded into price returns. Matches the historical behaviour
        of the backtester.
      * ``auto_adjust=False`` — raw OHLC are preserved and the separate
        ``Dividends`` / ``Stock Splits`` columns are retained so the engine
        can credit cash dividends explicitly and compute split-adjusted
        prices on demand via ``_normalize_frame``.

    Cached parquet files are keyed by ticker name; switching regimes will not
    collide because the regime is encoded in an optional ``_meta`` suffix
    when ``auto_adjust=False`` is selected.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        auto_adjust: bool = True,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.auto_adjust = bool(auto_adjust)

    def _cache_key(self, ticker: str) -> str:
        return ticker if self.auto_adjust else f"{ticker}__raw"

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf  # lazy import so tests without yfinance still run

        tickers = [t for t in tickers if t]
        results: dict[str, pd.DataFrame] = {}
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        for ticker in tickers:
            cache_key = self._cache_key(ticker)
            cached = _load_cached(cache_key)
            if cached is not None and not cached.empty:
                in_range = cached.loc[
                    (cached.index >= start_ts) & (cached.index <= end_ts)
                ]
                if not in_range.empty and in_range.index.min() <= start_ts + pd.Timedelta(days=3):
                    results[ticker] = in_range
                    continue
            try:
                download_kwargs = dict(
                    start=start_ts,
                    end=end_ts + pd.Timedelta(days=1),
                    auto_adjust=self.auto_adjust,
                    progress=False,
                    threads=False,
                )
                if not self.auto_adjust:
                    download_kwargs["actions"] = True
                raw = yf.download(ticker, **download_kwargs)
            except Exception:
                raw = pd.DataFrame()
            norm = _normalize_frame(raw)
            if norm.empty:
                results[ticker] = norm
                continue
            _save_cache(cache_key, norm)
            results[ticker] = norm.loc[
                (norm.index >= start_ts) & (norm.index <= end_ts)
            ]
        return results


def fetch_benchmark(
    symbol: str, start: date, end: date, fetcher: PriceFetcher
) -> pd.Series:
    """Return a benchmark close-price Series indexed by date.

    Uses the same ``PriceFetcher`` as the portfolio so tests can inject a stub.
    Returns an empty Series if the symbol has no data.
    """
    data = fetcher.fetch([symbol], start, end)
    frame = data.get(symbol)
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=symbol)
    series = frame["close"].astype(float).copy()
    series.name = symbol
    return series


def ensure_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Cannot convert {value!r} to date")
