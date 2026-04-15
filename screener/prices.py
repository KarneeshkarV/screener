import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path.home() / ".screener" / "prices"

_TV_CLIENT = None
_TV_WARNED = False


def _tv_client():
    global _TV_CLIENT
    if _TV_CLIENT is not None:
        return _TV_CLIENT
    from tvDatafeed import TvDatafeed

    username = os.environ.get("TV_USERNAME")
    password = os.environ.get("TV_PASSWORD")
    try:
        if username and password:
            _TV_CLIENT = TvDatafeed(username=username, password=password)
        else:
            _TV_CLIENT = TvDatafeed()
    except Exception as e:
        warnings.warn(f"tvdatafeed init failed: {e}")
        _TV_CLIENT = False
    return _TV_CLIENT


def _cache_path(market: str, tv_symbol: str) -> Path:
    safe = tv_symbol.replace(":", "__").replace("/", "_")
    d = CACHE_DIR / market
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.parquet"


def _split_symbol(tv_symbol: str) -> tuple[Optional[str], str]:
    if ":" in tv_symbol:
        ex, sym = tv_symbol.split(":", 1)
        return ex, sym
    return None, tv_symbol


def _yf_ticker(tv_symbol: str, market: str) -> str:
    _, sym = _split_symbol(tv_symbol)
    if market == "india":
        return f"{sym}.NS"
    return sym


def _fetch_yfinance(tv_symbol: str, market: str, start: date, end: date) -> Optional[pd.DataFrame]:
    import yfinance as yf

    yt = _yf_ticker(tv_symbol, market)
    try:
        df = yf.download(
            yt,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        warnings.warn(f"yfinance fetch failed for {yt}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    return df[["open", "high", "low", "close", "adj_close", "volume"]].reset_index()


def _fetch_tvdatafeed(tv_symbol: str, market: str, start: date, end: date) -> Optional[pd.DataFrame]:
    global _TV_WARNED
    client = _tv_client()
    if client is False or client is None:
        return None

    from tvDatafeed import Interval

    ex, sym = _split_symbol(tv_symbol)
    if ex is None:
        ex = "NSE" if market == "india" else "NASDAQ"

    days = (end - start).days + 5
    n_bars = min(max(days, 100), 5000)
    try:
        df = client.get_hist(symbol=sym, exchange=ex, interval=Interval.in_daily, n_bars=n_bars)
    except Exception as e:
        if not _TV_WARNED:
            warnings.warn(f"tvdatafeed fetch failed: {e}")
            _TV_WARNED = True
        return None

    if df is None or df.empty:
        return None

    df = df.reset_index().rename(
        columns={
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["adj_close"] = df["close"]
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    except Exception:
        return None


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        warnings.warn(f"cache write failed at {path}: {e}")


def fetch_ohlcv(
    tv_symbol: str,
    start: date,
    end: date,
    market: str,
    refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Return daily OHLCV DataFrame for tv_symbol in [start, end] (inclusive). None on failure."""
    req_start, req_end = start, end
    path = _cache_path(market, tv_symbol)
    cached = None if refresh else _load_cache(path)

    need_fetch = True
    fetch_start, fetch_end = start, end
    if cached is not None and not cached.empty:
        cmin = cached["date"].min().date()
        cmax = cached["date"].max().date()
        if cmin <= req_start and cmax >= req_end:
            need_fetch = False
        else:
            fetch_start = min(req_start, cmin)
            fetch_end = max(req_end, cmax, date.today())

    if need_fetch:
        providers = (
            [_fetch_yfinance, _fetch_tvdatafeed]
            if market == "us"
            else [_fetch_tvdatafeed, _fetch_yfinance]
        )
        fetched = None
        for fn in providers:
            fetched = fn(tv_symbol, market, fetch_start, fetch_end)
            if fetched is not None and not fetched.empty:
                break

        if fetched is None or fetched.empty:
            if cached is None or cached.empty:
                return None
        else:
            if cached is not None and not cached.empty:
                merged = pd.concat([cached, fetched], ignore_index=True)
                merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
            else:
                merged = fetched.sort_values("date").reset_index(drop=True)
            _save_cache(path, merged)
            cached = merged

    if cached is None or cached.empty:
        return None

    mask = (cached["date"].dt.date >= req_start) & (cached["date"].dt.date <= req_end)
    result = cached.loc[mask].reset_index(drop=True)
    if result.empty:
        return None
    return result


def fetch_adj_close_matrix(
    tv_symbols: list[str],
    start: date,
    end: date,
    market: str,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (wide adj_close DataFrame date x ticker, list of tickers that failed)."""
    series: dict[str, pd.Series] = {}
    failed: list[str] = []
    for sym in tv_symbols:
        df = fetch_ohlcv(sym, start, end, market, refresh=refresh)
        if df is None or df.empty:
            failed.append(sym)
            continue
        s = pd.Series(df["adj_close"].values, index=pd.to_datetime(df["date"]).values, name=sym)
        series[sym] = s

    if not series:
        return pd.DataFrame(), failed

    wide = pd.concat(series.values(), axis=1)
    wide.columns = list(series.keys())
    wide.index.name = "date"
    wide = wide.sort_index()
    return wide, failed
