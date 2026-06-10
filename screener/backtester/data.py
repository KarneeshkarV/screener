"""Historical OHLCV fetching adapter.

Defines the ``PriceFetcher`` protocol used by the engine, a default
``YFinancePriceFetcher`` with an on-disk parquet cache, and a small symbol
mapper that translates TradingView-style tickers to yfinance tickers.

Tests inject a ``StubPriceFetcher`` that returns pre-built synthetic frames;
the engine never depends directly on yfinance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import date, datetime
import io
import os
from pathlib import Path
from typing import Iterable, Optional, Protocol

import pandas as pd
import requests

from screener.resilience import call_with_resilience


CACHE_DIR = Path.home() / ".screener" / "prices"
FMP_CACHE_DIR = Path.home() / ".screener" / "fmp_prices"
_DOTENV_LOADED = False
_YFINANCE_CONFIGURED = False


def _configure_yfinance() -> None:
    """Point yfinance tz cache at tmpfs and avoid peewee SQLite lookups.

    The tz-cache dummy swap relies on yfinance private symbols
    (``_TzCacheManager`` / ``_TzCacheDummy``); upstream renames have happened
    in the past. We attempt the swap defensively and degrade to a warning if
    the symbols disappear — the bulk download still works without it, just a
    bit slower on first call. ``_YFINANCE_CONFIGURED`` is set regardless so we
    don't keep retrying the same monkey-patch on every fetch.
    """
    global _YFINANCE_CONFIGURED
    if _YFINANCE_CONFIGURED:
        return
    try:
        import yfinance as yf
        import yfinance.cache as yf_cache

        if os.path.isdir("/dev/shm"):
            yf.set_tz_cache_location("/dev/shm/screener-yftz")
        try:
            tz_cache_manager = yf_cache._TzCacheManager
            tz_cache_dummy = yf_cache._TzCacheDummy
        except AttributeError:
            from screener.logging_config import get_logger

            get_logger(__name__).warning(
                "yfinance_tz_cache_patch_unavailable",
                reason="missing private _TzCacheManager/_TzCacheDummy",
            )
        else:
            tz_cache_manager.get_tz_cache = classmethod(lambda cls: tz_cache_dummy())
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any swap failure
        from screener.logging_config import get_logger

        get_logger(__name__).warning(
            "yfinance_configure_failed",
            error=repr(exc),
        )
    finally:
        _YFINANCE_CONFIGURED = True


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
# Supplementary columns emitted only by the split-only / raw regimes. They
# are always optional on a bars DataFrame — callers should treat a missing
# column the same as a column of zeros.
CORPORATE_ACTION_COLUMNS = ["dividend", "split_factor", "stock_splits"]


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from the project .env if not exported."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def load_env_file() -> None:
    """Load simple KEY=VALUE pairs from the project .env if not exported."""
    _load_env_file()


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


def _cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}.parquet"


def _naive_normalized_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Normalize to tz-naive midnight without re-parsing an already-datetime index.

    ``pd.to_datetime()`` on a ``DatetimeIndex`` is a no-op conversion, but its
    ``should_cache`` heuristic iterates the whole index in Python — the single
    largest leaf in the sp500 profiles. Skip it when the index is already
    datetime, and only ``tz_localize`` when actually tz-aware. ``.normalize()``
    stays (it is vectorized C, not the hotspot).
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _load_cached(ticker: str, cache_dir: Path = CACHE_DIR) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = _naive_normalized_index(df.index)
        # Clean NaN-OHLCV rows that older cache writes may have persisted, so a
        # cache hit can't reintroduce the NaN bars that _normalize_frame drops.
        price_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        if price_cols:
            df = df.dropna(subset=price_cols)
        return df
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def _save_cache(ticker: str, df: pd.DataFrame, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_cache_path(ticker, cache_dir))
    except (OSError, ValueError):
        # parquet failure is non-fatal; just skip caching
        pass


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OHLCV_COLUMNS,
        index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
    )


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_ohlcv_frame()
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
    out.index = _naive_normalized_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # Drop bars with no valid OHLCV (yfinance emits NaN rows for halts,
    # illiquid/delisting tails, and multi-ticker index-union gaps). These are
    # not tradeable bars: an entry/exit fill or mark-to-market landing on one
    # propagates NaN into trade PnL and the equity endpoint. Mirrors the FMP
    # normalize path, which already drops these.
    price_cols = [c for c in OHLCV_COLUMNS if c in out.columns]
    if price_cols:
        out = out.dropna(subset=price_cols)
    return out


def _merge_cached(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], axis=0)
    if merged.empty:
        return merged
    merged.index = _naive_normalized_index(merged.index)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def _has_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    if df is None or df.empty:
        return False
    in_range = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return (
        not in_range.empty
        and in_range.index.min() <= start_ts + pd.Timedelta(days=3)
        and in_range.index.max() >= end_ts - pd.Timedelta(days=3)
    )


def _split_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {ticker: _empty_ohlcv_frame() for ticker in tickers}
    if not isinstance(raw.columns, pd.MultiIndex):
        ticker = tickers[0] if tickers else ""
        return {ticker: _normalize_frame(raw)}

    frames: dict[str, pd.DataFrame] = {}
    level_values = [
        set(raw.columns.get_level_values(i)) for i in range(raw.columns.nlevels)
    ]
    for ticker in tickers:
        frame = pd.DataFrame()
        for level, values in enumerate(level_values):
            if ticker in values:
                selected = raw.xs(ticker, level=level, axis=1, drop_level=True)
                frame = (
                    selected.to_frame() if isinstance(selected, pd.Series) else selected
                )
                break
        frames[ticker] = _normalize_frame(frame)
    return frames


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
        batch_size: int = 75,
        refresh: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.batch_size = max(1, int(batch_size))
        self.refresh = bool(refresh)
        self.max_workers = max(1, int(max_workers))

    def _cache_key(self, ticker: str) -> str:
        return ticker if self.auto_adjust else f"{ticker}__raw"

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tickers = [t for t in tickers if t]
        results: dict[str, pd.DataFrame] = {}
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        cached_by_ticker: dict[str, pd.DataFrame] = {}
        missing: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}

        for ticker in tickers:
            cache_key = self._cache_key(ticker)
            cached = None if self.refresh else _load_cached(cache_key, self.cache_dir)
            if cached is not None and not cached.empty:
                cached_by_ticker[ticker] = cached
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts)
            ):
                results[ticker] = cached.loc[
                    (cached.index >= start_ts) & (cached.index <= end_ts)
                ]
                continue

            fetch_start, fetch_end = start_ts, end_ts
            if not self.refresh and cached is not None and not cached.empty:
                min_cached = cached.index.min()
                max_cached = cached.index.max()
                if min_cached <= start_ts + pd.Timedelta(
                    days=3
                ) and max_cached < end_ts - pd.Timedelta(days=3):
                    fetch_start = max_cached + pd.Timedelta(days=1)
                elif max_cached >= end_ts - pd.Timedelta(
                    days=3
                ) and min_cached > start_ts + pd.Timedelta(days=3):
                    fetch_end = min_cached - pd.Timedelta(days=1)
            missing.setdefault((fetch_start, fetch_end), []).append(ticker)

        if not missing:
            return results

        _configure_yfinance()
        import yfinance as yf  # lazy import so tests without yfinance still run

        jobs: list[tuple[pd.Timestamp, pd.Timestamp, list[str]]] = []
        for (fetch_start, fetch_end), group in missing.items():
            for i in range(0, len(group), self.batch_size):
                jobs.append((fetch_start, fetch_end, group[i : i + self.batch_size]))

        def download_job(
            job: tuple[pd.Timestamp, pd.Timestamp, list[str]],
        ) -> tuple[list[str], pd.DataFrame]:
            fetch_start, fetch_end, batch = job
            download_kwargs = dict(
                start=fetch_start,
                end=fetch_end + pd.Timedelta(days=1),
                auto_adjust=self.auto_adjust,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if not self.auto_adjust:
                download_kwargs["actions"] = True
            target = " ".join(batch) if len(batch) > 1 else batch[0]
            raw = call_with_resilience(
                "yfinance",
                f"download {len(batch)} ticker(s)",
                lambda: yf.download(target, **download_kwargs),
                fallback=pd.DataFrame(),
            )
            return batch, raw

        # yfinance prints expected "possibly delisted" messages directly to
        # stderr for empty pre-listing ranges. The empty frame is enough for
        # FallbackPriceFetcher to call FMP, so keep the lab/CLI output focused
        # on actionable diagnostics. A single process-wide redirect covers the
        # worker threads too; per-batch redirects would race when batches
        # download concurrently.
        with contextlib.redirect_stderr(io.StringIO()):
            if len(jobs) == 1:
                downloads = [download_job(jobs[0])]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(jobs))
                ) as pool:
                    downloads = list(pool.map(download_job, jobs))

        # Each ticker belongs to exactly one job, so merge + cache writes can
        # stay on the main thread without coordination.
        for batch, raw in downloads:
            downloaded = _split_download(raw, batch)
            for ticker in batch:
                cache_key = self._cache_key(ticker)
                norm = downloaded.get(ticker, _empty_ohlcv_frame())
                merged = _merge_cached(cached_by_ticker.get(ticker), norm)
                if not merged.empty:
                    _save_cache(cache_key, merged, self.cache_dir)
                results[ticker] = merged.loc[
                    (merged.index >= start_ts) & (merged.index <= end_ts)
                ]
        return results


def _fmp_cache_key(ticker: str, auto_adjust: bool) -> str:
    suffix = "" if auto_adjust else "__raw"
    return f"fmp_{ticker}{suffix}"


def _normalize_fmp_historical(payload: object, auto_adjust: bool) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    rows = payload.get("historical")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    rename = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adjClose": "adj_close",
    }
    keep = [source for source in rename if source in df.columns]
    out = df[["date", *keep]].rename(columns=rename).copy()
    out.index = pd.to_datetime(out.pop("date"), errors="coerce")
    out = out[out.index.notna()]
    out.index = out.index.tz_localize(None).normalize()

    for col in [*OHLCV_COLUMNS, "adj_close"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if auto_adjust and "adj_close" in out.columns and "close" in out.columns:
        factor = out["adj_close"] / out["close"].replace(0, pd.NA)
        for col in ["open", "high", "low", "close"]:
            if col in out.columns:
                out[col] = out[col] * factor
    keep_cols = [col for col in [*OHLCV_COLUMNS, "adj_close"] if col in out.columns]
    out = out[keep_cols].dropna(
        subset=[col for col in OHLCV_COLUMNS if col in out.columns]
    )
    return out[~out.index.duplicated(keep="last")].sort_index()


class FMPPriceFetcher:
    """Fetch daily OHLCV from Financial Modeling Prep.

    The API key is read from ``FMP_API_KEY`` unless passed explicitly.
    """

    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Optional[Path] = None,
        auto_adjust: bool = True,
        refresh: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required to use the FMP price fetcher")
        self.cache_dir = cache_dir or FMP_CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.refresh = bool(refresh)
        self.session = session or requests.Session()

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        results: dict[str, pd.DataFrame] = {}

        for ticker in [t for t in tickers if t]:
            cache_key = _fmp_cache_key(ticker, self.auto_adjust)
            cached = None if self.refresh else _load_cached(cache_key, self.cache_dir)
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts)
            ):
                results[ticker] = cached.loc[
                    (cached.index >= start_ts) & (cached.index <= end_ts)
                ]
                continue

            def request_payload() -> object:
                response = self.session.get(
                    f"{self.base_url}/{ticker}",
                    params={
                        "from": start_ts.date().isoformat(),
                        "to": end_ts.date().isoformat(),
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

            payload = call_with_resilience(
                "fmp",
                f"historical prices {ticker}",
                request_payload,
                fallback={},
            )
            norm = _normalize_fmp_historical(payload, self.auto_adjust)
            merged = _merge_cached(cached, norm)
            if not merged.empty:
                _save_cache(cache_key, merged, self.cache_dir)
                results[ticker] = merged.loc[
                    (merged.index >= start_ts) & (merged.index <= end_ts)
                ]
            else:
                results[ticker] = pd.DataFrame(columns=OHLCV_COLUMNS)
        return results


class FallbackPriceFetcher:
    """Use a primary fetcher first and fill missing ticker frames from fallback."""

    def __init__(self, primary: PriceFetcher, fallback: PriceFetcher) -> None:
        self.primary = primary
        self.fallback = fallback

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        ticker_list = [ticker for ticker in tickers if ticker]
        primary_results = self.primary.fetch(ticker_list, start, end)
        missing = [
            ticker
            for ticker in ticker_list
            if ticker not in primary_results
            or primary_results[ticker] is None
            or primary_results[ticker].empty
        ]
        if not missing:
            return primary_results

        fallback_results = self.fallback.fetch(missing, start, end)
        results = dict(primary_results)
        for ticker in missing:
            frame = fallback_results.get(ticker)
            if frame is not None and not frame.empty:
                results[ticker] = frame
            else:
                results.setdefault(ticker, pd.DataFrame(columns=OHLCV_COLUMNS))
        return results


def build_price_fetcher(
    provider: str | None = None,
    *,
    auto_adjust: bool = True,
    refresh: bool = False,
) -> PriceFetcher:
    _load_env_file()
    resolved = (provider or os.environ.get("SCREENER_PRICE_PROVIDER") or "auto").lower()
    if resolved in {"auto", "default"}:
        primary = YFinancePriceFetcher(auto_adjust=auto_adjust, refresh=refresh)
        if os.environ.get("FMP_API_KEY"):
            fallback = FMPPriceFetcher(auto_adjust=auto_adjust, refresh=refresh)
            return FallbackPriceFetcher(primary, fallback)
        return primary
    if resolved in {"yf", "yfinance"}:
        return YFinancePriceFetcher(auto_adjust=auto_adjust, refresh=refresh)
    if resolved in {"fmp", "financialmodelingprep"}:
        return FMPPriceFetcher(auto_adjust=auto_adjust, refresh=refresh)
    raise ValueError(f"Unknown price provider: {provider}")


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
        return pd.Series(
            index=pd.DatetimeIndex([], name="date"), dtype=float, name=symbol
        )
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
