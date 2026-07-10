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
import time
from typing import Any, Iterable, Optional, Protocol, cast

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from screener.resilience import call_with_resilience

# Re-exported for backward compatibility: several modules and the docs import
# ``tv_to_yf`` from ``screener.backtester.data``. The definition now lives in
# ``screener.symbols``.
from screener.symbols import tv_to_yf as tv_to_yf


CACHE_DIR = Path.home() / ".screener" / "prices"
FMP_CACHE_DIR = Path.home() / ".screener" / "fmp_prices"
_DOTENV_LOADED = False
_YFINANCE_CONFIGURED = False

# Cap yfinance's internal scrape/API request timeout so a stuck provider
# request can't hang a whole download batch.
YFINANCE_TIMEOUT_SECONDS = 5
PRICE_TAIL_TTL_SECONDS = 60 * 60


def _cap_yfinance_request_timeout(
    yf_data: Any, *, seconds: float = YFINANCE_TIMEOUT_SECONDS
) -> None:
    """Wrap ``YfData.get``/``cache_get`` to cap the per-request timeout."""
    original_get = yf_data.YfData.get
    original_cache_get = yf_data.YfData.cache_get

    def capped_get(self, url, params=None, timeout=30):
        timeout = min(float(timeout or seconds), seconds)
        return original_get(self, url, params=params, timeout=timeout)

    def capped_cache_get(self, url, params=None, timeout=30):
        timeout = min(float(timeout or seconds), seconds)
        return original_cache_get(self, url, params=params, timeout=timeout)

    yf_data.YfData.get = capped_get
    yf_data.YfData.cache_get = capped_cache_get


def _configure_yfinance() -> None:
    """Point yfinance tz cache at tmpfs and avoid peewee SQLite lookups.

    The tz-cache dummy swap relies on yfinance private symbols
    (``_TzCacheManager`` / ``_TzCacheDummy``); upstream renames have happened
    in the past. We attempt the swap defensively and degrade to a warning if
    the symbols disappear — the bulk download still works without it, just a
    bit slower on first call. The internal scrape/API request timeout is also
    capped (see :func:`_cap_yfinance_request_timeout`) so a stuck provider
    request can't hang the batch. ``_YFINANCE_CONFIGURED`` is set regardless so
    we don't keep retrying the same monkey-patches on every fetch.
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
        # Cap yfinance's internal scrape/API request timeout.
        try:
            import yfinance.data as yf_data

            _cap_yfinance_request_timeout(yf_data)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any swap failure
            from screener.logging_config import get_logger

            get_logger(__name__).debug("yfinance_timeout_patch_failed", error=str(exc))
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


def _cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}.parquet"


def _naive_normalized_index(idx: pd.Index, interval: str = "1d") -> pd.DatetimeIndex:
    """Normalize to tz-naive index without re-parsing an already-datetime index.

    ``pd.to_datetime()`` on a ``DatetimeIndex`` is a no-op conversion, but its
    ``should_cache`` heuristic iterates the whole index in Python — the single
    largest leaf in the sp500 profiles. Skip it when the index is already
    datetime, and only ``tz_localize`` when actually tz-aware.

    For daily bars (``interval == "1d"``) the index is truncated to midnight via
    ``.normalize()`` — the historical behaviour, one bar per calendar day. For
    intraday intervals normalization is skipped so each bar keeps its
    time-of-day; only the tz is dropped (yfinance returns exchange-local time),
    leaving distinct tz-naive timestamps that the dedup pass no longer collapses.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        if interval == "1d":
            idx = idx.tz_localize(None)
        else:
            # Canonical intraday wall-clock is naive UTC. yfinance batches
            # already arrive UTC-aware, but pin it explicitly so single-ticker
            # downloads (exchange-local tz) and other providers line up on the
            # same simulation calendar.
            idx = idx.tz_convert("UTC").tz_localize(None)
    if interval == "1d":
        return idx.normalize()
    return idx


def _load_cached(
    ticker: str, cache_dir: Path = CACHE_DIR, interval: str = "1d"
) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = _naive_normalized_index(df.index, interval)
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


def _normalize_frame(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
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
    out.index = _naive_normalized_index(out.index, interval)
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


def apply_splits_only_adjustment(
    bars_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Back-adjust OHLC + per-share dividends for splits in the splits_only regime.

    For each frame carrying a ``split_factor`` column (emitted by
    ``_normalize_frame`` from yfinance ``Stock Splits``), historical bars are
    divided by the factor for open/high/low/close and the per-share
    ``dividend`` column, and volume is multiplied by the factor — so a flat
    series across a real 2:1 split stays flat instead of showing a phantom
    -50% step. Frames whose factors are all ``1.0`` are returned untouched
    (fast path).

    Frames lacking a ``split_factor`` are passed through unchanged here; the
    FMP-reconstruction / warning for those lives at the fetch sites.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars_dict.items():
        if frame is None or frame.empty or "split_factor" not in frame.columns:
            out[ticker] = frame
            continue
        factor = frame["split_factor"].astype(float)
        # Fast-path: nothing to do when no split is present.
        if bool((factor == 1.0).all()):
            out[ticker] = frame
            continue
        adjusted = frame.copy()
        for col in ("open", "high", "low", "close", "dividend"):
            if col in adjusted.columns:
                adjusted[col] = adjusted[col].astype(float) / factor
        if "volume" in adjusted.columns:
            adjusted["volume"] = adjusted["volume"].astype(float) * factor
        out[ticker] = adjusted
    return out


def warn_unadjustable_fmp_frames(
    bars_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Warn (once per call) about FMP-served frames that cannot be split-adjusted.

    FMP frames carry ``adj_close`` but no ``Stock Splits`` column, so they have
    no ``split_factor``. We deliberately do **not** reconstruct one from the
    ``adj_close``/``close`` ratio: ``adj_close`` is back-adjusted for *both*
    splits and dividends, so the ratio cannot separate the two — a pure dividend
    would be mis-read as a split and bake dividend return into the price series
    (corrupting the splits_only regime and partially double-counting dividends).

    Instead these frames pass through unadjusted and we emit a clear warning so
    the limitation is visible rather than silently producing inconsistent
    results. yfinance-served tickers (the default path) carry ``split_factor``
    and are still adjusted normally.
    """
    unadjusted = [
        ticker
        for ticker, frame in bars_dict.items()
        if frame is not None and not frame.empty and "split_factor" not in frame.columns
    ]
    if unadjusted:
        from screener.logging_config import get_logger

        get_logger(__name__).warning(
            "fmp_unadjusted_in_splits_only",
            reason=(
                "FMP frames lack a Stock Splits column; splits cannot be "
                "reliably recovered from adj_close (splits+dividends are "
                "conflated), so these tickers are left split-unadjusted"
            ),
            tickers=unadjusted[:20],
            count=len(unadjusted),
        )
    return bars_dict


def _merge_cached(
    existing: Optional[pd.DataFrame], new: pd.DataFrame, interval: str = "1d"
) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], axis=0)
    if merged.empty:
        return merged
    merged.index = _naive_normalized_index(merged.index, interval)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def _inclusive_fetch_bounds(
    start: date, end: date, interval: str = "1d"
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if interval != "1d" and end_ts == end_ts.normalize():
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return start_ts, end_ts


def _has_range(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str = "1d",
) -> bool:
    if df is None or df.empty:
        return False
    in_range = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return (
        not in_range.empty
        and in_range.index.min() <= start_ts + pd.Timedelta(days=3)
        and in_range.index.max() >= end_ts - pd.Timedelta(days=3)
    )


def _needs_tail_refresh(cache_path: Path, end_ts: pd.Timestamp) -> bool:
    """Return whether a near-present cache is old enough for a tail refresh."""
    today = date.today()
    if abs((end_ts.date() - today).days) > 2:
        return False
    try:
        ttl_seconds = float(
            os.environ.get("SCREENER_PRICE_TAIL_TTL_SECONDS", PRICE_TAIL_TTL_SECONDS)
        )
    except ValueError:
        ttl_seconds = PRICE_TAIL_TTL_SECONDS
    try:
        return time.time() - cache_path.stat().st_mtime > max(0.0, ttl_seconds)
    except OSError:
        return False


def _split_download(
    raw: pd.DataFrame, tickers: list[str], interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {ticker: _empty_ohlcv_frame() for ticker in tickers}
    if not isinstance(raw.columns, pd.MultiIndex):
        ticker = tickers[0] if tickers else ""
        return {ticker: _normalize_frame(raw, interval)}

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
        frames[ticker] = _normalize_frame(frame, interval)
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
        interval: str = "1d",
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.batch_size = max(1, int(batch_size))
        self.refresh = bool(refresh)
        self.max_workers = max(1, int(max_workers))
        self.interval = str(interval)

    def _cache_key(self, ticker: str) -> str:
        # Intraday intervals get their own cache namespace so the existing daily
        # parquet files stay valid and are never polluted with 15m/1h bars.
        base = ticker if self.interval == "1d" else f"{ticker}__{self.interval}"
        return base if self.auto_adjust else f"{base}__raw"

    # Approximate yfinance intraday history caps, in calendar days.
    _INTRADAY_CAP_DAYS = {
        "1m": 30,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
    }

    _INTRADAY_REQUEST_SPAN_DAYS = {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
    }

    def _intraday_chunks(
        self, fetch_start: pd.Timestamp, fetch_end: pd.Timestamp
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Clamp an intraday window to availability and split request spans."""
        cap = self._INTRADAY_CAP_DAYS.get(self.interval)
        span_days = self._INTRADAY_REQUEST_SPAN_DAYS.get(self.interval)
        if cap is None or span_days is None:
            return [(fetch_start, fetch_end)]
        cap_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=cap)
        if fetch_start < cap_start:
            from screener.logging_config import get_logger

            get_logger(__name__).warning(
                "yfinance_intraday_history_cap",
                interval=self.interval,
                requested_start=str(fetch_start.date()),
                clamped_start=str(cap_start.date()),
                reason="requested start predates yfinance intraday availability",
            )
            fetch_start = cap_start
        if fetch_start > fetch_end:
            return []

        chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        chunk_start = fetch_start
        while chunk_start <= fetch_end:
            chunk_end = min(
                fetch_end,
                chunk_start.normalize()
                + pd.Timedelta(days=span_days)
                - pd.Timedelta(1, "ns"),
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end.normalize() + pd.Timedelta(days=1)
        return chunks

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tickers = [t for t in tickers if t]
        results: dict[str, pd.DataFrame] = {}
        start_ts, end_ts = _inclusive_fetch_bounds(start, end, self.interval)
        cached_by_ticker: dict[str, pd.DataFrame] = {}
        missing: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
        tail_refresh_tickers: set[str] = set()

        for ticker in tickers:
            cache_key = self._cache_key(ticker)
            cached = (
                None
                if self.refresh
                else _load_cached(cache_key, self.cache_dir, self.interval)
            )
            if cached is not None and not cached.empty:
                cached_by_ticker[ticker] = cached
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts, self.interval)
            ):
                if _needs_tail_refresh(_cache_path(cache_key, self.cache_dir), end_ts):
                    tail_start = max(cached.index) - pd.Timedelta(days=7)
                    missing.setdefault((tail_start, end_ts), []).append(ticker)
                    tail_refresh_tickers.add(ticker)
                else:
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
            windows = [(fetch_start, fetch_end)]
            if self.interval != "1d":
                windows = self._intraday_chunks(fetch_start, fetch_end)
            for window_start, window_end in windows:
                for i in range(0, len(group), self.batch_size):
                    jobs.append(
                        (window_start, window_end, group[i : i + self.batch_size])
                    )

        def download_job(
            job: tuple[pd.Timestamp, pd.Timestamp, list[str]],
        ) -> tuple[list[str], pd.DataFrame]:
            fetch_start, fetch_end, batch = job
            download_end = fetch_end + pd.Timedelta(days=1)
            if self.interval != "1d":
                download_end = fetch_end.normalize() + pd.Timedelta(days=1)
            download_kwargs = dict(
                start=fetch_start,
                # yfinance treats ``end`` as exclusive for both daily and
                # intraday; keep the +1 day so the last requested bar is included.
                end=download_end,
                interval=self.interval,
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
            if not jobs:
                downloads = []
            elif len(jobs) == 1:
                downloads = [download_job(jobs[0])]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(jobs))
                ) as pool:
                    downloads = list(pool.map(download_job, jobs))

        downloaded_by_ticker: dict[str, pd.DataFrame] = {}
        for batch, raw in downloads:
            downloaded = _split_download(raw, batch, self.interval)
            for ticker in batch:
                norm = downloaded.get(ticker, _empty_ohlcv_frame())
                downloaded_by_ticker[ticker] = _merge_cached(
                    downloaded_by_ticker.get(ticker), norm, self.interval
                )

        for ticker in dict.fromkeys(t for group in missing.values() for t in group):
            cache_key = self._cache_key(ticker)
            norm = downloaded_by_ticker.get(ticker, _empty_ohlcv_frame())
            merged = _merge_cached(cached_by_ticker.get(ticker), norm, self.interval)
            if not merged.empty and (
                not norm.empty or ticker not in tail_refresh_tickers
            ):
                _save_cache(cache_key, merged, self.cache_dir)
            results[ticker] = merged.loc[
                (merged.index >= start_ts) & (merged.index <= end_ts)
            ]
        return results


# FMP spells intraday intervals differently from yfinance; the daily endpoint
# is a separate URL entirely (``historical-price-full`` vs ``historical-chart``).
_FMP_INTRADAY_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
}


def _fmp_cache_key(ticker: str, auto_adjust: bool, interval: str = "1d") -> str:
    suffix = "" if auto_adjust else "__raw"
    base = ticker if interval == "1d" else f"{ticker}__{interval}"
    return f"fmp_{base}{suffix}"


def _normalize_fmp_historical(
    payload: object, auto_adjust: bool, interval: str = "1d"
) -> pd.DataFrame:
    rows: object
    if interval == "1d":
        if not isinstance(payload, dict):
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        rows = payload.get("historical")
    else:
        # The intraday historical-chart endpoint returns a bare list of bars
        # (newest first) instead of a dict with a "historical" key.
        rows = payload
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
    idx = cast(pd.DatetimeIndex, out.index)
    if interval == "1d":
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out.index = idx.normalize()
    else:
        # FMP intraday timestamps are US/Eastern wall-clock with no tz marker.
        # Convert to naive UTC so FMP bars land on the same simulation calendar
        # as yfinance intraday bars (see _naive_normalized_index).
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York")
        out.index = idx.tz_convert("UTC").tz_localize(None)

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
    """Fetch OHLCV from Financial Modeling Prep.

    Daily bars come from ``historical-price-full`` (dividend/split adjusted via
    ``adjClose`` when ``auto_adjust`` is set). Intraday bars come from
    ``historical-chart/{interval}`` and are always raw — FMP publishes no
    adjusted intraday prices, which is acceptable inside the short history
    windows intraday backtests use.

    The API key is read from ``FMP_API_KEY`` unless passed explicitly.
    """

    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full"
    intraday_base_url = "https://financialmodelingprep.com/api/v3/historical-chart"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Optional[Path] = None,
        auto_adjust: bool = True,
        refresh: bool = False,
        session: requests.Session | None = None,
        interval: str = "1d",
        max_workers: int = 8,
    ) -> None:
        if interval != "1d" and interval not in _FMP_INTRADAY_INTERVALS:
            raise ValueError(
                f"FMPPriceFetcher supports intervals '1d' and "
                f"{sorted(_FMP_INTRADAY_INTERVALS)}; got {interval!r}"
            )
        self.interval = interval
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required to use the FMP price fetcher")
        self.cache_dir = cache_dir or FMP_CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.refresh = bool(refresh)
        self.max_workers = max(1, int(max_workers))
        self.session = session or requests.Session()
        if hasattr(self.session, "mount"):
            adapter = HTTPAdapter(
                pool_connections=self.max_workers, pool_maxsize=self.max_workers
            )
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        start_ts, end_ts = _inclusive_fetch_bounds(start, end, self.interval)
        ticker_list = [t for t in dict.fromkeys(tickers) if t]
        if not ticker_list:
            return {}

        def fetch_ticker(ticker: str) -> tuple[str, pd.DataFrame]:
            cache_key = _fmp_cache_key(ticker, self.auto_adjust, self.interval)
            cached = (
                None
                if self.refresh
                else _load_cached(cache_key, self.cache_dir, self.interval)
            )
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts, self.interval)
            ):
                if not _needs_tail_refresh(
                    _cache_path(cache_key, self.cache_dir), end_ts
                ):
                    return ticker, cached.loc[
                        (cached.index >= start_ts) & (cached.index <= end_ts)
                    ]
                fetch_start = max(cached.index) - pd.Timedelta(days=7)
                is_tail_refresh = True
            else:
                fetch_start = start_ts
                is_tail_refresh = False

            if self.interval == "1d":
                url = f"{self.base_url}/{ticker}"
            else:
                fmp_interval = _FMP_INTRADAY_INTERVALS[self.interval]
                url = f"{self.intraday_base_url}/{fmp_interval}/{ticker}"

            def request_payload(url: str = url) -> object:
                response = self.session.get(
                    url,
                    params={
                        "from": fetch_start.date().isoformat(),
                        "to": end_ts.date().isoformat(),
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

            empty_payload: object = {}
            payload: object = call_with_resilience(
                "fmp",
                f"historical prices {ticker}",
                request_payload,
                fallback=empty_payload,
            )
            norm = _normalize_fmp_historical(payload, self.auto_adjust, self.interval)
            merged = _merge_cached(cached, norm, self.interval)
            if not merged.empty:
                if not norm.empty or not is_tail_refresh:
                    _save_cache(cache_key, merged, self.cache_dir)
                return ticker, merged.loc[
                    (merged.index >= start_ts) & (merged.index <= end_ts)
                ]
            return ticker, pd.DataFrame(columns=OHLCV_COLUMNS)

        if len(ticker_list) == 1:
            fetched = [fetch_ticker(ticker_list[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(ticker_list))
            ) as pool:
                fetched = list(pool.map(fetch_ticker, ticker_list))
        return dict(fetched)


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
    interval: str = "1d",
) -> PriceFetcher:
    _load_env_file()
    resolved = (provider or os.environ.get("SCREENER_PRICE_PROVIDER") or "auto").lower()
    if resolved in {"auto", "default"}:
        primary = YFinancePriceFetcher(
            auto_adjust=auto_adjust, refresh=refresh, interval=interval
        )
        if os.environ.get("FMP_API_KEY"):
            fallback = FMPPriceFetcher(
                auto_adjust=auto_adjust, refresh=refresh, interval=interval
            )
            return FallbackPriceFetcher(primary, fallback)
        return primary
    if resolved in {"yf", "yfinance"}:
        return YFinancePriceFetcher(
            auto_adjust=auto_adjust, refresh=refresh, interval=interval
        )
    if resolved in {"fmp", "financialmodelingprep"}:
        return FMPPriceFetcher(
            auto_adjust=auto_adjust, refresh=refresh, interval=interval
        )
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
    if isinstance(value, pd.Timestamp):  # pragma: no cover - Timestamp is a datetime
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Cannot convert {value!r} to date")
