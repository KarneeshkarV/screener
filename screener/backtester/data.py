"""Historical OHLCV fetching adapter.

Defines the ``PriceFetcher`` protocol used by the engine, a default
``YFinancePriceFetcher`` with an on-disk parquet cache, and a small symbol
mapper that translates TradingView-style tickers to yfinance tickers.

Tests inject a ``StubPriceFetcher`` that returns pre-built synthetic frames;
the engine never depends directly on yfinance.
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, TypeVar, cast

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from screener import _optional
from screener.backtester.sessions import drop_incomplete_sessions
from screener.backtester.price_cache import (
    CACHE_DIR,
    FMP_CACHE_DIR,
)
from screener.backtester.price_cache import (
    PRICE_TAIL_TTL_SECONDS as PRICE_TAIL_TTL_SECONDS,
)
from screener.backtester.price_cache import (
    cache_path as _cache_path,
)
from screener.backtester.price_cache import (
    clear_empty_history as _clear_empty_history,
)
from screener.backtester.price_cache import (
    has_empty_history as _has_empty_history,
)
from screener.backtester.price_cache import (
    load_cached_frame as _load_cached,
)
from screener.backtester.price_cache import (
    needs_tail_refresh as _needs_tail_refresh,
)
from screener.backtester.price_cache import (
    record_empty_history as _record_empty_history,
)
from screener.backtester.price_cache import (
    save_cached_frame as _save_cache,
)
from screener.backtester.price_frames import (
    CORPORATE_ACTION_COLUMNS as CORPORATE_ACTION_COLUMNS,
)
from screener.backtester.price_frames import (
    OHLCV_COLUMNS,
    naive_normalized_index,
    normalize_price_frame,
)
from screener.backtester.price_frames import (
    apply_splits_only_adjustment as apply_splits_only_adjustment,
)
from screener.backtester.price_frames import (
    empty_ohlcv_frame as _empty_ohlcv_frame,
)
from screener.backtester.price_frames import (
    frame_has_range as _has_range,
    range_slice as _range_slice,
)
from screener.backtester.price_frames import (
    inclusive_fetch_bounds as _inclusive_fetch_bounds,
)
from screener.backtester.price_frames import (
    merge_price_frames as _merge_cached,
)
from screener.backtester.price_frames import (
    split_yfinance_download as _split_download,
)
from screener.backtester.price_frames import (
    warn_unadjustable_fmp_frames as warn_unadjustable_fmp_frames,
)

# Re-exported for backward compatibility: ``backtester.fundamentals`` and the
# docs import ``load_env_file`` from here. The definition now lives in the
# neutral ``screener.config`` so the FMP transport need not import the
# backtester to resolve a credential.
from screener.config import load_env_file as load_env_file
from screener.fmp import FMP_V3_BASE_URL, FmpClient, resolve_api_key
from screener.logging_config import suppressed_yfinance_errors
from screener.providers import StaleDataError
from screener.resilience import call_with_resilience

# Re-exported for backward compatibility: several modules and the docs import
# ``tv_to_yf`` from ``screener.backtester.data``. The definition now lives in
# ``screener.symbols``.
from screener.symbols import tv_to_yf as tv_to_yf

_naive_normalized_index = naive_normalized_index
_normalize_frame = normalize_price_frame
_YFINANCE_CONFIGURED = False
T = TypeVar("T")

# Passed directly to ``yf.download`` and enforced around Ticker-based fetch
# closures with ``call_yfinance_with_timeout``. The daemon worker deliberately
# is not joined after expiry, so a stuck provider cannot hold up the caller.
YFINANCE_TIMEOUT_SECONDS = 5


def call_yfinance_with_timeout(func: Callable[[], T]) -> T:
    """Run a Ticker-based yfinance fetch with the same five-second deadline."""
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            results.put((True, func()))
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(target=run, daemon=True, name="yfinance-fetch").start()
    try:
        succeeded, value = results.get(timeout=YFINANCE_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError(
            f"yfinance request exceeded {YFINANCE_TIMEOUT_SECONDS}s"
        ) from exc
    if succeeded:
        return cast(T, value)
    raise cast(BaseException, value)


def _configure_yfinance() -> None:
    """Configure yfinance through its supported timezone-cache API."""
    global _YFINANCE_CONFIGURED
    if _YFINANCE_CONFIGURED:
        return
    try:
        yf = _optional.load("yfinance")

        if os.path.isdir("/dev/shm"):
            yf.set_tz_cache_location("/dev/shm/screener-yftz")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any swap failure
        from screener.logging_config import get_logger

        get_logger(__name__).warning(
            "yfinance_configure_failed",
            error=repr(exc),
        )
    finally:
        _YFINANCE_CONFIGURED = True


class PriceFetcher(Protocol):
    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        """Return dict of yf-style ticker → OHLCV DataFrame indexed by date.

        Frames must have lowercase columns: open, high, low, close, volume.
        ``adj_close`` is optional; absent means ``close`` is already adjusted.
        """


def _strict_refresh_stale_data_error(
    leftover: Sequence[tuple[str, pd.DataFrame]],
    *,
    as_of: pd.Timestamp,
) -> StaleDataError:
    """StaleDataError for a strict refresh that would have scored leftover cache.

    The message starts with ``strict refresh could not refresh bars for`` so a
    log line greps back to this constructor. Each ticker is named with the
    date of its newest cached bar and the calendar-day gap to ``as_of``.
    """
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tz is not None:
        as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
    clauses: list[str] = []
    for ticker, cached in leftover:
        last_bar = pd.Timestamp(cached.index.max())
        if last_bar.tz is not None:
            last_bar = last_bar.tz_convert("UTC").tz_localize(None)
        age_days = int((as_of_ts.normalize() - last_bar.normalize()).days)
        clauses.append(
            f"{ticker} (newest bar {last_bar.date()}, {age_days} calendar days old)"
        )
    return StaleDataError(
        "strict refresh could not refresh bars for " + ", ".join(clauses)
    )


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

    ``refresh=True`` forces a download of the requested window even when a
    covering parquet already exists. A failed download is an empty frame,
    and that empty frame is merged with leftover cache so the caller still
    has data. ``strict=True`` changes only that failure path, and only
    together with ``refresh``: leftover cache is refused with
    :class:`~screener.providers.StaleDataError` instead of being returned
    to rank on. ``strict`` without ``refresh`` is a no-op here; that flag
    governs the TradingView snapshot, not these bars.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        auto_adjust: bool = True,
        batch_size: int = 75,
        refresh: bool = False,
        max_workers: int = 4,
        interval: str = "1d",
        strict: bool = False,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.batch_size = max(1, int(batch_size))
        self.refresh = bool(refresh)
        self.max_workers = max(1, int(max_workers))
        self.interval = str(interval)
        self.strict = bool(strict)

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

    CACHE_READ_WORKERS = 8

    def _load_cache_entries(
        self, cache_keys: dict[str, str]
    ) -> dict[str, pd.DataFrame | None]:
        """Every ticker's cached frame, read concurrently."""

        def read(cache_key: str) -> pd.DataFrame | None:
            return _load_cached(cache_key, self.cache_dir, self.interval)

        if len(cache_keys) < 2:
            return {ticker: read(key) for ticker, key in cache_keys.items()}
        with ThreadPoolExecutor(
            max_workers=min(self.CACHE_READ_WORKERS, len(cache_keys))
        ) as pool:
            frames = list(pool.map(read, cache_keys.values()))
        return dict(zip(cache_keys, frames))

    def _complete(self, ticker: str, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """``frame`` without the bar of a session that is still trading.

        A daily bar the vendor serves for the open session is a snapshot of the
        tape, not a summary of the day: it changes on every request, and
        storing it puts a value in the cache that tomorrow contradicts. It is
        dropped both on the way in and on the way out, so a run also stops
        serving one an earlier build already wrote. ``None`` (no cache entry)
        passes through.
        """
        return drop_incomplete_sessions(frame, ticker, interval=self.interval)

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tickers = [t for t in tickers if t]
        results: dict[str, pd.DataFrame] = {}
        start_ts, end_ts = _inclusive_fetch_bounds(start, end, self.interval)
        cached_by_ticker: dict[str, pd.DataFrame] = {}
        # Download windows grouped by *why* the ticker needs one. Each branch
        # below pins one bound and derives the other from that ticker's own
        # cache, so keying the group on the exact window put nearly every
        # ticker in a group of its own: an India screen wanted 409 names in 171
        # separate downloads. yfinance charges per request, not per bar, so
        # that dominated the run. Grouping by cause and downloading the union
        # of each group's windows keeps every ticker's bars a superset of what
        # it asked for -- the save-time merge keeps bars outside the window and
        # the result is sliced back to the caller's range -- while collapsing
        # the request count to one batch chain per cause. Causes stay separate
        # so a name that needs the whole history never widens the cheap
        # seven-day tail refresh of a name that is already current.
        missing: dict[str, list[str]] = {}
        window_by_cause: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        # The window each ticker actually asked for, kept alongside the union
        # window its cause group downloads. The download uses the union; the
        # empty-history marker uses this, because the marker is a claim about
        # one ticker and the union is a claim about the group. Recording the
        # union would let a name missing only today's bar inherit a six-month
        # emptiness claim from a group peer.
        window_by_ticker: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        tail_refresh_tickers: set[str] = set()

        def want(
            cause: str, ticker: str, first: pd.Timestamp, last: pd.Timestamp
        ) -> None:
            missing.setdefault(cause, []).append(ticker)
            window_by_ticker[ticker] = (first, last)
            known = window_by_cause.get(cause)
            window_by_cause[cause] = (
                (first, last)
                if known is None
                else (min(known[0], first), max(known[1], last))
            )

        # Load the stored frames even on a refresh. The save-time merge keeps
        # bars outside the re-downloaded window. ``cached`` stays None while
        # refreshing, so every download decision below stays unchanged. The
        # merge keeps the LAST duplicate, so the fresh download still wins on
        # the dates it covers.
        #
        # Reading them up front, in parallel, rather than one at a time inside
        # the decision loop: parquet reads release the GIL, and a warm India
        # field is ~1900 of them. Once the download count came down they cost
        # more than the network did.
        cache_keys = {ticker: self._cache_key(ticker) for ticker in tickers}
        stored_by_ticker = {
            ticker: self._complete(ticker, frame)
            for ticker, frame in self._load_cache_entries(cache_keys).items()
        }

        for ticker in tickers:
            cache_key = cache_keys[ticker]
            stored = stored_by_ticker[ticker]
            if stored is not None and not stored.empty:
                cached_by_ticker[ticker] = stored
            cached = None if self.refresh else stored
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts, self.interval)
            ):
                if _needs_tail_refresh(_cache_path(cache_key, self.cache_dir), end_ts):
                    tail_start = cached.index.max() - pd.Timedelta(days=7)
                    want("tail", ticker, tail_start, end_ts)
                    tail_refresh_tickers.add(ticker)
                else:
                    results[ticker] = _range_slice(cached, start_ts, end_ts)
                continue

            fetch_start, fetch_end = start_ts, end_ts
            cause = "full"
            if not self.refresh and cached is not None and not cached.empty:
                min_cached = cached.index.min()
                max_cached = cached.index.max()
                if min_cached <= start_ts + pd.Timedelta(
                    days=3
                ) and max_cached < end_ts - pd.Timedelta(days=3):
                    fetch_start = max_cached + pd.Timedelta(days=1)
                    cause = "extend"
                elif max_cached >= end_ts - pd.Timedelta(
                    days=3
                ) and min_cached > start_ts + pd.Timedelta(days=3):
                    fetch_end = min_cached - pd.Timedelta(days=1)
                    cause = "backfill"
            if not self.refresh and _has_empty_history(
                cache_key, fetch_start, fetch_end, self.cache_dir
            ):
                # A recent request for a superset of this window served no
                # bars. Asking again cannot change that, and it is what makes a
                # warm screen spend most of its time on the network.
                results[ticker] = (
                    _range_slice(cached, start_ts, end_ts)
                    if cached is not None and not cached.empty
                    else _empty_ohlcv_frame()
                )
                continue
            want(cause, ticker, fetch_start, fetch_end)

        if not missing:
            return results

        _configure_yfinance()
        yf = _optional.load("yfinance")

        jobs: list[tuple[pd.Timestamp, pd.Timestamp, list[str]]] = []
        for cause, group in missing.items():
            fetch_start, fetch_end = window_by_cause[cause]
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
        ) -> tuple[list[str], pd.DataFrame, bool]:
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
                timeout=YFINANCE_TIMEOUT_SECONDS,
            )
            if not self.auto_adjust:
                download_kwargs["actions"] = True
            target = " ".join(batch) if len(batch) > 1 else batch[0]
            # The fallback is indistinguishable from a successful empty
            # download, so the success flag rides along with it: an outage,
            # a 429 or an open circuit must never be recorded as "the vendor
            # has no history for these names".
            ok, raw = call_with_resilience(
                "yfinance",
                f"download {len(batch)} ticker(s)",
                lambda: (True, yf.download(target, **download_kwargs)),
                fallback=(False, pd.DataFrame()),
            )
            return batch, raw, ok

        # yfinance reports expected "possibly delisted" messages for empty
        # pre-listing ranges. The empty frame is enough for
        # FallbackPriceFetcher to call FMP, so keep the lab/CLI output focused
        # on actionable diagnostics. Those messages go through the ``yfinance``
        # logger, which only ``suppressed_yfinance_errors`` can reach; it
        # reports a count on the way out so a real outage still surfaces. The
        # stderr redirect stays for anything the library prints directly.
        # Both are process-wide, covering the worker threads: per-batch scoping
        # would race when batches download concurrently.
        with suppressed_yfinance_errors(), contextlib.redirect_stderr(io.StringIO()):
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
        # A ticker's history is only known-empty when every request covering it
        # came back from the vendor. One failed chunk leaves it unknown.
        download_ok: dict[str, bool] = {}
        for batch, raw, ok in downloads:
            downloaded = _split_download(raw, batch, self.interval)
            for ticker in batch:
                norm = downloaded.get(ticker, _empty_ohlcv_frame())
                downloaded_by_ticker[ticker] = _merge_cached(
                    downloaded_by_ticker.get(ticker), norm, self.interval
                )
                download_ok[ticker] = download_ok.get(ticker, True) and ok
        leftover: list[tuple[str, pd.DataFrame]] = []
        for ticker in dict.fromkeys(t for group in missing.values() for t in group):
            cache_key = self._cache_key(ticker)
            norm = downloaded_by_ticker.get(ticker, _empty_ohlcv_frame())
            # Tail refreshes are excluded: an empty tail is the ordinary
            # weekend answer, not a statement about the ticker's history.
            if ticker not in tail_refresh_tickers:
                if not norm.empty:
                    _clear_empty_history(cache_key, self.cache_dir)
                elif download_ok.get(ticker, False):
                    _record_empty_history(
                        cache_key, *window_by_ticker[ticker], self.cache_dir
                    )
            stored = cached_by_ticker.get(ticker)
            # After the bookkeeping above, which is a claim about what the
            # vendor served and must see the response as it came. From here on
            # the open session's snapshot is not a bar.
            norm = drop_incomplete_sessions(norm, ticker, interval=self.interval)
            merged = _merge_cached(stored, norm, self.interval)
            # A failed download is an empty frame, and the merge then returns
            # leftover cache. That is the availability-first default. ``strict``
            # plus ``refresh`` is the live engine's fresh-or-error contract:
            # refuse to rank on those leftover bars. ``strict`` without
            # ``refresh`` does not belong here; it governs the scan snapshot.
            if (
                self.strict
                and self.refresh
                and norm.empty
                and stored is not None
                and not stored.empty
            ):
                leftover.append((ticker, stored))
            if not merged.empty and (
                not norm.empty or ticker not in tail_refresh_tickers
            ):
                _save_cache(cache_key, merged, self.cache_dir)
            results[ticker] = _range_slice(merged, start_ts, end_ts)
        if leftover:
            raise _strict_refresh_stale_data_error(leftover, as_of=end_ts)
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

    HTTP goes through the shared :class:`screener.fmp.FmpClient` transport, so
    base URL, header policy, API-key resolution, query shape (``apikey`` last)
    and JSON decoding are the same ones every other FMP call site uses. The
    connection-pooling ``requests`` session is handed to the client rather than
    driven directly, which is why the request timeout is declared here as
    :data:`PRICE_TIMEOUT_SECONDS` instead of inline.

    The API key is resolved by :func:`screener.fmp.resolve_api_key` (env, then
    the project ``.env``) unless passed explicitly.

    ``strict`` plus ``refresh`` refuses leftover cache after a failed
    download, matching :class:`YFinancePriceFetcher`. ``strict`` without
    ``refresh`` is a no-op here.
    """

    # FMP price history responses are much larger than the metric endpoints, so
    # this path keeps its own longer deadline instead of ``fmp.DEFAULT_TIMEOUT``.
    PRICE_TIMEOUT_SECONDS = 30.0

    daily_path = "historical-price-full"
    intraday_path = "historical-chart"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        auto_adjust: bool = True,
        refresh: bool = False,
        session: requests.Session | None = None,
        interval: str = "1d",
        max_workers: int = 8,
        strict: bool = False,
    ) -> None:
        if interval != "1d" and interval not in _FMP_INTRADAY_INTERVALS:
            raise ValueError(
                f"FMPPriceFetcher supports intervals '1d' and "
                f"{sorted(_FMP_INTRADAY_INTERVALS)}; got {interval!r}"
            )
        self.interval = interval
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required to use the FMP price fetcher")
        self.cache_dir = cache_dir or FMP_CACHE_DIR
        self.auto_adjust = bool(auto_adjust)
        self.refresh = bool(refresh)
        self.strict = bool(strict)
        self.max_workers = max(1, int(max_workers))
        self.session = session or requests.Session()
        if hasattr(self.session, "mount"):
            adapter = HTTPAdapter(
                pool_connections=self.max_workers, pool_maxsize=self.max_workers
            )
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        self.client = FmpClient(
            self.api_key,
            base_url=FMP_V3_BASE_URL,
            timeout=self.PRICE_TIMEOUT_SECONDS,
            session=self.session,
        )

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        start_ts, end_ts = _inclusive_fetch_bounds(start, end, self.interval)
        ticker_list = [t for t in dict.fromkeys(tickers) if t]
        if not ticker_list:
            return {}

        def fetch_ticker(
            ticker: str,
        ) -> tuple[str, pd.DataFrame, pd.DataFrame | None]:
            cache_key = _fmp_cache_key(ticker, self.auto_adjust, self.interval)
            # Same contract as YFinancePriceFetcher.fetch. The stored frame
            # loads even on a refresh and goes into the save-time merge, so a
            # forced re-download never discards bars outside its window.
            stored = drop_incomplete_sessions(
                _load_cached(cache_key, self.cache_dir, self.interval),
                ticker,
                interval=self.interval,
            )
            cached = None if self.refresh else stored
            if (
                not self.refresh
                and cached is not None
                and _has_range(cached, start_ts, end_ts, self.interval)
            ):
                if not _needs_tail_refresh(
                    _cache_path(cache_key, self.cache_dir), end_ts
                ):
                    return ticker, _range_slice(cached, start_ts, end_ts), None
                fetch_start = cached.index.max() - pd.Timedelta(days=7)
                is_tail_refresh = True
            else:
                fetch_start = start_ts
                is_tail_refresh = False
                if not self.refresh and _has_empty_history(
                    cache_key, start_ts, end_ts, self.cache_dir
                ):
                    # Same contract as the yfinance fetcher: a recent request
                    # for this window served no bars, so skip the round trip.
                    # This leg runs one request per ticker, so the names FMP
                    # does not carry cost the most.
                    in_range = (
                        _range_slice(cached, start_ts, end_ts)
                        if cached is not None and not cached.empty
                        else pd.DataFrame(columns=OHLCV_COLUMNS)
                    )
                    return ticker, in_range, None

            if self.interval == "1d":
                path = f"{self.daily_path}/{ticker}"
            else:
                fmp_interval = _FMP_INTRADAY_INTERVALS[self.interval]
                path = f"{self.intraday_path}/{fmp_interval}/{ticker}"

            def request_payload(path: str = path) -> object:
                return self.client.get(
                    path,
                    {
                        "from": fetch_start.date().isoformat(),
                        "to": end_ts.date().isoformat(),
                    },
                )

            empty_payload: tuple[bool, object] = (False, {})
            # Same contract as the yfinance leg: the resilience fallback is an
            # empty payload, so the success flag rides with it and a failed
            # request is never recorded as "the vendor has no history".
            ok, payload = call_with_resilience(
                "fmp",
                f"historical prices {ticker}",
                lambda: (True, request_payload()),
                fallback=empty_payload,
            )
            norm = _normalize_fmp_historical(payload, self.auto_adjust, self.interval)
            if not is_tail_refresh:
                if not norm.empty:
                    _clear_empty_history(cache_key, self.cache_dir)
                elif ok:
                    _record_empty_history(
                        cache_key, fetch_start, end_ts, self.cache_dir
                    )
            # Same rule as the yfinance leg, and for the same reason: after
            # the empty-history bookkeeping, which reads the response, and
            # before anything stores or serves it.
            norm = drop_incomplete_sessions(norm, ticker, interval=self.interval)
            leftover_cache = (
                stored
                if (
                    self.strict
                    and self.refresh
                    and norm.empty
                    and stored is not None
                    and not stored.empty
                )
                else None
            )
            merged = _merge_cached(stored, norm, self.interval)
            if not merged.empty:
                if not norm.empty or not is_tail_refresh:
                    _save_cache(cache_key, merged, self.cache_dir)
                return (
                    ticker,
                    _range_slice(merged, start_ts, end_ts),
                    leftover_cache,
                )
            return ticker, pd.DataFrame(columns=OHLCV_COLUMNS), leftover_cache

        if len(ticker_list) == 1:
            fetched = [fetch_ticker(ticker_list[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(ticker_list))
            ) as pool:
                fetched = list(pool.map(fetch_ticker, ticker_list))
        leftover = [
            (ticker, cache) for ticker, _frame, cache in fetched if cache is not None
        ]
        if leftover:
            raise _strict_refresh_stale_data_error(leftover, as_of=end_ts)
        return {ticker: frame for ticker, frame, _cache in fetched}


class ExchangeFallbackPriceFetcher:
    """Retry Indian symbols on BSE when NSE returns nothing.

    Some Indian tickers are not listed on NSE at all -- BSE B-group names in
    particular -- so the ``.NS`` request comes back empty even though the
    company trades. Retry just those on ``.BO``.

    Results are keyed back to the requested ``.NS`` symbol so callers keep the
    symbol identity they asked for; only the wire request changes exchange.
    Symbols the caller already pinned to ``.BO`` are left alone, and the
    ``.NS`` test makes this a no-op for US universes.
    """

    def __init__(self, inner: PriceFetcher) -> None:
        self.inner = inner

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        ticker_list = [ticker for ticker in tickers if ticker]
        results = self.inner.fetch(ticker_list, start, end)

        retry = {
            ticker: f"{ticker[: -len('.NS')]}.BO"
            for ticker in dict.fromkeys(ticker_list)
            if ticker.endswith(".NS")
            and (results.get(ticker) is None or results[ticker].empty)
        }
        if not retry:
            return results

        bse_results = self.inner.fetch(retry.values(), start, end)
        merged = dict(results)
        for ns_symbol, bo_symbol in retry.items():
            frame = bse_results.get(bo_symbol)
            if frame is not None and not frame.empty:
                merged[ns_symbol] = frame
        return merged


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
    strict: bool = False,
) -> PriceFetcher:
    load_env_file()
    resolved = (provider or os.environ.get("SCREENER_PRICE_PROVIDER") or "auto").lower()
    # The BSE retry wraps the outermost fetcher so a symbol is only re-requested
    # on ``.BO`` once every configured provider has come back empty for ``.NS``.
    # ``strict`` is the screen's fresh-or-error flag. It is a no-op unless
    # ``refresh`` is also True: a failed download then raises StaleDataError
    # instead of merging leftover parquet cache into the ranking. Default
    # False keeps every backtest and CLI caller on the availability-first path.
    if resolved in {"auto", "default"}:
        primary = YFinancePriceFetcher(
            auto_adjust=auto_adjust,
            refresh=refresh,
            interval=interval,
            strict=strict,
        )
        if os.environ.get("FMP_API_KEY"):
            fallback = FMPPriceFetcher(
                auto_adjust=auto_adjust,
                refresh=refresh,
                interval=interval,
                strict=strict,
            )
            return ExchangeFallbackPriceFetcher(FallbackPriceFetcher(primary, fallback))
        return ExchangeFallbackPriceFetcher(primary)
    if resolved in {"yf", "yfinance"}:
        return ExchangeFallbackPriceFetcher(
            YFinancePriceFetcher(
                auto_adjust=auto_adjust,
                refresh=refresh,
                interval=interval,
                strict=strict,
            )
        )
    if resolved in {"fmp", "financialmodelingprep"}:
        return ExchangeFallbackPriceFetcher(
            FMPPriceFetcher(
                auto_adjust=auto_adjust,
                refresh=refresh,
                interval=interval,
                strict=strict,
            )
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
