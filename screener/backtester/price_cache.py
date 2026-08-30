"""Atomic parquet cache operations for historical price frames."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd

from screener.backtester.price_frames import OHLCV_COLUMNS, naive_normalized_index

CACHE_DIR = Path.home() / ".screener" / "prices"
FMP_CACHE_DIR = Path.home() / ".screener" / "fmp_prices"
PRICE_TAIL_TTL_SECONDS = 60 * 60
EMPTY_HISTORY_TTL_SECONDS = 24 * 60 * 60


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}.parquet"


def load_cached_frame(
    ticker: str, cache_dir: Path = CACHE_DIR, interval: str = "1d"
) -> pd.DataFrame | None:
    path = cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        frame.index = naive_normalized_index(frame.index, interval)
        price_columns = [column for column in OHLCV_COLUMNS if column in frame.columns]
        return frame.dropna(subset=price_columns) if price_columns else frame
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def save_cached_frame(
    ticker: str, frame: pd.DataFrame, cache_dir: Path = CACHE_DIR
) -> None:
    """Atomically replace one cache entry without exposing partial parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_path(ticker, cache_dir)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=cache_dir
        )
        temporary = Path(temporary_name)
        os.close(descriptor)
        descriptor = -1
        frame.to_parquet(temporary)
        os.replace(temporary, destination)
    except (OSError, ValueError):
        return
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def empty_history_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Sidecar marker recording a window the vendor served no bars for."""
    return cache_path(ticker, cache_dir).with_suffix(".empty.json")


def record_empty_history(
    ticker: str,
    first: pd.Timestamp,
    last: pd.Timestamp,
    cache_dir: Path = CACHE_DIR,
) -> None:
    """Remember that ``[first, last]`` came back with no bars.

    A vendor that has no history for a range still has none on the next run,
    so re-asking is pure latency. It is also the largest repeated cost of a
    warm screen: an India field re-requests several hundred names every run
    that have nothing older to give -- names listed after the window opens, and
    names the vendor does not carry at all. The marker lets the next run skip
    exactly those requests and nothing else.
    """
    destination = empty_history_path(ticker, cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"first": first.isoformat(), "last": last.isoformat()})
        )
    except (OSError, ValueError):
        return


def clear_empty_history(ticker: str, cache_dir: Path = CACHE_DIR) -> None:
    """Drop the marker once the vendor does serve bars for the ticker."""
    try:
        empty_history_path(ticker, cache_dir).unlink(missing_ok=True)
    except OSError:
        return


def has_empty_history(
    ticker: str,
    first: pd.Timestamp,
    last: pd.Timestamp,
    cache_dir: Path = CACHE_DIR,
) -> bool:
    """Whether a recent request for a *superset* of this window returned nothing.

    Both bounds must be covered: a marker only licenses skipping a request it
    already answered. The TTL bounds how long a newly listed symbol, or a
    vendor backfill, can stay hidden -- and ``--refresh`` ignores markers
    entirely.
    """
    path = empty_history_path(ticker, cache_dir)
    try:
        ttl_seconds = float(
            os.environ.get(
                "SCREENER_EMPTY_HISTORY_TTL_SECONDS", EMPTY_HISTORY_TTL_SECONDS
            )
        )
    except ValueError:
        ttl_seconds = EMPTY_HISTORY_TTL_SECONDS
    try:
        if time.time() - path.stat().st_mtime > max(0.0, ttl_seconds):
            return False
        marker = json.loads(path.read_text())
        recorded_first = pd.Timestamp(marker["first"])
        recorded_last = pd.Timestamp(marker["last"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return recorded_first <= first and recorded_last >= last


def needs_tail_refresh(path: Path, end: pd.Timestamp) -> bool:
    """Return whether a near-present cache is old enough for a tail refresh."""
    if abs((end.date() - date.today()).days) > 2:
        return False
    try:
        ttl_seconds = float(
            os.environ.get("SCREENER_PRICE_TAIL_TTL_SECONDS", PRICE_TAIL_TTL_SECONDS)
        )
    except ValueError:
        ttl_seconds = PRICE_TAIL_TTL_SECONDS
    try:
        return time.time() - path.stat().st_mtime > max(0.0, ttl_seconds)
    except OSError:
        return False


__all__ = [
    "CACHE_DIR",
    "EMPTY_HISTORY_TTL_SECONDS",
    "FMP_CACHE_DIR",
    "PRICE_TAIL_TTL_SECONDS",
    "cache_path",
    "clear_empty_history",
    "empty_history_path",
    "has_empty_history",
    "load_cached_frame",
    "needs_tail_refresh",
    "record_empty_history",
    "save_cached_frame",
]
