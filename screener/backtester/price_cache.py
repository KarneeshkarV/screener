"""Atomic parquet cache operations for historical price frames."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from screener.backtester.price_frames import (
    OHLCV_COLUMNS,
    drop_index_freq,
    naive_normalized_index,
)

CACHE_DIR = Path.home() / ".screener" / "prices"
FMP_CACHE_DIR = Path.home() / ".screener" / "fmp_prices"
PRICE_TAIL_TTL_SECONDS = 60 * 60
EMPTY_HISTORY_TTL_SECONDS = 24 * 60 * 60


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}.parquet"


#: Ticks per calendar day for each pandas datetime64 resolution, used to ask
#: whether an index is already midnight-aligned without building a new one.
_TICKS_PER_DAY = {
    "s": 86_400,
    "ms": 86_400_000,
    "us": 86_400_000_000,
    "ns": 86_400_000_000_000,
}


def _canonical_index(index: pd.Index, interval: str) -> pd.Index:
    """``naive_normalized_index``, skipped when the index is already canonical.

    Every frame this module writes is stored canonical, so the conversion is a
    no-op on essentially every cache hit -- but it still allocated a new index
    per ticker, and a warm screen reads hundreds of them. The checks below are
    all metadata reads except the midnight test, which is one integer modulo
    over the existing buffer and allocates nothing.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        return naive_normalized_index(index, interval)
    if interval != "1d":
        return index
    index = drop_index_freq(index)
    ticks = _TICKS_PER_DAY.get(index.unit)
    if ticks is None:
        return index.normalize()
    values = index.asi8
    if values.size and (values % ticks).any():
        return index.normalize()
    return index


def _has_missing(frame: pd.DataFrame, columns: list[str]) -> bool:
    """Whether any of ``columns`` holds a null, without materialising a mask.

    ``dropna`` copies the whole frame whether or not it has anything to drop,
    and a cache entry almost never does: the write path drops NaN rows before
    saving, so this guard exists for frames written by an older version. Asking
    first costs one pass over a float buffer and keeps the copy for the rare
    frame that needs it.
    """
    for column in columns:
        values = frame[column].to_numpy(copy=False)
        if values.dtype.kind == "f":
            if np.isnan(values).any():
                return True
        elif values.dtype.kind not in "iub" and pd.isna(values).any():
            return True
    return False


def _frame_from_table(table: pa.Table) -> pd.DataFrame:
    """``table.to_pandas()`` for a cache entry, taking a shortcut where it fits.

    ``to_pandas`` is the slowest step of a warm read - slower than the parquet
    decode itself - because it is general: any index, any dtype, any column
    layout, then a block consolidation over the lot. A cache entry is always
    the same narrow shape, so the shortcut below rebuilds that one shape
    directly from the arrow columns and leaves every other shape to
    ``to_pandas``.

    The shortcut applies only when the pandas metadata describes a single
    datetime index column and one column level, and every other column is
    numeric. That is checked, not assumed: anything else - a range index, a
    string column, a MultiIndex - falls through, so the frame this returns is
    the frame ``to_pandas`` would have returned, including its column-index
    name.
    """
    metadata = (table.schema.metadata or {}).get(b"pandas")
    if metadata is None:
        return cast(pd.DataFrame, table.to_pandas())
    try:
        described = json.loads(metadata)
        index_columns = described["index_columns"]
        column_indexes = described.get("column_indexes") or [{}]
    except (ValueError, KeyError, TypeError):
        return cast(pd.DataFrame, table.to_pandas())
    if len(index_columns) != 1 or len(column_indexes) != 1:
        return cast(pd.DataFrame, table.to_pandas())
    index_name = index_columns[0]
    if not isinstance(index_name, str) or index_name not in table.column_names:
        return cast(pd.DataFrame, table.to_pandas())

    index: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}
    for field, column in zip(table.schema, table.columns):
        values = column.to_numpy(zero_copy_only=False)
        if field.name == index_name:
            index = values
        elif pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
            columns[field.name] = values
        else:
            return cast(pd.DataFrame, table.to_pandas())
    if index is None or index.dtype.kind != "M":
        return cast(pd.DataFrame, table.to_pandas())

    frame = pd.DataFrame(columns, copy=False)
    # An unnamed index is stored under pandas' own placeholder name, which
    # ``to_pandas`` strips back off on the way out.
    restored = None if index_name.startswith("__index_level_") else index_name
    frame.index = pd.DatetimeIndex(index, name=restored)
    frame.columns.name = column_indexes[0].get("name")
    return frame


def load_cached_frame(
    ticker: str, cache_dir: Path = CACHE_DIR, interval: str = "1d"
) -> pd.DataFrame | None:
    path = cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    try:
        # ``pq.read_table`` rather than ``pd.read_parquet``: the arrow read
        # releases the GIL, so the thread pool in ``data.py`` that reads these
        # entries actually scales, and it skips a layer of pandas dispatch. The
        # pandas metadata parquet carries still restores the index, so the
        # frame is identical to what ``pd.read_parquet`` returned.
        # ``use_threads=False`` because the parallelism is already one thread
        # per file: arrow's own pool then only adds contention, and a cache
        # entry is one small row group anyway.
        frame = _frame_from_table(pq.read_table(path, use_threads=False))
        frame.index = _canonical_index(frame.index, interval)
        price_columns = [column for column in OHLCV_COLUMNS if column in frame.columns]
        if price_columns and _has_missing(frame, price_columns):
            return frame.dropna(subset=price_columns)
        return frame
    except (OSError, pd.errors.ParserError, ValueError, pa.ArrowException):
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
