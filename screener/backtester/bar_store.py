"""Interval-partitioned on-disk bar store.

Bars live under ``~/.screener/bars/{market}/{interval}/{symbol}.parquet`` —
one parquet file per (market, interval, symbol), replacing the per-interval
cache-key namespacing the legacy flat price cache used for intraday bars
(``AAPL__15m.parquet``). Writes reuse the atomic temp-file + ``os.replace``
discipline of :mod:`screener.backtester.price_cache`, and frames are
normalized on load by :func:`screener.backtester.price_frames.naive_normalized_index`,
so intraday bars keep canonical naive-UTC timestamps (daily stays
midnight-normalized).

The 1m interval is the canonical fine-grained archive: ``screener bars record``
appends its trailing window daily so the archive grows past the provider's
~30-day free-history cap, and coarser intraday requests (5m/15m/30m/1h) are
served by resampling the stored 1m series locally
(:func:`screener.backtester.price_frames.resample_intraday_bars`) instead of
separate per-interval downloads.

The raw (``auto_adjust=False``) regime is namespaced ``{symbol}__raw.parquet``
so adjusted and raw bars never mix in one file. Daily (``1d``) bars continue
to use the legacy flat cache in ``~/.screener/prices``; the store layout
already leaves room for a ``1d`` partition if that changes later.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Optional

import pandas as pd

from screener.backtester.price_frames import (
    OHLCV_COLUMNS,
    merge_price_frames,
    naive_normalized_index,
)


BARS_ROOT = Path.home() / ".screener" / "bars"


def bar_path(
    symbol: str,
    *,
    market: str,
    interval: str,
    raw: bool = False,
    root: Optional[Path] = None,
) -> Path:
    """Store path for one (market, interval, symbol) series."""
    safe = symbol.replace("/", "_").replace(":", "_")
    name = f"{safe}__raw.parquet" if raw else f"{safe}.parquet"
    return Path(root or BARS_ROOT) / market / interval / name


def load_bars(
    symbol: str,
    *,
    market: str,
    interval: str,
    raw: bool = False,
    root: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load one stored series, normalized to the canonical bar index."""
    path = bar_path(symbol, market=market, interval=interval, raw=raw, root=root)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        frame.index = naive_normalized_index(frame.index, interval)
        price_columns = [column for column in OHLCV_COLUMNS if column in frame.columns]
        return frame.dropna(subset=price_columns) if price_columns else frame
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def save_bars(
    symbol: str,
    frame: pd.DataFrame,
    *,
    market: str,
    interval: str,
    raw: bool = False,
    root: Optional[Path] = None,
) -> None:
    """Atomically replace one store file without exposing partial parquet."""
    destination = bar_path(symbol, market=market, interval=interval, raw=raw, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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


def append_bars(
    symbol: str,
    frame: pd.DataFrame,
    *,
    market: str,
    interval: str,
    raw: bool = False,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge ``frame`` into the stored series (dedupe, keep last) and persist.

    Returns the merged frame. Overlapping indexes are idempotent, so a daily
    recorder can re-append a trailing window without duplicating bars. No-op
    appends leave the store untouched (preserving the file mtime the tail-TTL
    freshness check relies on); revised bars (same timestamps, new values) are
    persisted because ``merge_price_frames`` keeps the last occurrence.
    """
    existing = load_bars(symbol, market=market, interval=interval, raw=raw, root=root)
    merged = merge_price_frames(existing, frame, interval)
    if merged.empty:
        return merged
    if existing is not None and merged.equals(existing):
        return merged
    save_bars(symbol, merged, market=market, interval=interval, raw=raw, root=root)
    return merged


__all__ = [
    "BARS_ROOT",
    "append_bars",
    "bar_path",
    "load_bars",
    "save_bars",
]
