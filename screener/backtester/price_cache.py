"""Atomic parquet cache operations for historical price frames."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import tempfile
import time
from typing import Optional

import pandas as pd

from screener.backtester.price_frames import OHLCV_COLUMNS, naive_normalized_index


CACHE_DIR = Path.home() / ".screener" / "prices"
FMP_CACHE_DIR = Path.home() / ".screener" / "fmp_prices"
PRICE_TAIL_TTL_SECONDS = 60 * 60


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}.parquet"


def load_cached_frame(
    ticker: str, cache_dir: Path = CACHE_DIR, interval: str = "1d"
) -> Optional[pd.DataFrame]:
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
    "FMP_CACHE_DIR",
    "PRICE_TAIL_TTL_SECONDS",
    "cache_path",
    "load_cached_frame",
    "needs_tail_refresh",
    "save_cached_frame",
]
