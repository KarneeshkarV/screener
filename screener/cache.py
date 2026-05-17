"""Small on-disk cache helpers for external market-data providers."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import pandas as pd


CACHE_ROOT = Path.home() / ".screener" / "cache"
T = TypeVar("T")


def stable_key(*parts: Any) -> str:
    """Return a deterministic key for JSON-serializable or repr-able parts."""
    payload = json.dumps(parts, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_ttl(
    value: str | int | float | None, *, default: float | None = None
) -> float | None:
    """Parse a TTL value in seconds, or with s/m/h/d suffixes."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().lower()
    if raw in {"", "none", "off", "disabled"}:
        return None
    suffix = raw[-1]
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(suffix)
    if multiplier is None:
        return float(raw)
    return float(raw[:-1]) * multiplier


def is_fresh(
    path: Path, ttl_seconds: float | None, *, now: float | None = None
) -> bool:
    if ttl_seconds is None or not path.exists():
        return False
    if ttl_seconds < 0:
        return True
    current = time.time() if now is None else now
    return current - path.stat().st_mtime <= ttl_seconds


def cache_path(namespace: str, key: str, suffix: str) -> Path:
    return CACHE_ROOT / namespace / f"{key}.{suffix.lstrip('.')}"


def read_json(path: Path, default: T | None = None) -> Any | T | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, default=str))
    tmp.replace(path)


def read_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)


PANEL_ROOT = Path.home() / ".screener" / "panels"


def panel_path(name: str) -> Path:
    """Path to an accumulating snapshot panel parquet (``~/.screener/panels``)."""
    return PANEL_ROOT / f"{name}.parquet"


def _normalize_dedupe_key_dates(
    existing: pd.DataFrame | None, rows: pd.DataFrame, keys: list[str]
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    out_existing = existing.copy() if existing is not None else None
    out_rows = rows.copy()
    for key in keys:
        key_l = key.lower()
        if not any(part in key_l for part in ("date", "as_of", "day")):
            continue
        series = out_rows.get(key)
        prior = out_existing.get(key) if out_existing is not None else None
        sample = pd.concat(
            [s.dropna().head(5) for s in (prior, series) if s is not None],
            ignore_index=True,
        )
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().all():
            if out_existing is not None and key in out_existing.columns:
                out_existing[key] = pd.to_datetime(
                    out_existing[key], errors="coerce"
                ).dt.normalize()
            if key in out_rows.columns:
                out_rows[key] = pd.to_datetime(
                    out_rows[key], errors="coerce"
                ).dt.normalize()
    return out_existing, out_rows


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """POSIX advisory lock on ``<path>.lock`` (dependency-free, Linux-only).

    Serializes the read-modify-write in ``append_panel_snapshot`` across
    concurrent processes so a snapshot append never loses rows to a racing
    writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append_panel_snapshot(
    name: str, rows: pd.DataFrame, *, dedupe_keys: list[str]
) -> pd.DataFrame:
    """Append ``rows`` to the named panel, dedupe (keep last), persist, return.

    Live-only sources (option chain, FII/DII) have no historical backfill, so
    each scan appends today's snapshot here and the panel accumulates over
    time into a backtestable history. Re-runs on the same key overwrite the
    prior row (``keep="last"``). The read-modify-write is serialized with a
    POSIX file lock and the parquet is written via a per-writer unique temp
    file + atomic ``os.replace`` so concurrent processes can't lose rows or
    clobber each other's ``.tmp``.
    """
    if rows is None or rows.empty:
        existing = read_frame(panel_path(name))
        return existing if existing is not None else pd.DataFrame()
    path = panel_path(name)
    with _file_lock(path):
        existing = read_frame(path)
        existing, rows = _normalize_dedupe_key_dates(existing, rows, dedupe_keys)
        merged = (
            pd.concat([existing, rows], ignore_index=True)
            if existing is not None and not existing.empty
            else rows.copy()
        )
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")
        merged = merged.sort_values(dedupe_keys).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            merged.to_parquet(tmp)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
    return merged


def cached_json_call(
    namespace: str,
    key_parts: Any,
    *,
    ttl_seconds: float | None,
    refresh: bool,
    fetch: Callable[[], T],
) -> T:
    path = cache_path(namespace, stable_key(key_parts), "json")
    if not refresh and is_fresh(path, ttl_seconds):
        cached = read_json(path)
        if cached is not None:
            return cached
    value = fetch()
    write_json(path, value)
    return value


def cached_frame_call(
    namespace: str,
    key_parts: Any,
    *,
    ttl_seconds: float | None,
    refresh: bool,
    fetch: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    path = cache_path(namespace, stable_key(key_parts), "parquet")
    if not refresh and is_fresh(path, ttl_seconds):
        cached = read_frame(path)
        if cached is not None:
            return cached
    frame = fetch()
    write_frame(path, frame)
    return frame
