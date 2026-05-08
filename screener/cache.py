"""Small on-disk cache helpers for external market-data providers."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd


CACHE_ROOT = Path.home() / ".screener" / "cache"
T = TypeVar("T")


def stable_key(*parts: Any) -> str:
    """Return a deterministic key for JSON-serializable or repr-able parts."""
    payload = json.dumps(parts, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_ttl(value: str | int | float | None, *, default: float | None = None) -> float | None:
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


def is_fresh(path: Path, ttl_seconds: float | None, *, now: float | None = None) -> bool:
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
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, default=str))
    tmp.replace(path)


def read_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)


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
