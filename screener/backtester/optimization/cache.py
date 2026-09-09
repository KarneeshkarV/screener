"""Optimizer result cache: identity-aware SQLite row store.

Legacy JSON cache files are ignored (not destroyed). Cross-run reuse requires an
explicit :class:`FrozenInputIdentity` covering actual prices and membership.
Provider class names and ``repr(fetcher)`` are never part of the key.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel

from screener.backtester.models import BacktestConfig

CACHE_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grid_results (
    cache_key TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    score REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    moments_json TEXT,
    trade_count INTEGER NOT NULL,
    created_ts TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class FrozenInputIdentity:
    """Caller-provided identity of the price and membership inputs."""

    price_snapshot_hash: str
    universe_membership_identity: str
    source: str | None = None


def _digest_len_prefixed(digest: Any, data: bytes) -> None:
    """Append an unambiguous length-prefixed byte span to ``digest``."""
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def hash_price_frames(frames: Mapping[str, pd.DataFrame]) -> str:
    """Content hash of OHLCV frames with length-prefixed field boundaries.

    Includes ticker, dtypes, index identity, column names, and values.
    Does not use object ``repr``.
    """
    digest = hashlib.sha256()
    for ticker in sorted(frames):
        frame = frames[ticker]
        _digest_len_prefixed(digest, str(ticker).encode("utf-8"))
        if frame is None or getattr(frame, "empty", True):
            _digest_len_prefixed(digest, b"empty")
            continue
        ordered = frame.sort_index()
        dtype_payload = [
            [str(name), str(dtype)] for name, dtype in ordered.dtypes.items()
        ]
        _digest_len_prefixed(
            digest, json.dumps(dtype_payload, sort_keys=False).encode("utf-8")
        )
        index = ordered.index
        _digest_len_prefixed(digest, str(index.dtype).encode("utf-8"))
        _digest_len_prefixed(digest, str(getattr(index, "freq", None)).encode("utf-8"))
        index_hash = pd.util.hash_pandas_object(index, index=False).to_numpy()
        _digest_len_prefixed(digest, index_hash.tobytes())
        for column in ordered.columns:
            _digest_len_prefixed(digest, str(column).encode("utf-8"))
            series = ordered[column]
            _digest_len_prefixed(digest, str(series.dtype).encode("utf-8"))
            values = series.to_numpy(copy=False)
            if values.dtype == object:
                encoded = json.dumps(
                    [None if pd.isna(item) else str(item) for item in values],
                    default=str,
                ).encode("utf-8")
                # Object columns: stable string form, length-prefixed as one span.
                _digest_len_prefixed(digest, encoded)
            else:
                _digest_len_prefixed(digest, np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _universe_file_content_identity(universe_file: str | None) -> str | None:
    """Hash universe file bytes when the path exists; else a missing-path marker."""
    if universe_file is None:
        return None
    path = Path(universe_file)
    if path.is_file():
        digest = hashlib.sha256()
        _digest_len_prefixed(digest, b"universe_file_bytes")
        _digest_len_prefixed(digest, path.read_bytes())
        return digest.hexdigest()
    digest = hashlib.sha256()
    _digest_len_prefixed(digest, b"universe_file_missing_path")
    _digest_len_prefixed(digest, str(universe_file).encode("utf-8"))
    return digest.hexdigest()


def universe_membership_identity_from_config(cfg: BacktestConfig) -> str:
    """Stable hash of universe membership fields on ``cfg``.

    ``universe_file`` contributes file **contents** (or a missing-path marker),
    never the path string alone.
    """
    payload = {
        "tickers": list(cfg.tickers) if cfg.tickers is not None else None,
        "universe_file_content": _universe_file_content_identity(cfg.universe_file),
        "membership_windows": [
            [str(a), str(b), None if c is None else str(c)]
            for a, b, c in (cfg.membership_windows or ())
        ],
        "membership_added": [[str(a), str(b)] for a, b in (cfg.membership_added or ())],
        "dynamic_universe_size": cfg.dynamic_universe_size,
        "dynamic_universe_lookback": cfg.dynamic_universe_lookback,
        "dynamic_universe_rebalance": cfg.dynamic_universe_rebalance,
        "max_universe": cfg.max_universe,
        "benchmark": cfg.benchmark,
        "market": cfg.market,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_frozen_input_identity(
    cfg: BacktestConfig,
    frames: Mapping[str, pd.DataFrame],
    *,
    source: str | None = None,
) -> FrozenInputIdentity:
    """Build identity from actual price frames and config membership."""
    return FrozenInputIdentity(
        price_snapshot_hash=hash_price_frames(frames),
        universe_membership_identity=universe_membership_identity_from_config(cfg),
        source=source,
    )


def code_fingerprint(package_root: Path | None = None) -> str:
    """SHA-256 over production ``screener/**/*.py`` plus lock/metadata when present.

    Includes uncommitted Python edits. When ``uv.lock``, ``pyproject.toml``, or
    package metadata files exist next to the package root, their bytes are mixed
    in so dependency changes invalidate cached scores.
    """
    root = package_root
    if root is None:
        root = Path(__file__).resolve().parents[2]  # screener/
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        _digest_len_prefixed(digest, str(path.relative_to(root)).encode("utf-8"))
        _digest_len_prefixed(digest, path.read_bytes())
    repo_root = root.parent
    for name in ("uv.lock", "pyproject.toml", "PKG-INFO", "METADATA"):
        candidate = repo_root / name
        if candidate.is_file():
            _digest_len_prefixed(digest, name.encode("utf-8"))
            _digest_len_prefixed(digest, candidate.read_bytes())
        # Also check inside the package root (editable / sdist layouts).
        nested = root / name
        if nested.is_file() and nested != candidate:
            _digest_len_prefixed(digest, f"pkg:{name}".encode("utf-8"))
            _digest_len_prefixed(digest, nested.read_bytes())
    return digest.hexdigest()


def sqlite_cache_path(cache_path: Path | str) -> Path:
    """Sibling SQLite path for a requested cache location."""
    path = Path(cache_path)
    if path.suffix.lower() in {".sqlite", ".db"}:
        return path
    if path.suffix.lower() == ".json":
        return path.with_suffix(".sqlite")
    return path.with_name(path.name + ".sqlite")


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def sanitize_for_strict_json(value: Any) -> Any:
    """Replace non-finite floats with ``None`` (JSON null). Keep structure otherwise."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, dict):
        return {str(k): sanitize_for_strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_strict_json(v) for v in value]
    return value


def dumps_strict_json(value: Any, *, sort_keys: bool = True) -> str:
    """JSON dump that never emits NaN/Infinity tokens (uses null instead)."""
    return json.dumps(
        sanitize_for_strict_json(value),
        sort_keys=sort_keys,
        allow_nan=False,
        default=_json_default,
    )


def loads_metrics_json(raw: str) -> dict[str, float]:
    """Load metrics JSON; JSON null becomes ``nan`` for in-memory model use."""
    data = json.loads(raw)
    out: dict[str, float] = {}
    for key, value in data.items():
        if value is None:
            out[str(key)] = float("nan")
        else:
            out[str(key)] = float(value)
    return out


def _stable_fingerprint(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_stable_fingerprint(item) for item in value]
    if isinstance(value, list):
        return [_stable_fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, BaseModel):
        return {
            "__class__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": _stable_fingerprint(value.model_dump()),
        }
    return {
        "__class__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "repr": repr(value),
    }


def config_fingerprint(cfg: BacktestConfig) -> dict[str, Any]:
    data = cfg.model_dump(exclude={"slippage_model"})
    data["slippage_model"] = _stable_fingerprint(cfg.slippage_model)
    return data


def make_cache_key(
    cfg: BacktestConfig,
    params: dict[str, Any],
    *,
    runner: str,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
    frozen_input_identity: FrozenInputIdentity,
    code_fp: str | None = None,
    n_trials: int = 1,
) -> str:
    """Build the v1 cache key. Requires an explicit frozen input identity.

    ``n_trials`` is part of the key because a cached Deflated Sharpe was
    deflated against the grid that produced it.
    """
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "code_fingerprint": code_fp if code_fp is not None else code_fingerprint(),
        "config": config_fingerprint(cfg),
        "params": _stable_fingerprint(params),
        "runner": runner,
        "start_date": None if start_date is None else start_date.isoformat(),
        "end_date": None if end_date is None else end_date.isoformat(),
        "metric": metric,
        "min_trades": min_trades,
        "n_trials": int(n_trials),
        "input": {
            "price_snapshot_hash": frozen_input_identity.price_snapshot_hash,
            "universe_membership_identity": (
                frozen_input_identity.universe_membership_identity
            ),
            "source": frozen_input_identity.source,
        },
    }
    raw = dumps_strict_json(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResultCache:
    """Transactional SQLite result cache. Legacy JSON is never read or deleted."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = None if path is None else sqlite_cache_path(path)
        self._code_fp: str | None = None

    @property
    def code_fp(self) -> str:
        if self._code_fp is None:
            self._code_fp = code_fingerprint()
        return self._code_fp

    def _connect(self) -> sqlite3.Connection | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT params_json, score, metrics_json, moments_json, trade_count
                FROM grid_results
                WHERE cache_key = ? AND schema_version = ?
                """,
                (key, CACHE_SCHEMA_VERSION),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        params_json, score, metrics_json, moments_json, trade_count = row
        score_value = float(score)
        return {
            "params": json.loads(params_json),
            "score": score_value,
            "metrics": loads_metrics_json(metrics_json),
            "moments": None if not moments_json else json.loads(moments_json),
            "trade_count": int(trade_count),
            "error": None,
        }

    def put(
        self,
        key: str,
        *,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, float],
        trade_count: int,
        moments: dict[str, Any] | None = None,
    ) -> None:
        """Upsert one successful deterministic result. Never stores error rows."""
        conn = self._connect()
        if conn is None:
            return
        score_value = float(score)
        # SQLite REAL can store -inf; reject other non-finite scores as unsafe.
        if math.isnan(score_value):
            score_value = float("-inf")
        try:
            conn.execute(
                """
                INSERT INTO grid_results (
                    cache_key, schema_version, params_json, score,
                    metrics_json, moments_json, trade_count, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(cache_key) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    params_json=excluded.params_json,
                    score=excluded.score,
                    metrics_json=excluded.metrics_json,
                    moments_json=excluded.moments_json,
                    trade_count=excluded.trade_count,
                    created_ts=excluded.created_ts
                """,
                (
                    key,
                    CACHE_SCHEMA_VERSION,
                    dumps_strict_json(params),
                    score_value,
                    dumps_strict_json(metrics),
                    None if moments is None else dumps_strict_json(moments),
                    int(trade_count),
                ),
            )
            conn.commit()
        finally:
            conn.close()
