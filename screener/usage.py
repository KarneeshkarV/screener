"""Feature usage tracking backed by Turso/libSQL."""

from __future__ import annotations

import atexit
import getpass
import json
import logging
import os
import platform
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, field_validator

from screener import _optional

logger = logging.getLogger(__name__)

PROJECT_NAME = "screener"
TABLE_NAME = "feature_usage"
INVOCATIONS_TABLE = "feature_usage_invocations"

# SCREENER_USAGE=0|off|false|no skips all usage I/O (local opt-out).
_USAGE_OFF_VALUES = frozenset({"0", "off", "false", "no"})

_FLATTENED_PARAM_KEYS = {
    "market",
    "criteria_names",
    "limit",
    "refresh",
    "output_csv",
    "cache_ttl",
}

# Process-level Turso client reuse: one connect, tables ensured once.
_client: UsageClient | None = None
_client_lock = threading.Lock()
_usage_table_ready = False
_invocations_table_ready = False

# Non-blocking write path: stage rows, flush on a daemon thread, best-effort join.
# SCREENER_USAGE_FLUSH_MS overrides the join budget (default 50 ms).
_DEFAULT_FLUSH_TIMEOUT_S = 0.05
# Interpreter-exit grace period: covers a remote Turso round-trip so an
# in-flight flush finishes instead of being killed with the process.
_ATEXIT_FLUSH_TIMEOUT_S = 1.0
# Bounded join for the test-reset helper: a wedged worker must not hang the suite.
_RESET_JOIN_TIMEOUT_S = 2.0
_pending_lock = threading.Lock()
_pending_usage: deque[dict[str, Any]] = deque()
_pending_invocation: deque[dict[str, Any]] = deque()
_flush_thread: threading.Thread | None = None


class UsageClient(Protocol):
    def execute(self, stmt: str, args: list[object] | None = None): ...

    def close(self) -> None: ...


class UsageCount(BaseModel):
    feature: str
    count: int
    last_used_at: str | None

    model_config = ConfigDict(frozen=True)

    @field_validator("feature")
    @classmethod
    def _normalize_feature(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("feature must not be empty")
        return normalized


class InvocationRollup(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    market: str
    criteria: str
    status: str
    count: int
    last_used_at: str | None
    top_extras: str


def _load_env_file(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(name: str) -> str | None:
    return os.environ.get(name) or _load_env_file().get(name)


def _database_url() -> str | None:
    url = _env_value("TURSO_DATABASE_URL")
    if url and url.startswith("libsql://"):
        return url.replace("libsql://", "https://", 1)
    return url


def _usage_disabled() -> bool:
    """True when usage tracking should no-op (tests or SCREENER_USAGE opt-out)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    raw = os.environ.get("SCREENER_USAGE", "").strip().lower()
    return raw in _USAGE_OFF_VALUES


def _connect() -> UsageClient | None:
    """Return a cached Turso client, creating it on first success in this process.

    Sequential ``record_feature_usage`` + ``record_feature_invocation`` share one
    HTTPS client (no double connect). The client lives for the process: writers
    and readers never close it under each other, and the atexit handler closes
    it at exit so libsql does not keep the process alive. Failures return None
    and do not poison later retries.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        url = _database_url()
        token = _env_value("TURSO_AUTH_TOKEN")
        if not url or not token:
            return None

        create_client_sync = _optional.load("libsql_client").create_client_sync

        client = cast(UsageClient, create_client_sync(url, auth_token=token))
        _client = client
        return client


def _close_client() -> None:
    """Close and drop the cached client (keeps DDL-ensured flags)."""
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        try:
            client.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _reset_client_state() -> None:
    """Drop cached client, pending rows, and DDL flags (for tests).

    Discards staged rows rather than flushing them: this runs from the autouse
    fixture on teardown, after monkeypatch is undone, so a flush here would use
    the real credentials and publish test rows to the real database. Every join
    is bounded so a wedged worker cannot hang the suite.
    """
    global _usage_table_ready, _invocations_table_ready, _flush_thread
    with _pending_lock:
        thread = _flush_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=_RESET_JOIN_TIMEOUT_S)
    _close_client()
    with _pending_lock:
        _pending_usage.clear()
        _pending_invocation.clear()
        if _flush_thread is None or not _flush_thread.is_alive():
            _flush_thread = None
    _usage_table_ready = False
    _invocations_table_ready = False


def _flush_timeout_s() -> float:
    raw = os.environ.get("SCREENER_USAGE_FLUSH_MS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw) / 1000.0)
        except ValueError:
            pass
    return _DEFAULT_FLUSH_TIMEOUT_S


def _write_usage_row(
    client: UsageClient,
    *,
    feature: str,
    command_path: str | None,
    status: str,
    duration_ms: int,
    username: str,
    hostname: str,
) -> None:
    _ensure_usage_table_once(client)
    client.execute(
        f"""
        INSERT INTO {TABLE_NAME}
            (project, feature, command_path, status, duration_ms, username, hostname)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            PROJECT_NAME,
            feature,
            command_path or feature,
            status,
            int(duration_ms),
            username,
            hostname,
        ],
    )


def _write_invocation_row(
    client: UsageClient,
    *,
    feature: str,
    duration_ms: int,
    status: str,
    params: dict[str, Any],
    username: str,
    hostname: str,
) -> None:
    _ensure_invocations_table_once(client)
    market = params.get("market")
    criteria = _normalize_criteria(params.get("criteria_names"))
    limit_n = params.get("limit")
    refresh = params.get("refresh")
    output_csv = params.get("output_csv")
    cache_ttl = params.get("cache_ttl")

    extras: dict[str, str] = {}
    for key, value in params.items():
        if key in _FLATTENED_PARAM_KEYS:
            continue
        if value is None:
            continue
        extras[key] = str(value)
    extras_json = json.dumps(extras, default=str) if extras else None

    client.execute(
        f"""
        INSERT INTO {INVOCATIONS_TABLE}
            (project, feature, market, criteria, limit_n, refresh,
             output_csv, cache_ttl, extras_json, duration_ms, status,
             username, hostname)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            PROJECT_NAME,
            feature,
            str(market) if market is not None else None,
            criteria,
            int(limit_n) if limit_n is not None else None,
            _coerce_bool_to_int(refresh) if refresh is not None else None,
            str(output_csv) if output_csv is not None else None,
            str(cache_ttl) if cache_ttl is not None else None,
            extras_json,
            int(duration_ms),
            status,
            username,
            hostname,
        ],
    )


def _drain_pending() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Take every staged row. Caller owns the lists."""
    with _pending_lock:
        usage_rows = list(_pending_usage)
        inv_rows = list(_pending_invocation)
        _pending_usage.clear()
        _pending_invocation.clear()
    return usage_rows, inv_rows


def _has_pending() -> bool:
    with _pending_lock:
        return bool(_pending_usage or _pending_invocation)


def _flush_pending_sync() -> None:
    """Write staged rows on one connection; loop until the queues are empty.

    The worker keeps draining until nothing is pending, so rows staged while a
    flush is already in flight are written instead of stranded. Runs on the
    daemon flush thread and inline from ``flush_usage``. The shared client is
    left open for other callers; the atexit handler closes it.
    """
    global _flush_thread
    try:
        while True:
            usage_rows, inv_rows = _drain_pending()
            if not usage_rows and not inv_rows:
                return
            client = _connect()
            if client is None:
                return
            for usage_row in usage_rows:
                _write_usage_row(client, **usage_row)
            for inv_row in inv_rows:
                _write_invocation_row(client, **inv_row)
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        logger.debug("feature usage flush failed: %s", exc)
    finally:
        # Release the slot on every exit path so a later staged row starts a
        # fresh worker instead of joining this dead one.
        with _pending_lock:
            if _flush_thread is threading.current_thread():
                _flush_thread = None


def _start_flush_worker() -> threading.Thread:
    """Start a daemon worker if none is alive. Returns the live thread."""
    global _flush_thread
    with _pending_lock:
        existing = _flush_thread
        if existing is not None and existing.is_alive():
            return existing
        thread = threading.Thread(
            target=_flush_pending_sync,
            name="screener-usage-flush",
            daemon=True,
        )
        _flush_thread = thread
        thread.start()
        return thread


def flush_usage(timeout_s: float | None = None) -> None:
    """Best-effort wait for a background usage flush (or run one inline).

    Returns once every staged row is written and no flush worker is still
    running, or when the time budget expires. This is the explicit-flush API;
    the interactive CLI path stays fast via the short ``_schedule_flush`` join.
    """
    global _flush_thread
    budget = _flush_timeout_s() if timeout_s is None else max(0.0, timeout_s)
    deadline = time.monotonic() + budget
    while True:
        with _pending_lock:
            thread = _flush_thread
        # _has_pending takes the same lock, so read it outside the block above.
        has_pending = _has_pending()
        if not has_pending and (thread is None or not thread.is_alive()):
            return
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        elif has_pending:
            _flush_pending_sync()
        if time.monotonic() >= deadline:
            return


def _flush_pending_on_exit() -> None:
    """Best-effort flush of pending rows at interpreter exit, then release client."""
    try:
        flush_usage(timeout_s=_ATEXIT_FLUSH_TIMEOUT_S)
    except Exception:  # pragma: no cover - defensive exit path
        pass
    _close_client()


atexit.register(_flush_pending_on_exit)


def _schedule_flush() -> None:
    """Start a daemon flush thread and best-effort join for a short budget."""
    thread = _start_flush_worker()
    thread.join(timeout=_flush_timeout_s())
    # Keep the caller budget at one join. If rows arrived after this worker
    # exited, start another worker and do not wait again.
    if _has_pending():
        with _pending_lock:
            current = _flush_thread
        if current is None or not current.is_alive():
            _start_flush_worker()


def ensure_usage_table(client: UsageClient) -> None:
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            project TEXT NOT NULL,
            feature TEXT NOT NULL,
            command_path TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            username TEXT NOT NULL,
            hostname TEXT NOT NULL
        )
        """
    )
    client.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_project_feature
        ON {TABLE_NAME} (project, feature)
        """
    )


def ensure_invocations_table(client: UsageClient) -> None:
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {INVOCATIONS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            project TEXT NOT NULL,
            feature TEXT NOT NULL,
            market TEXT,
            criteria TEXT,
            limit_n INTEGER,
            refresh INTEGER,
            output_csv TEXT,
            cache_ttl TEXT,
            extras_json TEXT,
            duration_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            username TEXT NOT NULL,
            hostname TEXT NOT NULL
        )
        """
    )
    client.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{INVOCATIONS_TABLE}_project_feature
        ON {INVOCATIONS_TABLE} (project, feature)
        """
    )


def _ensure_usage_table_once(client: UsageClient) -> None:
    global _usage_table_ready
    if _usage_table_ready:
        return
    ensure_usage_table(client)
    _usage_table_ready = True


def _ensure_invocations_table_once(client: UsageClient) -> None:
    global _invocations_table_ready
    if _invocations_table_ready:
        return
    ensure_invocations_table(client)
    _invocations_table_ready = True


def _coerce_bool_to_int(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def _normalize_criteria(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value if v is not None]
        if not parts:
            return None
        return ",".join(parts)
    return str(value)


def record_feature_invocation(
    feature: str,
    *,
    command_path: str | None = None,
    duration_ms: int = 0,
    status: str = "success",
    params: dict[str, Any] | None = None,
) -> None:
    """Stage one CLI invocation row and flush pending usage on a daemon thread.

    CLI always calls this last in ``finally``. Network I/O runs in the
    background; the caller only waits up to ``SCREENER_USAGE_FLUSH_MS``
    (default 50 ms) so Turso RTT does not dominate process wall time.
    """
    if _usage_disabled():
        return
    # Capture identity on the calling thread (getpass/platform can be slow-ish
    # but stay off the Turso critical path relative to HTTPS).
    row = {
        "feature": feature,
        "duration_ms": int(duration_ms),
        "status": status,
        "params": dict(params or {}),
        "username": getpass.getuser(),
        "hostname": platform.node(),
    }
    with _pending_lock:
        _pending_invocation.append(row)
    _schedule_flush()


def invocation_rollup(limit: int = 30) -> list[InvocationRollup]:
    client = _connect()
    if client is None:
        return []
    try:
        _ensure_invocations_table_once(client)
        rows = client.execute(
            f"""
            SELECT feature,
                   COALESCE(market, '') AS market,
                   COALESCE(criteria, '') AS criteria,
                   status,
                   created_at,
                   extras_json
            FROM {INVOCATIONS_TABLE}
            WHERE project = ?
            """,
            [PROJECT_NAME],
        ).rows

        groups: dict[
            tuple[str, str, str, str],
            dict[str, Any],
        ] = {}
        for row in rows:
            feature = str(row[0])
            market = str(row[1])
            criteria = str(row[2])
            status = str(row[3])
            created_at = str(row[4]) if row[4] is not None else None
            extras_raw = row[5]

            key = (feature, market, criteria, status)
            entry = groups.setdefault(
                key,
                {"count": 0, "last_used_at": None, "extras_counter": {}},
            )
            entry["count"] += 1
            if created_at and (
                entry["last_used_at"] is None or created_at > entry["last_used_at"]
            ):
                entry["last_used_at"] = created_at

            if extras_raw is None:
                continue
            try:
                payload = json.loads(str(extras_raw))
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            counter: dict[str, dict[str, int]] = entry["extras_counter"]
            for k, v in payload.items():
                key_s = str(k)
                val_s = str(v)
                counter.setdefault(key_s, {})
                counter[key_s][val_s] = counter[key_s].get(val_s, 0) + 1

        sorted_groups = sorted(
            groups.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["last_used_at"] or ""),
            reverse=True,
        )

        results: list[InvocationRollup] = []
        for (feature, market, criteria, status), entry in sorted_groups[: int(limit)]:
            counter = entry["extras_counter"]
            top_parts: list[tuple[int, str]] = []
            for key_s, vals in counter.items():
                best_val, best_count = max(vals.items(), key=lambda kv: kv[1])
                top_parts.append((best_count, f"{key_s}={best_val}"))
            top_parts.sort(key=lambda kv: kv[0], reverse=True)
            top_extras = ", ".join(part for _, part in top_parts[:3])

            results.append(
                InvocationRollup(
                    feature=feature,
                    market=market,
                    criteria=criteria,
                    status=status,
                    count=entry["count"],
                    last_used_at=entry["last_used_at"],
                    top_extras=top_extras,
                )
            )
        return results
    except Exception as exc:  # pragma: no cover - defensive read path
        logger.debug("invocation rollup failed: %s", exc)
        return []


def record_feature_usage(
    feature: str,
    *,
    command_path: str | None = None,
    status: str = "success",
    duration_ms: int = 0,
) -> None:
    """Stage one successful CLI feature usage row (flushed with invocation).

    Does not block on Turso. A following ``record_feature_invocation`` (or an
    explicit ``flush_usage``) ships staged rows on one shared connection.
    """
    if _usage_disabled():
        return
    row = {
        "feature": feature,
        "command_path": command_path,
        "status": status,
        "duration_ms": int(duration_ms),
        "username": getpass.getuser(),
        "hostname": platform.node(),
    }
    with _pending_lock:
        _pending_usage.append(row)


def feature_usage_counts() -> list[UsageCount]:
    client = _connect()
    if client is None:
        return []
    try:
        _ensure_usage_table_once(client)
        rows = client.execute(
            f"""
            SELECT feature, COUNT(*) AS usage_count, MAX(created_at) AS last_used_at
            FROM {TABLE_NAME}
            WHERE project = ? AND status = 'success'
            GROUP BY feature
            ORDER BY usage_count DESC, feature ASC
            """,
            [PROJECT_NAME],
        ).rows
        return [
            UsageCount(
                feature=str(row[0]),
                count=int(row[1]),
                last_used_at=str(row[2]) if row[2] is not None else None,
            )
            for row in rows
        ]
    except Exception as exc:  # pragma: no cover - defensive read path
        logger.debug("feature usage counts failed: %s", exc)
        return []


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))
