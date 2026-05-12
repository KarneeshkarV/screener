"""Feature usage tracking backed by Turso/libSQL."""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)

PROJECT_NAME = "screener"
TABLE_NAME = "feature_usage"
INVOCATIONS_TABLE = "feature_usage_invocations"

_FLATTENED_PARAM_KEYS = {
    "market",
    "criteria_names",
    "limit",
    "refresh",
    "output_csv",
    "cache_ttl",
}


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


def _connect() -> UsageClient | None:
    url = _database_url()
    token = _env_value("TURSO_AUTH_TOKEN")
    if not url or not token:
        return None

    from libsql_client import create_client_sync  # type: ignore[import-untyped]

    return create_client_sync(url, auth_token=token)


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
    """Record one CLI invocation with its full Click parameter payload."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        client = _connect()
        if client is None:
            return
        try:
            ensure_invocations_table(client)
            params = params or {}

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
                    getpass.getuser(),
                    platform.node(),
                ],
            )
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        logger.debug("feature invocation tracking failed: %s", exc)


def invocation_rollup(limit: int = 30) -> list[InvocationRollup]:
    client = _connect()
    if client is None:
        return []
    try:
        ensure_invocations_table(client)
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
    finally:
        client.close()


def record_feature_usage(
    feature: str,
    *,
    command_path: str | None = None,
    status: str = "success",
    duration_ms: int = 0,
) -> None:
    """Record one successful CLI feature usage without affecting CLI behavior."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        client = _connect()
        if client is None:
            return
        try:
            ensure_usage_table(client)
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
                    getpass.getuser(),
                    platform.node(),
                ],
            )
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        logger.debug("feature usage tracking failed: %s", exc)


def feature_usage_counts() -> list[UsageCount]:
    client = _connect()
    if client is None:
        return []
    try:
        ensure_usage_table(client)
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
    finally:
        client.close()


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))
