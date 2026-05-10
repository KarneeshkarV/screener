"""Feature usage tracking backed by Turso/libSQL."""

from __future__ import annotations

import getpass
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

PROJECT_NAME = "screener"
TABLE_NAME = "feature_usage"


class UsageClient(Protocol):
    def execute(self, stmt: str, args: list[object] | None = None): ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class UsageCount:
    feature: str
    count: int
    last_used_at: str | None


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
