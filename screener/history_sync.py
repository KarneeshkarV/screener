"""Back up local screen-run history to Turso/libSQL (and restore it back).

The local store (:mod:`screener.history`) is a single-machine SQLite file. This
module mirrors it into two remote tables — ``screen_runs`` and
``screen_run_rows`` — so history survives a lost laptop and can be pulled onto a
fresh machine. Pushes and pulls are idempotent: runs are keyed on their natural
``(run_ts, market, criteria)`` tuple and rows additionally on ``ticker``, so
repeating a backup never duplicates data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from screener import history, usage

REMOTE_RUNS_TABLE = "screen_runs"
REMOTE_ROWS_TABLE = "screen_run_rows"

_ROW_COLUMNS = (
    "ticker",
    "name",
    "close",
    "change",
    "volume",
    "market_cap",
    "setup_score",
    "rank",
)


class SyncClient(Protocol):
    """The subset of the libsql sync client this module relies on."""

    def execute(self, stmt: str, args: list[object] | None = None) -> Any: ...

    def batch(self, stmts: list[tuple[str, list[object]]]) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BackupSummary:
    runs_pushed: int
    rows_pushed: int
    remote_runs: int
    remote_rows: int


@dataclass(frozen=True)
class RestoreSummary:
    runs_restored: int
    rows_restored: int
    local_runs: int


def connect() -> SyncClient | None:
    """Return a Turso client, or ``None`` when credentials are absent.

    Reuses :mod:`screener.usage`'s env resolution — including the
    ``libsql://`` -> ``https://`` rewrite the account's endpoint needs.
    """
    url = usage._database_url()
    token = usage._env_value("TURSO_AUTH_TOKEN")
    if not url or not token:
        return None

    from libsql_client import create_client_sync  # type: ignore[import-untyped]

    return cast(SyncClient, create_client_sync(url, auth_token=token))


def ensure_remote_tables(client: SyncClient) -> None:
    """Create the mirror tables if they do not yet exist (idempotent)."""
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REMOTE_RUNS_TABLE} (
            id            INTEGER,
            run_ts        TEXT NOT NULL,
            market        TEXT NOT NULL,
            criteria      TEXT NOT NULL,
            total_matches INTEGER NOT NULL,
            UNIQUE(run_ts, market, criteria)
        )
        """
    )
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REMOTE_ROWS_TABLE} (
            run_id      INTEGER NOT NULL,
            run_ts      TEXT NOT NULL,
            market      TEXT NOT NULL,
            criteria    TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            name        TEXT,
            close       REAL,
            change      REAL,
            volume      REAL,
            market_cap  REAL,
            setup_score REAL,
            rank        INTEGER NOT NULL,
            UNIQUE(run_ts, market, criteria, ticker)
        )
        """
    )


def _run_batches(
    client: SyncClient, statements: list[tuple[str, list[object]]], batch_size: int
) -> None:
    for start in range(0, len(statements), batch_size):
        client.batch(statements[start : start + batch_size])


def _remote_run_keys(client: SyncClient) -> set[tuple[str, str, str]]:
    result = client.execute(f"SELECT run_ts, market, criteria FROM {REMOTE_RUNS_TABLE}")
    return {(str(r[0]), str(r[1]), str(r[2])) for r in result.rows}


def _remote_totals(client: SyncClient) -> tuple[int, int]:
    runs = client.execute(f"SELECT COUNT(*) FROM {REMOTE_RUNS_TABLE}").rows[0][0]
    rows = client.execute(f"SELECT COUNT(*) FROM {REMOTE_ROWS_TABLE}").rows[0][0]
    return int(runs), int(rows)


def _local_runs() -> list[tuple[int, str, str, str, int]]:
    conn = history._connect()
    try:
        return [
            (int(r[0]), str(r[1]), str(r[2]), str(r[3]), int(r[4]))
            for r in conn.execute(
                "SELECT id, run_ts, market, criteria, total_matches "
                "FROM runs ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()


def _local_rows(run_id: int) -> list[tuple[object, ...]]:
    conn = history._connect()
    try:
        return [
            tuple(r)
            for r in conn.execute(
                f"SELECT {', '.join(_ROW_COLUMNS)} FROM run_rows "
                "WHERE run_id = ? ORDER BY rank",
                (run_id,),
            ).fetchall()
        ]
    finally:
        conn.close()


def backup_history(client: SyncClient, *, batch_size: int = 200) -> BackupSummary:
    """Push every local run not already on the remote, with its rows."""
    ensure_remote_tables(client)
    existing = _remote_run_keys(client)
    to_push = [run for run in _local_runs() if (run[1], run[2], run[3]) not in existing]

    run_stmts: list[tuple[str, list[object]]] = [
        (
            f"""
            INSERT INTO {REMOTE_RUNS_TABLE} (id, run_ts, market, criteria, total_matches)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_ts, market, criteria)
            DO UPDATE SET id = excluded.id, total_matches = excluded.total_matches
            """,
            [run_id, run_ts, market, criteria, total],
        )
        for run_id, run_ts, market, criteria, total in to_push
    ]
    _run_batches(client, run_stmts, batch_size)

    row_stmts: list[tuple[str, list[object]]] = []
    for run_id, run_ts, market, criteria, _total in to_push:
        for row in _local_rows(run_id):
            row_stmts.append(
                (
                    f"""
                    INSERT INTO {REMOTE_ROWS_TABLE}
                        (run_id, run_ts, market, criteria, ticker, name, close,
                         change, volume, market_cap, setup_score, rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_ts, market, criteria, ticker) DO UPDATE SET
                        run_id = excluded.run_id, name = excluded.name,
                        close = excluded.close, change = excluded.change,
                        volume = excluded.volume, market_cap = excluded.market_cap,
                        setup_score = excluded.setup_score, rank = excluded.rank
                    """,
                    [run_id, run_ts, market, criteria, *row],
                )
            )
    _run_batches(client, row_stmts, batch_size)

    remote_runs, remote_rows = _remote_totals(client)
    return BackupSummary(
        runs_pushed=len(to_push),
        rows_pushed=len(row_stmts),
        remote_runs=remote_runs,
        remote_rows=remote_rows,
    )


def _local_run_keys() -> set[tuple[str, str, str]]:
    conn = history._connect()
    try:
        return {
            (str(r[0]), str(r[1]), str(r[2]))
            for r in conn.execute(
                "SELECT run_ts, market, criteria FROM runs"
            ).fetchall()
        }
    finally:
        conn.close()


def restore_history(client: SyncClient) -> RestoreSummary:
    """Pull remote runs whose natural key is missing locally into the local DB."""
    ensure_remote_tables(client)
    local_keys = _local_run_keys()

    remote_runs = client.execute(
        f"SELECT run_ts, market, criteria, total_matches FROM {REMOTE_RUNS_TABLE}"
    ).rows
    missing = [
        r for r in remote_runs if (str(r[0]), str(r[1]), str(r[2])) not in local_keys
    ]

    runs_restored = 0
    rows_restored = 0
    conn = history._connect()
    try:
        for run in missing:
            run_ts, market, criteria, total = (
                str(run[0]),
                str(run[1]),
                str(run[2]),
                int(run[3]),
            )
            conn.execute(
                "INSERT INTO runs (run_ts, market, criteria, total_matches) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(run_ts, market, criteria) DO NOTHING",
                (run_ts, market, criteria, total),
            )
            id_row = conn.execute(
                "SELECT id FROM runs WHERE run_ts = ? AND market = ? AND criteria = ?",
                (run_ts, market, criteria),
            ).fetchone()
            local_id = int(id_row[0])

            remote_rows = client.execute(
                f"SELECT {', '.join(_ROW_COLUMNS)} FROM {REMOTE_ROWS_TABLE} "
                "WHERE run_ts = ? AND market = ? AND criteria = ? ORDER BY rank",
                [run_ts, market, criteria],
            ).rows
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO run_rows
                    (run_id, {", ".join(_ROW_COLUMNS)})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(local_id, *tuple(row)) for row in remote_rows],
            )
            runs_restored += 1
            rows_restored += len(remote_rows)
        conn.commit()
        local_total = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    finally:
        conn.close()

    return RestoreSummary(
        runs_restored=runs_restored,
        rows_restored=rows_restored,
        local_runs=local_total,
    )


__all__ = [
    "BackupSummary",
    "RestoreSummary",
    "SyncClient",
    "backup_history",
    "connect",
    "ensure_remote_tables",
    "restore_history",
]
