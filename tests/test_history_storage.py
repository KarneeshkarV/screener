"""Tests for local history hardening and the Turso backup/restore path."""

from __future__ import annotations

import importlib
import sqlite3
import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import history as history_mod
from screener import history_sync
from screener.cli import cli


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


def _screen_df(tickers):
    return pd.DataFrame(
        [
            {
                "name": t,
                "description": f"{t} Corp",
                "close": 10.0 + i,
                "change": 1.0,
                "volume": 1_000_000,
                "market_cap_basic": 1_000_000_000,
                "setup_score": 50.0 + i,
            }
            for i, t in enumerate(tickers)
        ]
    )


# --- A. Local SQLite hardening -------------------------------------------------


def test_delete_run_cascades_to_rows(history_db):
    run_id = history_mod.save_run("us", "ema", 2, _screen_df(["AAA", "BBB"]))

    conn = history_mod._connect()
    try:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM run_rows WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    assert remaining == 0


def test_connect_enables_pragmas(history_db):
    conn = history_mod._connect()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_save_run_collision_reuses_id_and_replaces_rows(history_db, monkeypatch):
    frozen = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - mimic datetime.now signature
            return frozen if tz is None else frozen.astimezone(tz)

    monkeypatch.setattr(history_mod, "datetime", _FrozenDatetime)

    first = history_mod.save_run("us", "ema", 2, _screen_df(["AAA", "BBB"]))
    second = history_mod.save_run("us", "ema", 1, _screen_df(["CCC"]))

    assert first == second
    snap = history_mod.load_run(second)
    assert snap is not None
    assert snap.total_matches == 1
    assert snap.tickers == ["CCC"]


def test_db_path_honors_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "history.db"
    monkeypatch.setenv("SCREENER_HISTORY_DB", str(target))
    try:
        reloaded = importlib.reload(history_mod)
        assert reloaded.DB_PATH == target
    finally:
        monkeypatch.delenv("SCREENER_HISTORY_DB", raising=False)
        importlib.reload(history_mod)


# --- B. Turso backup / restore -------------------------------------------------


class StubTursoClient:
    """In-memory SQLite standing in for the libsql sync client."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.closed = False

    def execute(self, stmt: str, args: list[object] | None = None):
        cur = self.conn.execute(stmt, tuple(args) if args else ())
        rows = cur.fetchall() if stmt.lstrip().upper().startswith("SELECT") else []
        self.conn.commit()
        return types.SimpleNamespace(rows=rows)

    def batch(self, stmts):
        return [self.execute(sql, args) for sql, args in stmts]

    def close(self) -> None:
        self.closed = True


def test_backup_pushes_runs_and_is_idempotent(history_db):
    history_mod.save_run("us", "ema", 2, _screen_df(["AAA", "BBB"]))
    history_mod.save_run("india", "breakout", 1, _screen_df(["XXX"]))

    client = StubTursoClient()
    summary = history_sync.backup_history(client, batch_size=2)

    assert summary.runs_pushed == 2
    assert summary.rows_pushed == 3
    assert summary.remote_runs == 2
    assert summary.remote_rows == 3

    again = history_sync.backup_history(client)
    assert again.runs_pushed == 0
    assert again.rows_pushed == 0
    assert again.remote_runs == 2
    assert again.remote_rows == 3


def test_restore_pulls_missing_runs(history_db, tmp_path, monkeypatch):
    history_mod.save_run("us", "ema", 2, _screen_df(["AAA", "BBB"]))
    client = StubTursoClient()
    history_sync.backup_history(client)

    fresh = tmp_path / "fresh.db"
    monkeypatch.setattr(history_mod, "DB_PATH", fresh)

    summary = history_sync.restore_history(client)
    assert summary.runs_restored == 1
    assert summary.rows_restored == 2
    assert summary.local_runs == 1

    restored = history_mod.list_runs()
    assert restored.iloc[0]["market"] == "us"
    snap = history_mod.load_run(int(restored.iloc[0]["id"]))
    assert snap is not None
    assert snap.tickers == ["AAA", "BBB"]

    # Restoring again is a no-op now that the key exists locally.
    second = history_sync.restore_history(client)
    assert second.runs_restored == 0


def test_connect_returns_none_without_credentials(monkeypatch):
    monkeypatch.setattr(history_sync.usage, "_database_url", lambda: None)
    monkeypatch.setattr(history_sync.usage, "_env_value", lambda name: None)
    assert history_sync.connect() is None


def test_connect_builds_client_when_configured(monkeypatch):
    created = {}

    def create_client_sync(url, auth_token):
        created["url"] = url
        created["auth_token"] = auth_token
        return "client"

    monkeypatch.setattr(history_sync.usage, "_database_url", lambda: "https://remote")
    monkeypatch.setattr(history_sync.usage, "_env_value", lambda name: "token")
    monkeypatch.setitem(
        sys.modules,
        "libsql_client",
        types.SimpleNamespace(create_client_sync=create_client_sync),
    )

    assert history_sync.connect() == "client"
    assert created == {"url": "https://remote", "auth_token": "token"}


def test_history_backup_command_reports_summary(monkeypatch):
    client = StubTursoClient()
    monkeypatch.setattr(history_sync, "connect", lambda: client)
    monkeypatch.setattr(
        history_sync,
        "backup_history",
        lambda c, **kw: history_sync.BackupSummary(3, 5, 3, 5),
    )

    res = CliRunner().invoke(cli, ["history-backup"])
    assert res.exit_code == 0, res.output
    assert "Pushed 3 runs (5 rows)" in res.output
    assert client.closed


def test_history_backup_command_without_credentials(monkeypatch):
    monkeypatch.setattr(history_sync, "connect", lambda: None)

    res = CliRunner().invoke(cli, ["history-backup"])
    assert res.exit_code == 1
    assert "Turso is not configured" in res.output


def test_history_backup_command_handles_failure(monkeypatch):
    client = StubTursoClient()
    monkeypatch.setattr(history_sync, "connect", lambda: client)

    def boom(c, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(history_sync, "backup_history", boom)

    res = CliRunner().invoke(cli, ["history-backup"])
    assert res.exit_code == 1
    assert "history-backup failed: network down" in res.output
    assert client.closed


def test_history_backup_command_restore(monkeypatch):
    client = StubTursoClient()
    monkeypatch.setattr(history_sync, "connect", lambda: client)
    monkeypatch.setattr(
        history_sync,
        "restore_history",
        lambda c, **kw: history_sync.RestoreSummary(2, 4, 2),
    )

    res = CliRunner().invoke(cli, ["history-backup", "--restore"])
    assert res.exit_code == 0, res.output
    assert "Restored 2 runs (4 rows)" in res.output
