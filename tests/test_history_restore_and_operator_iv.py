"""Regressions for restore-time lock scope and operator-scan IV derivation."""

from __future__ import annotations

import sqlite3
import types
from datetime import date

import pandas as pd

from screener import history as history_mod
from screener import history_sync
from screener.operator import process


class _RemoteClient:
    """In-memory SQLite standing in for the libsql sync client.

    Records every statement and can run a callback while a remote row fetch is
    in flight, which is exactly the window where the old ``restore_history``
    held the local SQLite write lock.
    """

    def __init__(self, on_row_fetch=None) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.executed: list[str] = []
        self.row_fetches: list[tuple[str, str, str]] = []
        self.on_row_fetch = on_row_fetch

    def execute(self, stmt: str, args: list[object] | None = None):
        self.executed.append(stmt)
        is_select = stmt.lstrip().upper().startswith("SELECT")
        if is_select and history_sync.REMOTE_ROWS_TABLE in stmt and args:
            key = tuple(str(arg) for arg in args)
            self.row_fetches.append(key)
            if self.on_row_fetch is not None:
                self.on_row_fetch(self, key)
        cur = self.conn.execute(stmt, tuple(args) if args else ())
        rows = cur.fetchall() if is_select else []
        self.conn.commit()
        return types.SimpleNamespace(rows=rows)

    def batch(self, stmts):
        return [self.execute(sql, args) for sql, args in stmts]

    def close(self) -> None:
        pass


def _screen_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": ticker,
                "description": f"{ticker} Corp",
                "close": 10.0 + i,
                "change": 1.0,
                "volume": 1_000_000,
                "market_cap_basic": 1_000_000_000,
                "setup_score": 50.0 + i,
            }
            for i, ticker in enumerate(tickers)
        ]
    )


def _seed_remote(monkeypatch, tmp_path) -> _RemoteClient:
    """Back up two local runs into a remote stub, then point at a fresh DB."""
    source_db = tmp_path / "source.db"
    monkeypatch.setattr(history_mod, "DB_PATH", source_db)
    history_mod.save_run("us", "ema", 2, _screen_df(["AAA", "BBB"]))
    history_mod.save_run("india", "breakout", 1, _screen_df(["XXX"]))
    remote = _RemoteClient()
    history_sync.backup_history(remote)
    return remote


def test_restore_fetches_all_remote_rows_before_writing_locally(monkeypatch, tmp_path):
    remote = _seed_remote(monkeypatch, tmp_path)

    target_db = tmp_path / "target.db"
    monkeypatch.setattr(history_mod, "DB_PATH", target_db)

    real_connect = history_mod._connect
    # Snapshot how many remote row fetches had completed at each local connect.
    connects: list[int] = []

    def tracking_connect():
        connects.append(len(remote.row_fetches))
        return real_connect()

    monkeypatch.setattr(history_mod, "_connect", tracking_connect)

    def probe_concurrent_write(_client, _key) -> None:
        # A concurrent `screener screen` writer must get the lock while the
        # restore is still talking to Turso. The old implementation had an
        # uncommitted INSERT into `runs` and raised "database is locked" here.
        conn = sqlite3.connect(target_db, timeout=0.05)
        try:
            conn.execute(
                "INSERT INTO runs (run_ts, market, criteria, total_matches) "
                "VALUES (?, ?, ?, ?)",
                (
                    f"1999-12-31T00:00:0{len(remote.row_fetches)}+00:00",
                    "probe",
                    "probe",
                    0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    remote.on_row_fetch = probe_concurrent_write
    summary = history_sync.restore_history(remote)

    # Two remote row fetches, both completed before the write connection opened.
    assert len(remote.row_fetches) == 2
    assert connects == [0, 2]

    # The concurrent writes succeeded: restore never held the local write lock
    # while a remote fetch was outstanding.
    probe = sqlite3.connect(target_db)
    try:
        assert (
            probe.execute(
                "SELECT COUNT(*) FROM runs WHERE market = 'probe'"
            ).fetchone()[0]
            == 2
        )
    finally:
        probe.close()

    # Restored payload is identical to what was backed up.
    assert summary.runs_restored == 2
    assert summary.rows_restored == 3
    assert summary.local_runs == 4  # two restored runs + the two probe writes
    us = history_mod.list_runs(market="us")
    assert us.iloc[0]["total_matches"] == 2
    snapshot = history_mod.load_run(int(us.iloc[0]["id"]))
    assert snapshot is not None
    assert snapshot.tickers == ["AAA", "BBB"]
    assert history_mod.list_runs(market="india").iloc[0]["total_matches"] == 1

    # Everything is local now, so a second restore is a no-op.
    again = history_sync.restore_history(remote)
    assert again.runs_restored == 0


def test_restore_network_failure_leaves_local_db_untouched(monkeypatch, tmp_path):
    remote = _seed_remote(monkeypatch, tmp_path)
    target_db = tmp_path / "target.db"
    monkeypatch.setattr(history_mod, "DB_PATH", target_db)

    def fail_second_fetch(_client, key) -> None:
        if len(remote.row_fetches) == 2:
            raise ConnectionError("turso: connection reset")

    remote.on_row_fetch = fail_second_fetch

    try:
        history_sync.restore_history(remote)
    except ConnectionError:
        pass
    else:  # pragma: no cover - the failure is the point of this test
        raise AssertionError("restore should propagate the network failure")

    conn = sqlite3.connect(target_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM run_rows").fetchone()[0] == 0
    finally:
        conn.close()


def test_operator_scan_loads_chains_without_deriving_iv(monkeypatch):
    as_of = date(2024, 1, 5)
    trailing = [date(2024, 1, day) for day in [4, 3, 2, 1, 1, 4]]
    latest_calls: list[date] = []

    def fake_latest(d: date) -> date:
        latest_calls.append(d)
        if len(latest_calls) == 1:
            return as_of
        return trailing.pop(0)

    monkeypatch.setattr(process, "latest_trading_day", fake_latest)
    monkeypatch.setattr(
        process, "combined_universe", lambda d, mode: (["AAA"], {"AAA"})
    )
    monkeypatch.setattr(
        process,
        "fetch_cash_bhavcopy",
        lambda d: pd.DataFrame(
            {
                "SYMBOL": ["AAA"],
                "PREV_CLOSE": [100.0],
                "CLOSE_PRICE": [110.0],
                "AVG_PRICE": [108.0],
                "TTL_TRD_QNTY": [1000.0],
                "DELIV_QTY": [200.0],
                "DELIV_PER": [60.0],
            }
        ),
    )
    monkeypatch.setattr(
        process,
        "fetch_fo_bhavcopy",
        lambda d: pd.DataFrame(
            {
                "SYMBOL": ["AAA"],
                "EXPIRY": [pd.Timestamp("2024-01-25")],
                "OI": [100.0],
            }
        ),
    )
    monkeypatch.setattr(
        process,
        "fifty_two_week_hl",
        lambda symbols, d: pd.DataFrame(
            {"SYMBOL": list(symbols), "_52W_High": [120.0], "_52W_Low": [80.0]}
        ),
    )

    chain_calls: list[tuple[date, dict[str, object]]] = []

    def fake_chains(d: date, **kwargs: object):
        chain_calls.append((d, kwargs))
        return {}

    monkeypatch.setattr(process, "load_bhavcopy_chains", fake_chains)

    process.build_dataset(as_of, universe_mode="fo")

    # The OI confirmation never reads IV/greeks, so the expensive inversion
    # across the whole F&O bhavcopy must be skipped.
    assert chain_calls == [(as_of, {"derive_iv": False})]
