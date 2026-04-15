import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


DB_PATH = Path.home() / ".screener" / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts        TEXT NOT NULL,
    market        TEXT NOT NULL,
    criteria      TEXT NOT NULL,
    total_matches INTEGER NOT NULL,
    UNIQUE(run_ts, market, criteria)
);

CREATE TABLE IF NOT EXISTS run_rows (
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker      TEXT NOT NULL,
    name        TEXT,
    close       REAL,
    change      REAL,
    volume      REAL,
    market_cap  REAL,
    setup_score REAL,
    rank        INTEGER NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_runs_key ON runs(market, criteria, run_ts DESC);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def save_run(market: str, criteria: str, total: int, df: pd.DataFrame) -> int:
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO runs (run_ts, market, criteria, total_matches) VALUES (?, ?, ?, ?)",
            (run_ts, market, criteria, int(total)),
        )
        run_id = cur.lastrowid

        rows = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            ticker = str(row.get("name") or "").strip()
            if not ticker:
                continue
            rows.append(
                (
                    run_id,
                    ticker,
                    str(row["description"]) if row.get("description") is not None and not pd.isna(row.get("description")) else None,
                    _to_float(row.get("close")),
                    _to_float(row.get("change")),
                    _to_float(row.get("volume")),
                    _to_float(row.get("market_cap_basic")),
                    _to_float(row.get("setup_score")),
                    rank,
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO run_rows
                    (run_id, ticker, name, close, change, volume, market_cap, setup_score, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
        return run_id
    finally:
        conn.close()


def previous_run(market: str, criteria: str, before_id: int) -> Optional[pd.DataFrame]:
    conn = _connect()
    try:
        prev = conn.execute(
            """
            SELECT id FROM runs
            WHERE market = ? AND criteria = ? AND id < ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (market, criteria, before_id),
        ).fetchone()

        if prev is None:
            return None

        prev_id = prev[0]
        return pd.read_sql_query(
            "SELECT ticker, name, close, change, volume, market_cap, setup_score, rank "
            "FROM run_rows WHERE run_id = ? ORDER BY rank",
            conn,
            params=(prev_id,),
        )
    finally:
        conn.close()


def diff(
    current: pd.DataFrame, previous: pd.DataFrame
) -> tuple[list[str], list[str]]:
    if current is None or current.empty:
        current_set: set[str] = set()
    else:
        current_set = {str(t) for t in current["name"].dropna().tolist()}

    if previous is None or previous.empty:
        previous_set: set[str] = set()
    else:
        previous_set = {str(t) for t in previous["ticker"].dropna().tolist()}

    added = sorted(current_set - previous_set)
    removed = sorted(previous_set - current_set)
    return added, removed
