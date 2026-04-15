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

CREATE TABLE IF NOT EXISTS backtests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope         TEXT NOT NULL,
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL,
    hold_days     INTEGER NOT NULL,
    benchmark     TEXT NOT NULL,
    total_return  REAL,
    cagr          REAL,
    sharpe        REAL,
    max_drawdown  REAL,
    hit_rate      REAL,
    alpha         REAL,
    beta          REAL,
    computed_ts   TEXT NOT NULL,
    UNIQUE(run_id, scope, hold_days)
);

CREATE TABLE IF NOT EXISTS backtest_tickers (
    backtest_id   INTEGER NOT NULL REFERENCES backtests(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    entry_close   REAL,
    exit_close    REAL,
    return_pct    REAL,
    trading_days  INTEGER,
    PRIMARY KEY (backtest_id, ticker)
);
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


def load_last_run(
    market: Optional[str] = None, criteria: Optional[str] = None
) -> Optional[dict]:
    """Return the most recent run matching filters, with its tickers. None if no run exists."""
    conn = _connect()
    try:
        where = []
        params: list = []
        if market:
            where.append("market = ?")
            params.append(market)
        if criteria:
            where.append("criteria = ?")
            params.append(criteria)
        sql = "SELECT id, run_ts, market, criteria, total_matches FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT 1"

        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        run_id, run_ts, run_market, run_criteria, total = row

        rows = pd.read_sql_query(
            "SELECT ticker, name, close, setup_score, rank FROM run_rows "
            "WHERE run_id = ? ORDER BY rank",
            conn,
            params=(run_id,),
        )
        return {
            "id": run_id,
            "run_ts": run_ts,
            "market": run_market,
            "criteria": run_criteria,
            "total_matches": total,
            "rows": rows,
        }
    finally:
        conn.close()


def save_backtest(
    run_id: int,
    scope: str,
    start_date: str,
    end_date: str,
    hold_days: int,
    benchmark: str,
    summary: dict,
    per_ticker: pd.DataFrame,
) -> int:
    conn = _connect()
    try:
        computed_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = conn.execute(
            """
            INSERT OR REPLACE INTO backtests
                (run_id, scope, start_date, end_date, hold_days, benchmark,
                 total_return, cagr, sharpe, max_drawdown, hit_rate, alpha, beta, computed_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scope,
                start_date,
                end_date,
                hold_days,
                benchmark,
                _to_float(summary.get("total_return")),
                _to_float(summary.get("cagr")),
                _to_float(summary.get("sharpe")),
                _to_float(summary.get("max_drawdown")),
                _to_float(summary.get("hit_rate")),
                _to_float(summary.get("alpha")),
                _to_float(summary.get("beta")),
                computed_ts,
            ),
        )
        backtest_id = cur.lastrowid

        conn.execute("DELETE FROM backtest_tickers WHERE backtest_id = ?", (backtest_id,))
        rows = []
        for _, r in per_ticker.iterrows():
            rows.append(
                (
                    backtest_id,
                    str(r.get("ticker")),
                    _to_float(r.get("entry_close")),
                    _to_float(r.get("exit_close")),
                    _to_float(r.get("return_pct")),
                    int(r["trading_days"]) if pd.notna(r.get("trading_days")) else 0,
                )
            )
        if rows:
            conn.executemany(
                """
                INSERT INTO backtest_tickers
                    (backtest_id, ticker, entry_close, exit_close, return_pct, trading_days)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
        return backtest_id
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
