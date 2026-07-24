import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


_DEFAULT_DB_PATH = Path.home() / ".screener" / "history.db"
# Overridable via SCREENER_HISTORY_DB; tests may also monkeypatch this directly.
DB_PATH = Path(os.environ.get("SCREENER_HISTORY_DB") or _DEFAULT_DB_PATH)

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
    # foreign_keys must be enabled per-connection for run_rows' ON DELETE CASCADE
    # to fire; WAL + NORMAL trade a little durability for far better concurrency.
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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


def save_run(
    market: str,
    criteria: str,
    total: int,
    df: pd.DataFrame,
    run_ts: Optional[str] = None,
) -> int:
    # ``run_ts`` defaults to now (byte-identical for the daily screen path); the
    # intraday live loop passes each pass's own timestamp so successive passes
    # land on distinct run rows instead of colliding within the same second.
    if run_ts is None:
        run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = _connect()
    try:
        # Two runs landing in the same second collide on UNIQUE(run_ts, market,
        # criteria); upsert so the later one wins and reuses the existing id.
        conn.execute(
            """
            INSERT INTO runs (run_ts, market, criteria, total_matches)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_ts, market, criteria)
            DO UPDATE SET total_matches = excluded.total_matches
            """,
            (run_ts, market, criteria, int(total)),
        )
        id_row = conn.execute(
            "SELECT id FROM runs WHERE run_ts = ? AND market = ? AND criteria = ?",
            (run_ts, market, criteria),
        ).fetchone()
        if id_row is None:  # pragma: no cover - the upsert just wrote this row
            raise RuntimeError("upsert into runs did not yield a rowid")
        run_id = int(id_row[0])
        # Replace any rows from a prior run that shared this natural key.
        conn.execute("DELETE FROM run_rows WHERE run_id = ?", (run_id,))

        rows = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            ticker = str(row.get("name") or "").strip()
            if not ticker:
                continue
            rows.append(
                (
                    run_id,
                    ticker,
                    str(row["description"])
                    if row.get("description") is not None
                    and not pd.isna(row.get("description"))
                    else None,
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


@dataclass(frozen=True)
class RunSnapshot:
    """A persisted screen run: its metadata plus the ranked rows that were shown."""

    run_id: int
    run_ts: str
    market: str
    criteria: str
    total_matches: int
    rows: pd.DataFrame

    @property
    def run_date(self) -> date:
        return datetime.fromisoformat(self.run_ts).date()

    @property
    def tickers(self) -> list[str]:
        return [str(t) for t in self.rows["ticker"].dropna().tolist()]


def load_run(run_id: int) -> Optional[RunSnapshot]:
    """Load one persisted run with its ranked rows, or ``None`` if absent."""
    conn = _connect()
    try:
        meta = conn.execute(
            "SELECT id, run_ts, market, criteria, total_matches FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if meta is None:
            return None
        rows = pd.read_sql_query(
            "SELECT ticker, name, close, change, volume, market_cap, setup_score, rank "
            "FROM run_rows WHERE run_id = ? ORDER BY rank",
            conn,
            params=(run_id,),
        )
        return RunSnapshot(
            run_id=int(meta[0]),
            run_ts=str(meta[1]),
            market=str(meta[2]),
            criteria=str(meta[3]),
            total_matches=int(meta[4]),
            rows=rows,
        )
    finally:
        conn.close()


def find_run(
    market: str, criteria: str, on_or_before: Optional[date] = None
) -> Optional[int]:
    """Return the id of the most recent run for ``market``/``criteria``.

    With ``on_or_before``, only runs whose date is on or before it qualify —
    the lookup used to replay "the screen as of N days ago".
    """
    conn = _connect()
    try:
        if on_or_before is None:
            row = conn.execute(
                "SELECT id FROM runs WHERE market = ? AND criteria = ? "
                "ORDER BY id DESC LIMIT 1",
                (market, criteria),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM runs WHERE market = ? AND criteria = ? "
                "AND substr(run_ts, 1, 10) <= ? ORDER BY id DESC LIMIT 1",
                (market, criteria, on_or_before.isoformat()),
            ).fetchone()
        return int(row[0]) if row is not None else None
    finally:
        conn.close()


def list_runs(
    market: Optional[str] = None,
    criteria: Optional[str] = None,
    limit: int = 20,
) -> pd.DataFrame:
    """List persisted runs (newest first) with the count of saved rows."""
    where = []
    params: list[str | int] = []
    if market:
        where.append("r.market = ?")
        params.append(market)
    if criteria:
        where.append("r.criteria = ?")
        params.append(criteria)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(int(limit))
    conn = _connect()
    try:
        return pd.read_sql_query(
            f"""
            SELECT r.id, r.run_ts, r.market, r.criteria, r.total_matches,
                   COUNT(rr.ticker) AS saved_rows
            FROM runs r
            LEFT JOIN run_rows rr ON rr.run_id = r.id
            {where_sql}
            GROUP BY r.id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()


def resolve_replay_run(spec: str, min_age_days: int = 0) -> RunSnapshot:
    """Resolve a ``--from-run`` spec to a :class:`RunSnapshot`.

    ``spec`` is either a numeric run id or ``MARKET:CRITERIA``, which picks the
    most recent run at least ``min_age_days`` calendar days old. Raises
    ``ValueError`` with a user-facing message when nothing matches.
    """
    spec = spec.strip()
    if spec.isdigit():
        snap = load_run(int(spec))
        if snap is None:
            raise ValueError(f"no screen run with id {spec} in {DB_PATH}")
    else:
        market, sep, criteria = spec.partition(":")
        if not sep or not market or not criteria:
            raise ValueError(
                f"invalid --from-run {spec!r}: expected a run id or MARKET:CRITERIA "
                "(e.g. 42 or india:ema)"
            )
        cutoff = (
            date.today() - timedelta(days=int(min_age_days))
            if min_age_days > 0
            else None
        )
        run_id = find_run(market, criteria, on_or_before=cutoff)
        if run_id is None:
            aged = f" at least {min_age_days} days old" if min_age_days > 0 else ""
            raise ValueError(
                f"no persisted {market}/{criteria} screen run{aged} in {DB_PATH}; "
                "run `screener screen` first or check `screener history`"
            )
        snap = load_run(run_id)
        assert snap is not None  # find_run just returned this id
    if snap.rows.empty:
        raise ValueError(f"screen run #{snap.run_id} has no saved rows to replay")
    return snap


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


def diff(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[list[str], list[str]]:
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
