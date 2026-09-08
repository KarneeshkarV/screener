"""Persisted experiment trial register for optimizer searches.

Records attempted, rejected, error, and completed trials before top_n cutoff.
Works with or without a result-cache frozen input identity.

Trial uniqueness is ``(experiment_id, trial_key)`` where ``trial_key`` is the
complete tested config identity (logical strategy config, dates, runner,
metric, min_trades). Swept ``params_json`` alone is not enough: a shared
``experiment_id`` groups histories but must not erase distinct strategies that
happen to share the same swept parameter values.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np

from screener.backtester.models import BacktestConfig
from screener.backtester.optimization.cache import (
    config_fingerprint,
    dumps_strict_json,
    universe_membership_identity_from_config,
)

TrialStatus = Literal["attempted", "completed", "rejected", "error"]

_DEFAULT_DB = Path.home() / ".screener" / "optimizer_trials.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    trial_key TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    sharpe REAL,
    metrics_json TEXT,
    error TEXT,
    trade_count INTEGER NOT NULL DEFAULT 0,
    created_ts TEXT NOT NULL,
    UNIQUE(experiment_id, trial_key)
);
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
"""


def default_trial_db_path() -> Path:
    override = os.environ.get("SCREENER_OPTIMIZER_TRIALS_DB")
    if override:
        return Path(override)
    return _DEFAULT_DB


def resolve_trial_db_path(path: Path | str | None) -> Path:
    if path is None:
        return default_trial_db_path()
    return Path(path)


def default_experiment_id(
    cfg: BacktestConfig,
    parameter_grid: dict[str, list[Any]],
    *,
    runner: str,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> str:
    """Default search-family id from base config, grid keys, universe, window.

    This default **fragments** when base config fields or grid key sets change.
    For iterative searches that must count rejected families together, pass an
    explicit ``experiment_id`` and reuse the same ``trial_db_path``.
    """
    payload = {
        "config": config_fingerprint(cfg),
        "universe_membership_identity": universe_membership_identity_from_config(cfg),
        "parameter_keys": sorted(parameter_grid),
        "runner": runner,
        "start_date": None if start_date is None else start_date.isoformat(),
        "end_date": None if end_date is None else end_date.isoformat(),
        "metric": metric,
        "min_trades": min_trades,
    }
    raw = dumps_strict_json(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def trial_config_identity(
    tested_cfg: BacktestConfig,
    *,
    runner: str,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> str:
    """Complete tested-config identity for the trial unique key.

    Includes dates, runner, metric, min_trades, and the full logical strategy
    config (base fields with swept params applied). Excludes price snapshot and
    code fingerprint so repeating the same experiment does not double-count.
    """
    payload = {
        "config": config_fingerprint(tested_cfg),
        "universe_membership_identity": universe_membership_identity_from_config(
            tested_cfg
        ),
        "runner": runner,
        "start_date": None if start_date is None else start_date.isoformat(),
        "end_date": None if end_date is None else end_date.isoformat(),
        "metric": metric,
        "min_trades": min_trades,
    }
    raw = dumps_strict_json(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialSearchStats:
    experiment_id: str
    n_trials_nominal: int
    n_trials_scored: int
    n_trials_effective: int | None
    sr_trial_std_annual: float | None
    sharpes: tuple[float, ...]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create trials table; recreate if a pre-trial_key schema is detected."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trials'"
    ).fetchall()
    if rows:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)").fetchall()}
        if "trial_key" not in cols:
            # Pre-review schema used UNIQUE(experiment_id, params_json) and could
            # collide across distinct strategies. Drop and recreate (new feature).
            conn.execute("DROP TABLE trials")
            conn.commit()
    conn.executescript(_SCHEMA)


def _params_key(params: dict[str, Any]) -> str:
    return dumps_strict_json(params)


def _finite_sharpe(metrics: dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    value = metrics.get("sharpe")
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def record_trial(
    experiment_id: str,
    params: dict[str, Any],
    *,
    status: TrialStatus,
    score: float | None,
    metrics: dict[str, float] | None,
    trade_count: int,
    trial_key: str,
    error: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Upsert one trial row keyed by ``(experiment_id, trial_key)``.

    Missing Sharpe stays SQL NULL (never coerced to 0). Non-finite scores other
    than ``-inf`` are stored as NULL; ``-inf`` is kept for rejected gates.
    """
    path = resolve_trial_db_path(db_path)
    sharpe = _finite_sharpe(metrics)
    score_value: float | None
    if score is None:
        score_value = None
    else:
        number = float(score)
        if math.isnan(number):
            score_value = None
        elif math.isinf(number) and number < 0:
            score_value = float("-inf")
        elif math.isfinite(number):
            score_value = number
        else:
            score_value = None
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO trials (
                experiment_id, trial_key, params_json, status, score, sharpe,
                metrics_json, error, trade_count, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(experiment_id, trial_key) DO UPDATE SET
                params_json=excluded.params_json,
                status=excluded.status,
                score=excluded.score,
                sharpe=excluded.sharpe,
                metrics_json=excluded.metrics_json,
                error=excluded.error,
                trade_count=excluded.trade_count,
                created_ts=excluded.created_ts
            """,
            (
                experiment_id,
                trial_key,
                _params_key(params),
                status,
                score_value,
                sharpe,
                None if metrics is None else dumps_strict_json(metrics),
                error,
                int(trade_count),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_trial_search_stats(
    experiment_id: str,
    *,
    db_path: Path | str | None = None,
    n_trials_effective: int | None = None,
) -> TrialSearchStats:
    """Aggregate trial counts and measured annualized Sharpe dispersion.

    ``n_trials_effective`` is only the caller's explicit correlated-trial
    estimate and must be an ``int`` in ``1 .. n_trials_nominal`` when provided.
    When omitted, it stays ``None`` (report nominal; do not claim independence).
    """
    if n_trials_effective is not None:
        if (
            not isinstance(n_trials_effective, int)
            or isinstance(n_trials_effective, bool)
            or n_trials_effective < 1
        ):
            raise ValueError("n_trials_effective must be an int >= 1 when provided")
    path = resolve_trial_db_path(db_path)
    rows: list[tuple[Any, Any]]
    if not path.exists():
        rows = []
    else:
        conn = _connect(path)
        try:
            rows = conn.execute(
                "SELECT status, sharpe FROM trials WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
        finally:
            conn.close()
    sharpes: list[float] = []
    for _status, sharpe in rows:
        if sharpe is None:
            continue
        value = float(sharpe)
        if math.isfinite(value):
            sharpes.append(value)
    nominal = len(rows)
    if n_trials_effective is not None:
        if nominal < 1 or n_trials_effective > nominal:
            raise ValueError(
                "n_trials_effective must be between 1 and n_trials_nominal inclusive"
            )
    std: float | None
    if len(sharpes) >= 2:
        std = float(np.std(sharpes, ddof=1))
        if not math.isfinite(std):
            std = None
    else:
        std = None
    return TrialSearchStats(
        experiment_id=experiment_id,
        n_trials_nominal=nominal,
        n_trials_scored=len(sharpes),
        n_trials_effective=n_trials_effective,
        sr_trial_std_annual=std,
        sharpes=tuple(sharpes),
    )


def classify_trial_status(
    *,
    error: str | None,
    score: float,
    trade_count: int,
    min_trades: int,
) -> TrialStatus:
    if error:
        return "error"
    if trade_count < min_trades or not math.isfinite(score):
        return "rejected"
    return "completed"
