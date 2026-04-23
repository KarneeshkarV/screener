"""Unit tests for screener/history.py: diff, list_runs, prune."""
from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Reimport history.py with DB_PATH pointing at a temp directory so each
    test gets a clean SQLite file."""
    from screener import history as _history
    monkeypatch.setattr(_history, "DB_PATH", tmp_path / "history.db")
    # Force _connect() to pick up the new path
    return _history


def _df(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": n,
                "description": n,
                "close": 10.0,
                "change": 1.0,
                "volume": 1_000.0,
                "market_cap_basic": 1e9,
                "setup_score": 50.0,
            }
            for n in names
        ]
    )


def test_diff_empty_cases(history):
    added, removed = history.diff(pd.DataFrame(), pd.DataFrame())
    assert added == [] and removed == []

    cur = _df(["AAA", "BBB"])
    added, removed = history.diff(cur, pd.DataFrame())
    assert added == ["AAA", "BBB"] and removed == []


def test_diff_sorts_and_dedups(history):
    cur = _df(["CCC", "AAA", "DDD"])
    prev = pd.DataFrame({"ticker": ["AAA", "BBB", "DDD"]})
    added, removed = history.diff(cur, prev)
    assert added == ["CCC"]
    assert removed == ["BBB"]


def test_list_runs_and_prune(history, monkeypatch):
    # seed 5 runs for (us, ema) and 2 for (india, ema). Stub datetime.now
    # so each save_run gets a distinct timestamp (the table has a UNIQUE
    # (run_ts, market, criteria) constraint and the test inserts faster
    # than 1-second wallclock resolution).
    from datetime import datetime, timedelta
    base = datetime(2026, 1, 1, tzinfo=UTC)
    counter = {"i": 0}

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            counter["i"] += 1
            return base + timedelta(seconds=counter["i"])

    monkeypatch.setattr(history, "datetime", _FakeDatetime)

    for i in range(5):
        history.save_run("us", "ema", 100 + i, _df([f"US{i}"]))
    for i in range(2):
        history.save_run("india", "ema", 10 + i, _df([f"IN{i}"]))

    rows = history.list_runs()
    assert len(rows) == 7
    assert list(rows["id"]) == sorted(rows["id"], reverse=True)

    only_us = history.list_runs(market="us")
    assert len(only_us) == 5
    assert (only_us["market"] == "us").all()

    deleted = history.prune(keep_per_key=2)
    # kept 2 of 5 US (deleted 3), kept 2 of 2 India (deleted 0) → 3 total
    assert deleted == 3
    surviving = history.list_runs(limit=100)
    assert len(surviving) == 4
    us_surv = surviving[surviving["market"] == "us"]
    # prune keeps the most recent per key — should be the last two inserted
    assert sorted(us_surv["total_matches"]) == [103, 104]


def test_prune_zero_keeps_nothing(history):
    history.save_run("us", "ema", 1, _df(["X"]))
    assert history.prune(keep_per_key=0) == 1
    assert history.list_runs().empty
