"""Intraday live-loop screening — fully offline via injected seams.

No test touches the network or a real database: the clock (``now_fn``), pacing
(``sleep_fn``), and the trailing bar refresh (``refresh_bars``) are monkeypatched,
the scanner runs against a tmp bar store, and history persistence goes to a tmp
``history.db``. Covers session gating, per-pass persistence with intraday
timestamps, the entrant/exit diff across passes, and the max-passes bound.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import screener.history as history_mod
import screener.screen_live as live
from screener.backtester.bar_store import save_bars
from screener.screen_live import LiveRequest, in_session, run_screen_live

US_TZ = "America/New_York"


def _two_session_index(bars_per: int = 30) -> pd.DatetimeIndex:
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-07-20 14:30:00")  # Monday 09:30 ET
    for _ in range(2):
        stamps.extend(day + pd.Timedelta(minutes=b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _rising(index: pd.DatetimeIndex, *, start: float, step: float, volume: float):
    closes = start + step * np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(len(index), volume),
        },
        index=index,
    )


def _write(root: Path, symbol: str) -> None:
    index = _two_session_index(bars_per=40)
    save_bars(
        symbol,
        _rising(index, start=100.0, step=0.3, volume=50_000.0),
        market="us",
        interval="1m",
        root=root,
    )


# --------------------------------------------------------------------------- #
# Session gating
# --------------------------------------------------------------------------- #
def test_in_session_us_hours() -> None:
    # 2026-07-20 is a Monday. 14:00 UTC = 10:00 ET (open); 21:00 UTC = 17:00 ET.
    assert in_session(datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc), "us") is True
    assert in_session(datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc), "us") is False


def test_in_session_weekend() -> None:
    # 2026-07-25 is a Saturday.
    assert in_session(datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc), "us") is False


def test_in_session_india_hours() -> None:
    # 04:30 UTC = 10:00 IST (open); 12:00 UTC = 17:30 IST (closed).
    assert (
        in_session(datetime(2026, 7, 20, 4, 30, tzinfo=timezone.utc), "india") is True
    )
    assert (
        in_session(datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc), "india") is False
    )


# --------------------------------------------------------------------------- #
# Live loop
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_history(monkeypatch, tmp_path):
    monkeypatch.setattr(history_mod, "DB_PATH", tmp_path / "history.db")


def test_run_screen_live_persists_passes_and_diffs(
    monkeypatch, tmp_path, tmp_history
) -> None:
    _write(tmp_path, "AAA")  # present from the start

    # Clock advances one minute per read so passes get distinct run rows.
    ticks = iter(
        [
            datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 20, 14, 1, tzinfo=timezone.utc),
        ]
    )
    monkeypatch.setattr(live, "now_fn", lambda: next(ticks))

    sleeps: list[float] = []
    monkeypatch.setattr(live, "sleep_fn", lambda seconds: sleeps.append(seconds))

    refresh_calls: list[int] = []

    def fake_refresh(market: str, days: int, *, root=None) -> None:
        refresh_calls.append(days)
        if len(refresh_calls) == 2:  # a new entrant appears before pass 2
            _write(tmp_path, "BBB")

    monkeypatch.setattr(live, "refresh_bars", fake_refresh)

    request = LiveRequest(
        market="us",
        criteria_names=("ema",),
        interval="5m",
        order_by="volume",
        every_seconds=300.0,
        max_passes=2,
        refresh_days=1,
        bar_store_root=tmp_path,
    )
    session = run_screen_live(request)

    assert refresh_calls == [1, 1]  # refresh before every pass
    assert sleeps == [300.0]  # one sleep between the two passes, none after the last
    assert len(session.passes) == 2

    first, second = session.passes
    assert first.first_pass is True
    assert first.df["name"].tolist() == ["AAA"]
    assert first.added == () and first.removed == ()

    assert second.first_pass is False
    assert set(second.df["name"]) == {"AAA", "BBB"}
    assert second.added == ("BBB",)
    assert second.removed == ()

    # Both passes persisted to the shared history schema with intraday timestamps.
    runs = history_mod.list_runs(market="us", criteria="ema")
    assert len(runs) == 2
    assert runs["run_ts"].tolist() == [
        "2026-07-20T14:01:00+00:00",
        "2026-07-20T14:00:00+00:00",
    ]


def test_run_screen_live_stops_when_market_closed(
    monkeypatch, tmp_path, tmp_history
) -> None:
    _write(tmp_path, "AAA")
    # After hours (21:00 UTC = 17:00 ET) → gate fails immediately, no passes.
    monkeypatch.setattr(
        live, "now_fn", lambda: datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("refresh must not run when the market is closed")

    monkeypatch.setattr(live, "refresh_bars", unexpected)

    request = LiveRequest(
        market="us",
        criteria_names=("ema",),
        max_passes=3,
        bar_store_root=tmp_path,
    )
    session = run_screen_live(request)
    assert session.passes == []
