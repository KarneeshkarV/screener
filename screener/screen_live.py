"""Intraday live-loop screening on top of the local bar-store scanner.

``screener screen live`` re-runs the local scanner every ``--every`` window
during the trading session, persisting each pass to the shared
``~/.screener/history.db`` through the *existing* :func:`history.save_run`
writer (no schema change — each pass is just another run row with an intraday
timestamp), then diffs against the previous pass to emit only new entrants and
exits. ``backtest-historical --from-run`` and the Turso backup keep working
because the on-disk schema is untouched.

Every external effect is a module-level seam so the loop is fully testable
offline: ``now_fn`` (clock), ``sleep_fn`` (pacing), ``refresh_bars`` (the
trailing ``bars record --days`` refresh that touches the network),
``local_scan`` (the scanner), and the ``save_run`` / ``previous_run`` / ``diff``
history collaborators. Tests monkeypatch these — the seam pattern
:mod:`screener.screen_workflow` already documents — so no test hits the network
or a real database. Session gating uses ``markets.get_market(market).timezone``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from screener.criteria import resolve_criteria
from screener.history import diff, previous_run, save_run
from screener.local_scanner import local_scan
from screener.markets import get_market

# Exchange regular-session windows (local time) used only to gate the live loop.
# The market timezone comes from ``markets.get_market(market).timezone``.
SESSION_HOURS: dict[str, tuple[time, time]] = {
    "us": (time(9, 30), time(16, 0)),
    "india": (time(9, 15), time(15, 30)),
}


# Injectable seams (monkeypatched in tests; production defaults below).
def now_fn() -> datetime:
    return datetime.now(timezone.utc)


sleep_fn: Callable[[float], None] = __import__("time").sleep


def refresh_bars(market: str, days: int, *, root: Path | None = None) -> None:
    """Append the trailing ``days`` window of 1m bars for the market universe.

    Mirrors ``screener bars record --days N`` so a live pass sees near-live bars.
    Isolated behind this seam so tests inject a no-op and never hit the network.
    """
    from screener.backtester.bar_store import append_bars
    from screener.backtester.data import build_price_fetcher, tv_to_yf
    from screener.universes import load_current_universe

    from datetime import date, timedelta

    name = get_market(market).default_universe
    symbols = list(load_current_universe(name).symbols)
    yf_symbols = sorted({tv_to_yf(symbol, market) for symbol in symbols})
    end = date.today()
    start = end - timedelta(days=days)
    fetcher = build_price_fetcher(interval="1m", market=market)
    frames = fetcher.fetch(yf_symbols, start, end)
    for symbol in yf_symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        append_bars(symbol, frame, market=market, interval="1m", root=root)


def in_session(moment: datetime, market: str) -> bool:
    """True when ``moment`` (any tz) falls inside the market's regular session."""
    tz = get_market(market).timezone
    local = moment.astimezone(ZoneInfo(tz))
    if local.weekday() >= 5:  # Saturday/Sunday
        return False
    open_time, close_time = SESSION_HOURS.get(market, (time(0, 0), time(23, 59)))
    return open_time <= local.time() <= close_time


@dataclass(frozen=True)
class LiveRequest:
    market: str
    criteria_names: tuple[str, ...]
    interval: str = "5m"
    limit: int = 50
    order_by: str = "volume"
    every_seconds: float = 300.0
    # 0 = run until the session closes; a positive cap bounds test/manual runs.
    max_passes: int = 0
    refresh_days: int = 1
    bar_store_root: Path | None = None


@dataclass(frozen=True)
class LivePass:
    """One live evaluation: its persisted run plus the diff vs. the prior pass."""

    run_id: int
    run_ts: str
    total: int
    df: pd.DataFrame
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    first_pass: bool = False


@dataclass
class LiveSession:
    """Accumulated passes from one live-loop run."""

    passes: list[LivePass] = field(default_factory=list)


def run_screen_live(request: LiveRequest) -> LiveSession:
    """Run the intraday live loop, returning every pass it evaluated.

    Each iteration: session-gates on the current clock, refreshes the trailing
    bar window, scans locally, persists the pass to history.db, and diffs against
    the previous pass. Stops when the session closes or ``max_passes`` is hit.
    """
    selection = resolve_criteria(request.criteria_names)
    session = LiveSession()
    count = 0
    while True:
        # Session gate / loop pacing clock — taken before the (possibly slow)
        # refresh so we still stop when the open session has already closed.
        moment = now_fn()
        if not in_session(moment, request.market):
            break

        refresh_bars(request.market, request.refresh_days, root=request.bar_store_root)
        total, df = local_scan(
            market=request.market,
            filters=selection.filters,
            interval=request.interval,
            limit=request.limit,
            order_by=request.order_by,
            root=request.bar_store_root,
        )
        # Persist run_ts *after* refresh+scan so it is >= every bar used in the
        # pass (refresh can take minutes and pull bars stamped after ``moment``).
        run_ts_moment = now_fn()
        run_ts = run_ts_moment.isoformat(timespec="seconds")
        run_id = save_run(
            request.market,
            selection.label,
            total,
            df,
            run_ts=run_ts,
        )
        prev = previous_run(request.market, selection.label, run_id)
        if prev is None:
            added: list[str] = []
            removed: list[str] = []
            first_pass = True
        else:
            added, removed = diff(df, prev)
            first_pass = False

        session.passes.append(
            LivePass(
                run_id=run_id,
                run_ts=run_ts,
                total=total,
                df=df,
                added=tuple(added),
                removed=tuple(removed),
                first_pass=first_pass,
            )
        )
        count += 1
        if request.max_passes and count >= request.max_passes:
            break
        sleep_fn(request.every_seconds)
    return session


__all__ = [
    "LivePass",
    "LiveRequest",
    "LiveSession",
    "in_session",
    "refresh_bars",
    "run_screen_live",
]
