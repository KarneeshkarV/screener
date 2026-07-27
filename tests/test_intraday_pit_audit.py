"""Point-in-time (no-lookahead) property audit for the intraday options engine.

The roadmap's Phase 5 PIT guarantee: for every fill/exit in an intraday run, a
decision at time T may only depend on data observed at timestamps ≤ T. We assert
this over seeded random price paths and several exit configs:

1. **Prefix truncation invariance** — for every snapshot T strictly before each
   trade's exit, re-running on chains truncated to ``as_of <= T`` must reproduce
   every decision with timestamp ≤ T (entry ts/premium, any exit at/before T).
2. **Timestamp membership + ordering** — every entry/exit timestamp is an actual
   snapshot time and entry_ts ≤ exit_ts.
3. **Price causality** — entry/exit fills equal the mark at *that* snapshot when
   the contract is present (carried-mark exits are recorded in details).
4. **Session flatness** — no position outlives the session (every entry has a
   matching exit; last trade of a session is closed).
5. **Short-leg structures** — the same properties hold for credit structures.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, time, timezone

from screener.options.intraday_backtest import (
    IntradayOptionsBacktestConfig,
    run_intraday_options_backtest,
)
from screener.options.models import OptionChain
from screener.options.position_backtest import _mark_price
from tests.test_intraday_options_backtest import (
    DAY,
    _call,
    _chain,
    _put,
    _StubProvider,
)

_SESSION_OPEN_UTC = datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc)
_SESSION_CLOSE_UTC = datetime(2026, 7, 8, 20, 0, tzinfo=timezone.utc)


def _session(seed: int, *, n_min: int = 6, n_max: int = 10) -> list[OptionChain]:
    """A random-walk long-call session: snapshots at 30-min steps from 13:30 UTC.

    Paths may end before the 16:00 ET close (data_end) or include a final bar at
    scheduled close (session_end) so both H8 paths are exercised.
    """
    rng = random.Random(seed)
    n = rng.randint(n_min, n_max)
    last = rng.uniform(5.0, 20.0)
    spot = 100.0
    chains: list[OptionChain] = []
    for i in range(n):
        ts = _SESSION_OPEN_UTC + timedelta(minutes=i * 30)
        if ts > _SESSION_CLOSE_UTC:
            break
        step = rng.uniform(-2.5, 2.5)
        last = max(0.5, last + step)
        spot = max(1.0, spot + step)
        chains.append(_chain(ts, _call(ts, last=round(last, 2)), spot=round(spot, 2)))
    # ~half the seeds pin a final bar at scheduled close for session_end coverage.
    if chains and rng.random() < 0.5:
        if chains[-1].as_of < _SESSION_CLOSE_UTC:
            last = max(0.5, last + rng.uniform(-1.0, 1.0))
            spot = max(1.0, spot + rng.uniform(-1.0, 1.0))
            chains.append(
                _chain(
                    _SESSION_CLOSE_UTC,
                    _call(_SESSION_CLOSE_UTC, last=round(last, 2)),
                    spot=round(spot, 2),
                )
            )
    return chains


def _short_put_session(seed: int) -> list[OptionChain]:
    """Credit short-put path for the short-leg structure case."""
    rng = random.Random(seed + 10_000)
    n = rng.randint(5, 8)
    last = rng.uniform(3.0, 12.0)
    spot = 100.0
    chains: list[OptionChain] = []
    for i in range(n):
        ts = _SESSION_OPEN_UTC + timedelta(minutes=i * 30)
        if ts > _SESSION_CLOSE_UTC:
            break
        step = rng.uniform(-1.5, 1.5)
        last = max(0.5, last + step)
        spot = max(1.0, spot + step)
        chains.append(_chain(ts, _put(ts, last=round(last, 2)), spot=round(spot, 2)))
    if chains and chains[-1].as_of < _SESSION_CLOSE_UTC:
        chains.append(
            _chain(
                _SESSION_CLOSE_UTC,
                _put(_SESSION_CLOSE_UTC, last=round(last, 2)),
                spot=round(spot, 2),
            )
        )
    return chains


def _key(trade: object) -> tuple:
    t = trade  # OptionPositionTrade
    return (
        t.entry_date,  # type: ignore[attr-defined]
        t.exit_date,  # type: ignore[attr-defined]
        t.exit_reason,  # type: ignore[attr-defined]
        t.legs[0].entry_price,  # type: ignore[attr-defined]
        t.legs[0].exit_price,  # type: ignore[attr-defined]
        round(t.pnl, 6),  # type: ignore[attr-defined]
        t.details["entry_ts"],  # type: ignore[attr-defined]
        t.details["exit_ts"],  # type: ignore[attr-defined]
    )


def _decision_key(trade: object) -> tuple:
    """Identity of a trade's entry decision (and exit if closed)."""
    t = trade  # OptionPositionTrade
    return (
        t.details["entry_ts"],  # type: ignore[attr-defined]
        t.legs[0].entry_price,  # type: ignore[attr-defined]
        t.details["exit_ts"],  # type: ignore[attr-defined]
        t.legs[0].exit_price,  # type: ignore[attr-defined]
        t.exit_reason,  # type: ignore[attr-defined]
        round(t.pnl, 6),  # type: ignore[attr-defined]
    )


_CONFIGS = [
    dict(),
    dict(target_pct=10.0),
    dict(stop_pct=15.0),
    dict(target_pct=8.0, stop_pct=8.0),
    dict(exit_time=time(11, 0)),  # 11:00 ET
    dict(entry_time=time(10, 30), target_pct=12.0),
]


def _assert_session_ends_flat(result: object) -> None:
    """Every trade is closed; no open risk after the last snapshot of a session."""
    trades = result.trades  # type: ignore[attr-defined]
    for trade in trades:
        assert trade.details.get("exit_ts"), trade
        assert trade.legs[0].exit_price is not None
        assert trade.exit_reason in {
            "target",
            "stop",
            "time",
            "session_end",
            "data_end",
        }


def _assert_prefix_invariance(
    cfg: IntradayOptionsBacktestConfig,
    chains: list[OptionChain],
    full_result: object,
) -> None:
    """For every T before an exit, full-run decisions ≤ T match when truncated to T.

    Truncation can force an extra ``data_end`` at the cut (no future bar) — that
    artifact is ignored. We only require that every *full-run* entry/exit whose
    timestamp is ≤ T is reproduced identically on the prefix.
    """
    trades = full_result.trades  # type: ignore[attr-defined]
    if not trades:
        return
    snapshot_times = sorted({c.as_of for c in chains})
    # Unique cut points: every snapshot strictly before any full-run exit.
    exit_times = {datetime.fromisoformat(tr.details["exit_ts"]) for tr in trades}
    cut_points = [t for t in snapshot_times if any(t < ex for ex in exit_times)]
    for t_cut in cut_points:
        truncated = [c for c in chains if c.as_of <= t_cut]
        if len(truncated) < 2:
            continue
        trunc_result = run_intraday_options_backtest(
            cfg, _StubProvider({("SPY", DAY): truncated})
        )
        trunc_by_key = {_decision_key(tr): tr for tr in trunc_result.trades}
        trunc_entries = {
            (tr.details["entry_ts"], tr.legs[0].entry_price)
            for tr in trunc_result.trades
        }

        # Full-run trades fully closed by T must appear byte-identical.
        for tr in trades:
            exit_dt = datetime.fromisoformat(tr.details["exit_ts"])
            if exit_dt > t_cut:
                continue
            key = _decision_key(tr)
            assert key in trunc_by_key, (
                f"closed trade missing after truncate to {t_cut.isoformat()}: {key}"
            )

        # Full-run entries at/before T (openable on the prefix) must match premium.
        for tr in trades:
            entry_ts = tr.details["entry_ts"]
            entry_dt = datetime.fromisoformat(entry_ts)
            if entry_dt > t_cut:
                continue
            # Engine refuses to open on the last recorded bar of a series.
            if entry_dt == truncated[-1].as_of:
                continue
            assert (entry_ts, tr.legs[0].entry_price) in trunc_entries, (
                f"entry missing after truncate to {t_cut.isoformat()}: "
                f"{entry_ts}@{tr.legs[0].entry_price}"
            )


def test_intraday_engine_has_no_lookahead():
    for seed in range(20):
        chains = _session(seed)
        if len(chains) < 2:
            continue
        snapshot_ts = {c.as_of.isoformat() for c in chains}
        mark_at = {c.as_of.isoformat(): _mark_price(c.contracts[0]) for c in chains}
        for overrides in _CONFIGS:
            cfg = IntradayOptionsBacktestConfig(
                tickers=("SPY",), start=DAY, end=DAY, market="us", **overrides
            )
            result = run_intraday_options_backtest(
                cfg, _StubProvider({("SPY", DAY): chains})
            )
            _assert_session_ends_flat(result)

            for trade in result.trades:
                entry_ts = trade.details["entry_ts"]
                exit_ts = trade.details["exit_ts"]

                # (2) timestamps are real snapshots and correctly ordered.
                assert entry_ts in snapshot_ts, (seed, overrides, entry_ts)
                assert exit_ts in snapshot_ts, (seed, overrides, exit_ts)
                assert entry_ts <= exit_ts, (seed, overrides)

                # (3) fills equal the mark at their own snapshot when observed.
                assert trade.legs[0].entry_price == mark_at[entry_ts]
                if not trade.details.get("carried_marks"):
                    assert trade.legs[0].exit_price == mark_at[exit_ts]

                # Classic exit-truncation check (still useful for completed trades).
                exit_dt = datetime.fromisoformat(exit_ts)
                truncated = [c for c in chains if c.as_of <= exit_dt]
                trunc_result = run_intraday_options_backtest(
                    cfg, _StubProvider({("SPY", DAY): truncated})
                )
                assert trunc_result.trades, (seed, overrides, "trade vanished")
                assert _key(trunc_result.trades[0]) == _key(trade), (seed, overrides)

            # (1) stronger prefix invariance for every T before each exit.
            _assert_prefix_invariance(cfg, chains, result)


def test_intraday_short_leg_structure_has_no_lookahead():
    """Short-put (credit) paths: same PIT properties + session ends flat."""
    for seed in range(12):
        chains = _short_put_session(seed)
        if len(chains) < 2:
            continue
        snapshot_ts = {c.as_of.isoformat() for c in chains}
        mark_at = {c.as_of.isoformat(): _mark_price(c.contracts[0]) for c in chains}
        for overrides in (
            dict(structure="short_put"),
            dict(structure="short_put", target_pct=10.0),
            dict(structure="short_put", stop_pct=20.0),
            dict(structure="short_put", exit_time=time(12, 0)),
        ):
            cfg = IntradayOptionsBacktestConfig(
                tickers=("SPY",),
                start=DAY,
                end=DAY,
                market="us",
                **overrides,
            )
            result = run_intraday_options_backtest(
                cfg, _StubProvider({("SPY", DAY): chains})
            )
            _assert_session_ends_flat(result)
            for trade in result.trades:
                assert trade.structure == "short_put"
                entry_ts = trade.details["entry_ts"]
                exit_ts = trade.details["exit_ts"]
                assert entry_ts in snapshot_ts
                assert exit_ts in snapshot_ts
                assert entry_ts <= exit_ts
                assert trade.legs[0].entry_price == mark_at[entry_ts]
                if not trade.details.get("carried_marks"):
                    assert trade.legs[0].exit_price == mark_at[exit_ts]
            _assert_prefix_invariance(cfg, chains, result)
