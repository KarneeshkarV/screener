"""Point-in-time (no-lookahead) property audit for the intraday options engine.

The roadmap's Phase 5 PIT guarantee: for every fill/exit in an intraday run, a
decision at time T may only depend on data observed at timestamps ≤ T. We assert
this three ways over many seeded random price paths and several exit configs:

1. **Truncation invariance** — the strongest lookahead check. Re-running on the
   session *truncated to snapshots ≤ the trade's exit timestamp* must reproduce
   the identical trade. If any future snapshot had leaked into an entry/exit
   decision, deleting the future would change the result.
2. **Timestamp membership + ordering** — every entry/exit timestamp is an actual
   snapshot time and entry_ts ≤ exit_ts.
3. **Price causality** — the entry/exit fill equals the mark at *that* snapshot,
   never a later one.
"""

from __future__ import annotations

import random
from datetime import datetime, time, timezone

from screener.options.intraday_backtest import (
    IntradayOptionsBacktestConfig,
    run_intraday_options_backtest,
)
from screener.options.models import OptionChain
from screener.options.position_backtest import _mark_price
from tests.test_intraday_options_backtest import DAY, _call, _chain, _StubProvider


def _session(seed: int) -> list[OptionChain]:
    """A random-walk session: 6-10 snapshots at 30-min steps from 13:30 UTC."""
    rng = random.Random(seed)
    n = rng.randint(6, 10)
    last = rng.uniform(5.0, 20.0)
    spot = 100.0
    chains: list[OptionChain] = []
    for i in range(n):
        minutes = 90 + i * 30  # 13:30, 14:00, ... UTC
        ts = datetime(2026, 7, 8, 13 + minutes // 60, minutes % 60, tzinfo=timezone.utc)
        step = rng.uniform(-2.5, 2.5)
        last = max(0.5, last + step)
        spot = max(1.0, spot + step)
        chains.append(_chain(ts, _call(ts, last=round(last, 2)), spot=round(spot, 2)))
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


_CONFIGS = [
    dict(),
    dict(target_pct=10.0),
    dict(stop_pct=15.0),
    dict(target_pct=8.0, stop_pct=8.0),
    dict(exit_time=time(11, 0)),  # 11:00 ET
    dict(entry_time=time(10, 30), target_pct=12.0),
]


def test_intraday_engine_has_no_lookahead():
    for seed in range(60):
        chains = _session(seed)
        snapshot_ts = {c.as_of.isoformat() for c in chains}
        # The mark the engine computes at each snapshot (bid/ask mid); a legacy
        # long-call fill equals this mark with zero slippage.
        mark_at = {c.as_of.isoformat(): _mark_price(c.contracts[0]) for c in chains}
        for overrides in _CONFIGS:
            cfg = IntradayOptionsBacktestConfig(
                tickers=("SPY",), start=DAY, end=DAY, market="us", **overrides
            )
            result = run_intraday_options_backtest(
                cfg, _StubProvider({("SPY", DAY): chains})
            )
            for trade in result.trades:
                entry_ts = trade.details["entry_ts"]
                exit_ts = trade.details["exit_ts"]

                # (2) timestamps are real snapshots and correctly ordered.
                assert entry_ts in snapshot_ts, (seed, overrides, entry_ts)
                assert exit_ts in snapshot_ts, (seed, overrides, exit_ts)
                assert entry_ts <= exit_ts, (seed, overrides)

                # (3) fills equal the mark at their own snapshot (no later peek).
                assert trade.legs[0].entry_price == mark_at[entry_ts]
                assert trade.legs[0].exit_price == mark_at[exit_ts]

                # (1) truncation invariance: deleting snapshots after the exit
                # must not change the trade.
                exit_dt = datetime.fromisoformat(exit_ts)
                truncated = [c for c in chains if c.as_of <= exit_dt]
                trunc_result = run_intraday_options_backtest(
                    cfg, _StubProvider({("SPY", DAY): truncated})
                )
                assert trunc_result.trades, (seed, overrides, "trade vanished")
                assert _key(trunc_result.trades[0]) == _key(trade), (seed, overrides)
