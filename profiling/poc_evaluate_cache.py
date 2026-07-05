"""Proof-of-concept: confirm redundant Pine `evaluate` calls dominate.

We monkeypatch `screener.backtester.pine.evaluate` (and the copies imported into
core/historical) with a thin memo keyed by (id(ast_node), id(bars_frame)). Frames
are long-lived objects reused across the run, so this collapses repeated
evaluations of the SAME expression over the SAME frame into one — exactly the
redundancy the profile shows in `_make_slot_state` (rolling) and
`select_candidates`/`_eligible_reserve_signal_idx` (historical).

This is a MEASUREMENT wrapper, not an engine change: it proves the ceiling of
"precompute indicators once per ticker instead of per open/per scan".
"""

from __future__ import annotations

import argparse
import time

import screener.backtester.core as core
import screener.backtester.historical as historical
import screener.backtester.pine as pine

_orig_evaluate = pine.evaluate


def _run(path: str) -> tuple[float, int]:
    import profiling.harness as h

    n_bars = int(3 * 252) + h.SLOW * 3 + 400
    data = h.build_data(300, n_bars, 12345)
    tickers = tuple(f"T{i:04d}" for i in range(300))
    fetcher = h.StubFetcher(data)
    idx = data["SPY"].index
    end_date = idx[-1].date()
    start_date = idx[-756].date()
    t0 = time.perf_counter()
    if path == "rolling":
        cfg = h.make_cfg("rolling", tickers, 10, end_date)
        res = h.run_rolling_backtest(
            cfg, fetcher, start_date=start_date, end_date=end_date
        )
    else:
        cfg = h.make_cfg("historical", tickers, 10, start_date)
        res = h.run_backtest(cfg, fetcher)
    return time.perf_counter() - t0, len(res.trades)


def _install_cache() -> dict:
    stats = {"calls": 0, "hits": 0}
    memo: dict = {}

    def cached(node, bars):
        key = (id(node), id(bars), len(bars))
        stats["calls"] += 1
        hit = memo.get(key)
        if hit is not None:
            stats["hits"] += 1
            return hit
        out = _orig_evaluate(node, bars)
        memo[key] = out
        return out

    pine.evaluate = cached
    core.evaluate = cached
    historical.evaluate = cached
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["rolling", "historical"], required=True)
    args = ap.parse_args()

    base, ntr = _run(args.path)
    stats = _install_cache()
    cached, ntr2 = _run(args.path)

    print(f"[{args.path}]")
    print(f"  baseline           : {base:.3f}s  ({ntr} trades)")
    print(f"  with evaluate-memo : {cached:.3f}s  ({ntr2} trades)")
    print(
        f"  evaluate calls={stats['calls']}  memo-hits={stats['hits']} "
        f"({100 * stats['hits'] / max(stats['calls'], 1):.0f}% redundant)"
    )
    print(f"  speedup            : {base / cached:.2f}x")


if __name__ == "__main__":
    main()
