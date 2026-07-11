"""Benchmark the rolling grid-search multiplier with deterministic data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
from screener.backtester.optimization.grid import grid_search  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=50)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", default="1,2,4")
    args = parser.parse_args()

    n_bars = int(args.years * 252) + harness.SLOW * 3 + 400
    data = harness.build_data(args.tickers, n_bars, args.seed)
    tickers = tuple(f"T{i:04d}" for i in range(args.tickers))
    fetcher = harness.StubFetcher(data)
    index = data["SPY"].index
    window_bars = int(args.years * 252)
    start_date = index[-window_bars].date()
    end_date = index[-1].date()
    cfg = harness.make_cfg("rolling", tickers, args.top, end_date)
    parameter_grid = {
        "hold": [10, 20],
        "stop_loss": [None, 0.08],
        "take_profit": [None, 0.20],
    }
    combo_count = 8

    runs: list[dict[str, object]] = []
    for workers in [int(item) for item in args.workers.split(",") if item.strip()]:
        started = time.perf_counter()
        results = grid_search(
            cfg,
            fetcher,
            parameter_grid,
            metric="sharpe",
            top_n=combo_count,
            min_trades=1,
            max_workers=workers,
            runner="rolling",
            start_date=start_date,
            end_date=end_date,
        )
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "workers": workers,
                "seconds": round(elapsed, 6),
                "seconds_per_combo": round(elapsed / combo_count, 6),
                "results": len(results),
                "errors": sum(result.error is not None for result in results),
            }
        )

    print(
        json.dumps(
            {
                "tickers": args.tickers,
                "years": args.years,
                "top": args.top,
                "combinations": combo_count,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
