"""Measure full-hit parquet cache reload cost versus an in-memory panel."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
from screener.backtester.data import (  # noqa: E402
    YFinancePriceFetcher,
    _save_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=300)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    n_bars = int(args.years * 252) + harness.SLOW * 3 + 400
    data = harness.build_data(args.tickers, n_bars, args.seed)
    tickers = [f"T{i:04d}" for i in range(args.tickers)] + ["SPY"]
    index = data["SPY"].index
    start = index[0].date()
    end = index[-1].date()

    in_memory = harness.StubFetcher(data)
    started = time.perf_counter()
    for _ in range(args.repeat):
        memory_result = in_memory.fetch(tickers, start, end)
    memory_seconds = (time.perf_counter() - started) / args.repeat

    with tempfile.TemporaryDirectory(prefix="screener-cache-bench-") as raw_dir:
        cache_dir = Path(raw_dir)
        for ticker in tickers:
            _save_cache(ticker, data[ticker], cache_dir)
        parquet = YFinancePriceFetcher(cache_dir=cache_dir)
        samples: list[float] = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            parquet_result = parquet.fetch(tickers, start, end)
            samples.append(time.perf_counter() - started)

    assert len(memory_result) == len(parquet_result) == len(tickers)
    print(
        json.dumps(
            {
                "tickers_including_benchmark": len(tickers),
                "bars_per_ticker": n_bars,
                "repeat": args.repeat,
                "in_memory_seconds_per_fetch": round(memory_seconds, 6),
                "parquet_seconds_per_fetch": [round(value, 6) for value in samples],
                "parquet_median_seconds": round(sorted(samples)[len(samples) // 2], 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
