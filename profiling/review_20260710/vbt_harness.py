"""Deterministic harness for the vectorbt parameter-sweep path."""

from __future__ import annotations

import argparse
import cProfile
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
from screener.backtester.vbt.sweep import run_parameter_sweep  # noqa: E402


def _panel(
    data: dict[str, pd.DataFrame], tickers: list[str], column: str
) -> pd.DataFrame:
    return pd.DataFrame({ticker: data[ticker][column] for ticker in tickers})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=100)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--cprofile-out", type=Path)
    args = parser.parse_args()

    n_bars = int(args.years * 252) + harness.SLOW * 3 + 400
    data = harness.build_data(args.tickers, n_bars, args.seed)
    tickers = [f"T{i:04d}" for i in range(args.tickers)]
    close = _panel(data, tickers, "close")
    open_ = _panel(data, tickers, "open")
    high = _panel(data, tickers, "high")
    low = _panel(data, tickers, "low")
    volume = _panel(data, tickers, "volume")

    kwargs = {
        "fast_values": [5, 10, 20, 30],
        "slow_values": [40, 50, 100],
        "hold_values": [0, 10, 20],
        "indicators": ["sma", "ema", "breakout", "rsi", "macd"],
        "open_": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "chunk_size": args.chunk_size,
    }

    warm_started = time.perf_counter()
    run_parameter_sweep(
        close.iloc[:100, :2],
        fast_values=[5],
        slow_values=[20],
        hold_values=[10],
        indicators=["sma"],
        open_=open_.iloc[:100, :2],
    )
    warm_seconds = time.perf_counter() - warm_started

    if args.cprofile_out:
        profiler = cProfile.Profile()
        profiler.enable()
        result = run_parameter_sweep(close, **kwargs)
        profiler.disable()
        profiler.dump_stats(args.cprofile_out)
        elapsed = float("nan")
    else:
        started = time.perf_counter()
        result = run_parameter_sweep(close, **kwargs)
        elapsed = time.perf_counter() - started

    print(
        f"[vbt] tickers={args.tickers} bars={n_bars} combos={len(result)} "
        f"chunk_size={args.chunk_size or 'auto'}"
    )
    print(f"  JIT WARMUP: {warm_seconds:.3f}s")
    print(f"  WALL CLOCK: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
