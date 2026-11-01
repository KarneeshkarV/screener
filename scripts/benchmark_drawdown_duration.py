"""Offline benchmark of drawdown duration against the original scalar algorithm.

Run with ``uv run python scripts/benchmark_drawdown_duration.py``.
Timing measures this metric only, not the complete backtest.
"""

from __future__ import annotations

import statistics
from timeit import repeat

import numpy as np
import pandas as pd

from screener.backtester.metrics import _max_drawdown_duration_days


def scalar_drawdown_duration(equity: pd.Series) -> float:
    """Reference peak-to-recovery duration in calendar days for finite equity."""
    peak = float(equity.iloc[0])
    peak_time = equity.index[0]
    longest = 0.0
    in_drawdown = False
    drawdown_start = peak_time
    for position, value in enumerate(equity.to_numpy(dtype=float)):
        stamp = equity.index[position]
        if value >= peak:
            if in_drawdown:
                longest = max(
                    longest, (stamp - drawdown_start).total_seconds() / 86400.0
                )
                in_drawdown = False
            peak = value
            peak_time = stamp
        elif not in_drawdown:
            in_drawdown = True
            drawdown_start = peak_time
    if in_drawdown:
        longest = max(
            longest, (equity.index[-1] - drawdown_start).total_seconds() / 86400.0
        )
    return float(longest)


def main() -> None:
    """Compare identical results and median timings on a fixed 100,000-bar curve."""
    equity = pd.Series(
        100 + np.random.default_rng(42).normal(size=100_000).cumsum(),
        index=pd.date_range("2020-01-01", periods=100_000, freq="15min"),
    )
    expected = scalar_drawdown_duration(equity)
    actual = _max_drawdown_duration_days(equity)
    if actual != expected:
        raise AssertionError(f"Drawdown duration mismatch: {actual} != {expected}")
    scalar = statistics.median(
        repeat(lambda: scalar_drawdown_duration(equity), number=1, repeat=5)
    )
    vectorized = statistics.median(
        repeat(lambda: _max_drawdown_duration_days(equity), number=1, repeat=5)
    )
    print(f"bars={len(equity)} duration_days={actual:.6f}")
    print(f"scalar_median_ms={scalar * 1000:.3f}")
    print(f"vectorized_median_ms={vectorized * 1000:.3f}")
    print(f"metric_speedup={scalar / vectorized:.1f}x")


if __name__ == "__main__":
    main()
