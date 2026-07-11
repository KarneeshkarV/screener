"""Measure output-equivalent optimization ceilings for current hot spots.

The substitutions here are deliberately local monkeypatches, not production
changes. Each candidate run is checked field-for-field against the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
import screener.backtester.core as core  # noqa: E402
import screener.backtester.metrics as metrics_module  # noqa: E402
import screener.backtester.pine as pine  # noqa: E402
from screener.backtester.models import BacktestResult, Trade  # noqa: E402


def _assert_equal(left: BacktestResult, right: BacktestResult) -> None:
    assert left.config == right.config
    assert left.trades == right.trades
    pd.testing.assert_series_equal(
        left.equity_curve, right.equity_curve, check_exact=True
    )
    pd.testing.assert_series_equal(
        left.benchmark_curve, right.benchmark_curve, check_exact=True
    )
    pd.testing.assert_frame_equal(left.selection, right.selection, check_exact=True)
    assert left.metrics == right.metrics
    assert left.warnings == right.warnings


def _fast_exposure(
    equity_index: pd.DatetimeIndex, trades: list[Trade], slot_count: int
) -> float:
    """Difference-array equivalent of the current inclusive exposure masks."""

    if not trades or len(equity_index) == 0:
        return 0.0
    changes = np.zeros(len(equity_index) + 1, dtype=np.int64)
    entries = pd.DatetimeIndex([pd.Timestamp(trade.entry_date) for trade in trades])
    exits = pd.DatetimeIndex([pd.Timestamp(trade.exit_date) for trade in trades])
    starts = equity_index.searchsorted(entries, side="left")
    stops = equity_index.searchsorted(exits, side="right")
    valid = (starts < len(equity_index)) & (stops > 0)
    np.add.at(changes, starts[valid], 1)
    np.add.at(changes, np.minimum(stops[valid], len(equity_index)), -1)
    open_count = np.cumsum(changes[:-1])
    return float(open_count.mean() / max(slot_count, 1))


def _freeze_node(value: Any) -> Any:
    """Semantic Pine key that ignores source-column metadata."""

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_node(item) for item in value)
    fields = getattr(value.__class__, "model_fields", None)
    if fields is None:
        return value
    return (
        value.__class__.__name__,
        tuple(
            (name, _freeze_node(getattr(value, name)))
            for name in fields
            if name != "col"
        ),
    )


def _run(run: Any) -> tuple[float, BacktestResult]:
    started = time.perf_counter()
    result = run()
    return time.perf_counter() - started, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=300)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--exposure-repeat", type=int, default=2)
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

    def run() -> BacktestResult:
        return harness.run_rolling_backtest(
            cfg, fetcher, start_date=start_date, end_date=end_date
        )

    baseline_seconds, baseline = _run(run)

    original_liquidity = core._cached_trailing_liquidity
    core._cached_trailing_liquidity = lambda *_args, **_kwargs: (0.0, 0.0)
    try:
        no_unused_liquidity_seconds, no_unused_liquidity = _run(run)
    finally:
        core._cached_trailing_liquidity = original_liquidity
    _assert_equal(baseline, no_unused_liquidity)

    original_eval_call = pine._eval_call
    call_cache: dict[tuple[int, Any], Any] = {}
    call_stats = {"calls": 0, "hits": 0}

    def cached_eval_call(node: Any, bars: pd.DataFrame) -> Any:
        key = (id(bars), _freeze_node(node))
        call_stats["calls"] += 1
        if key in call_cache:
            call_stats["hits"] += 1
            return call_cache[key]
        value = original_eval_call(node, bars)
        call_cache[key] = value
        return value

    pine._eval_call = cached_eval_call
    try:
        indicator_memo_seconds, indicator_memo = _run(run)
    finally:
        pine._eval_call = original_eval_call
    _assert_equal(baseline, indicator_memo)
    indicator_call_stats = dict(call_stats)

    original_exposure = metrics_module._exposure
    call_cache.clear()
    call_stats.update(calls=0, hits=0)
    core._cached_trailing_liquidity = lambda *_args, **_kwargs: (0.0, 0.0)
    pine._eval_call = cached_eval_call
    metrics_module._exposure = _fast_exposure
    try:
        combined_seconds, combined = _run(run)
    finally:
        metrics_module._exposure = original_exposure
        pine._eval_call = original_eval_call
        core._cached_trailing_liquidity = original_liquidity
    _assert_equal(baseline, combined)

    exposure_started = time.perf_counter()
    exposure_value = 0.0
    for _ in range(args.exposure_repeat):
        exposure_value = metrics_module._exposure(
            pd.DatetimeIndex(baseline.equity_curve.index),
            baseline.trades,
            args.top,
        )
    exposure_seconds = (time.perf_counter() - exposure_started) / args.exposure_repeat

    fast_started = time.perf_counter()
    fast_value = 0.0
    fast_repeats = max(100, args.exposure_repeat)
    for _ in range(fast_repeats):
        fast_value = _fast_exposure(
            pd.DatetimeIndex(baseline.equity_curve.index),
            baseline.trades,
            args.top,
        )
    fast_seconds = (time.perf_counter() - fast_started) / fast_repeats
    assert exposure_value == fast_value

    payload = {
        "workload": {
            "tickers": args.tickers,
            "years": args.years,
            "top": args.top,
            "trades": len(baseline.trades),
            "equity_points": len(baseline.equity_curve),
        },
        "baseline_seconds": round(baseline_seconds, 6),
        "fixed_slippage_skip_unused_liquidity": {
            "seconds": round(no_unused_liquidity_seconds, 6),
            "speedup": round(baseline_seconds / no_unused_liquidity_seconds, 3),
            "output_exact": True,
        },
        "semantic_indicator_call_memo": {
            "seconds": round(indicator_memo_seconds, 6),
            "speedup": round(baseline_seconds / indicator_memo_seconds, 3),
            "call_count": indicator_call_stats["calls"],
            "cache_hits": indicator_call_stats["hits"],
            "output_exact": True,
        },
        "combined_candidates": {
            "seconds": round(combined_seconds, 6),
            "speedup": round(baseline_seconds / combined_seconds, 3),
            "output_exact": True,
        },
        "exposure_difference_array": {
            "current_seconds": round(exposure_seconds, 6),
            "candidate_seconds": round(fast_seconds, 6),
            "speedup": round(exposure_seconds / fast_seconds, 1),
            "value": exposure_value,
            "output_exact": True,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
