"""Offline phase timer for the historical and rolling backtest engines.

This intentionally reuses ``profiling/harness.py`` so the workload and seed are
the same as the repository's existing performance baseline. Timed phases are
inclusive: for example, ``assemble_results`` contains ``compute_metrics``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
import screener.backtester.historical as historical  # noqa: E402
import screener.backtester.rolling_simulation as rolling  # noqa: E402
import screener.backtester.signal_panel as signal_panel  # noqa: E402


class TimedFetcher:
    """Delegate to the deterministic fetcher while accumulating fetch time."""

    def __init__(
        self, delegate: harness.StubFetcher, timings: dict[str, float]
    ) -> None:
        self._delegate = delegate
        self._timings = timings

    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._delegate.fetch(*args, **kwargs)
        finally:
            self._timings["fetch"] = self._timings.get("fetch", 0.0) + (
                time.perf_counter() - started
            )


@contextmanager
def timed_functions(
    specs: list[tuple[ModuleType, str, str]], timings: dict[str, float]
) -> Iterator[None]:
    """Temporarily wrap module globals with low-overhead inclusive timers."""

    originals: list[tuple[ModuleType, str, Callable[..., Any]]] = []
    for module, attribute, label in specs:
        original = getattr(module, attribute)
        originals.append((module, attribute, original))

        def wrapper(
            *args: Any,
            __original: Callable[..., Any] = original,
            __label: str = label,
            **kwargs: Any,
        ) -> Any:
            started = time.perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                timings[__label] = timings.get(__label, 0.0) + (
                    time.perf_counter() - started
                )

        setattr(module, attribute, wrapper)
    try:
        yield
    finally:
        for module, attribute, original in reversed(originals):
            setattr(module, attribute, original)


def _workload(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    n_bars = int(args.years * 252) + harness.SLOW * 3 + 400
    data = harness.build_data(args.tickers, n_bars, args.seed)
    tickers = tuple(f"T{i:04d}" for i in range(args.tickers))
    index = data["SPY"].index
    window_bars = int(args.years * 252)
    start_date = index[-window_bars].date()
    end_date = index[-1].date()
    cfg_date = end_date if args.path == "rolling" else start_date
    cfg = harness.make_cfg(args.path, tickers, args.top, cfg_date)
    return data, start_date, end_date, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=["rolling", "historical"], required=True)
    parser.add_argument("--tickers", type=int, default=300)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    data, start_date, end_date, cfg = _workload(args)
    timings: dict[str, float] = {}
    fetcher = TimedFetcher(harness.StubFetcher(data), timings)

    if args.path == "rolling":
        specs = [
            (rolling, "build_price_panel", "price_panel"),
            (rolling, "build_signal_panel", "signal_panel"),
            (signal_panel, "_precompute_entry_signals", "entry_signals"),
            (signal_panel, "_precompute_filter_signals", "filter_signals"),
            (signal_panel, "_build_rolling_candidate_matrices", "candidate_matrices"),
            (rolling, "run_day_loop", "day_loop"),
            (rolling, "_candidate_rows_for_day", "candidate_rows"),
            (rolling, "_make_slot_state", "make_slot_state"),
            (rolling, "_force_close_open_slots", "force_close"),
            (rolling, "_assemble_results", "assemble_results"),
            (rolling, "build_equity_curve", "equity_curve"),
            (rolling, "compute_metrics", "metrics"),
            (rolling, "compute_regime_metrics", "regime_metrics"),
        ]

        def run() -> Any:
            return rolling.run_rolling_backtest(
                cfg, fetcher, start_date=start_date, end_date=end_date
            )

    else:
        specs = [
            (historical, "_prepare_strategy_bars", "strategy_bars"),
            (historical, "select_candidates", "select_candidates"),
            (historical, "_run_event_driven_sim", "event_simulation"),
            (historical, "run_day_loop", "day_loop"),
            (historical, "_make_slot_state", "make_slot_state"),
            (historical, "build_equity_curve", "equity_curve"),
            (historical, "compute_metrics", "metrics"),
            (historical, "compute_regime_metrics", "regime_metrics"),
        ]

        def run() -> Any:
            return historical.run_backtest(cfg, fetcher)

    started = time.perf_counter()
    with timed_functions(specs, timings):
        for _ in range(args.repeat):
            result = run()
    total = time.perf_counter() - started
    per_run = total / args.repeat
    payload = {
        "path": args.path,
        "tickers": args.tickers,
        "years": args.years,
        "top": args.top,
        "repeat": args.repeat,
        "trades": len(result.trades),
        "seconds_per_run": round(per_run, 6),
        "inclusive_phases": {
            key: {
                "seconds_per_run": round(value / args.repeat, 6),
                "percent_of_total": round(100.0 * value / total, 2),
            }
            for key, value in sorted(timings.items(), key=lambda item: -item[1])
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
