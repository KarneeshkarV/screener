"""Deterministic, offline profiling harness for the backtester hot paths.

Generates seeded synthetic OHLCV frames and drives either the rolling engine
(``run_rolling_backtest``) or the historical engine (``run_backtest``) through a
stub fetcher, so the CPU profile reflects pure engine work with zero network I/O.

Usage:
    python profiling/harness.py --path rolling    [--tickers N --years Y --top T --seed S]
    python profiling/harness.py --path historical [--tickers N --years Y --top T --seed S]
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pickle
import tempfile
import time
from datetime import date
from typing import Iterable, cast

import numpy as np
import pandas as pd

from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.rolling_simulation import run_rolling_backtest

CACHE_DIR = os.environ.get(
    "HARNESS_CACHE",
    os.path.join(tempfile.gettempdir(), "screener-harness-cache"),
)

# SMA-crossover strategy (pure-pandas indicators, exercises the Pine evaluator).
FAST = 20
SLOW = 50
ENTRY_EXPR = f"crossover(close, sma(close, {SLOW})) and close > sma(close, {FAST})"
EXIT_EXPR = f"crossunder(close, sma(close, {SLOW}))"


class StubFetcher:
    """Offline fetcher: returns pre-built synthetic frames sliced to [start, end]."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)
        frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            frame = self._data.get(ticker)
            if frame is None or frame.empty:
                frames[ticker] = pd.DataFrame()
                continue
            frames[ticker] = frame.loc[
                (frame.index >= start_timestamp) & (frame.index <= end_timestamp)
            ]
        return frames


def _make_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Geometric random walk with drift -> realistic OHLCV producing crossovers."""
    idx = pd.bdate_range("2016-01-01", periods=n)
    # daily log-returns: small drift + noise, occasional regime via slow sine
    t = np.arange(n)
    drift = 0.0003 + 0.0006 * np.sin(2.0 * np.pi * t / 252.0)
    shocks = rng.normal(0.0, 0.018, size=n)
    log_ret = drift + shocks
    close = 50.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.003, size=n - 1))
    intraday = np.abs(rng.normal(0.0, 0.01, size=n))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def build_data(n_tickers: int, n_bars: int, seed: int) -> dict[str, pd.DataFrame]:
    """Build (or load from cache) the synthetic panel.

    Caching keeps data-generation cost (dominated by pandas ``bdate_range``) out
    of repeated profiling runs so the flamegraph reflects engine work only.
    """
    cache = os.path.join(CACHE_DIR, f"panel_{n_tickers}_{n_bars}_{seed}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return cast(dict[str, pd.DataFrame], pickle.load(fh))
    rng = np.random.default_rng(seed)
    data: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        data[f"T{i:04d}"] = _make_frame(rng, n_bars)
    data["SPY"] = _make_frame(rng, n_bars)  # benchmark
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(data, fh)
    return data


def make_cfg(
    path: str, tickers: tuple[str, ...], top: int, as_of: date
) -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=as_of,
        hold=20,
        top=top,
        entry_expr=ENTRY_EXPR,
        exit_expr=EXIT_EXPR,
        stop_loss=0.08,
        take_profit=0.20,
        trailing_stop=None,
        slippage_bps=1.0,
        commission_bps=1.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=tickers,
        max_universe=0,
        min_price=1.0,
        min_avg_dollar_volume=1000.0,
        avg_dollar_volume_window=20,
        reserve_multiple=3,
        reinvest=True,
        gap_fills=True,
        entry_order_type="moo",
        allow_reentry=True,
        max_reentries=5,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["rolling", "historical"], required=True)
    ap.add_argument("--tickers", type=int, default=300)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument(
        "--cprofile-out",
        default=None,
        help="If set, cProfile ONLY the engine call (data prebuilt) to this path.",
    )
    args = ap.parse_args()

    n_bars = int(args.years * 252) + SLOW * 3 + 400  # window + warmup buffer
    data = build_data(args.tickers, n_bars, args.seed)
    tickers = tuple(f"T{i:04d}" for i in range(args.tickers))
    fetcher = StubFetcher(data)

    full_idx = data["SPY"].index
    end_date = full_idx[-1].date()
    # Window = last ``years`` of the frame; warmup history precedes it.
    window_bars = int(args.years * 252)
    start_date = full_idx[-window_bars].date()
    as_of = full_idx[-window_bars].date()

    if args.path == "rolling":
        cfg = make_cfg("rolling", tickers, args.top, end_date)

        def _run() -> BacktestResult:
            return run_rolling_backtest(
                cfg, fetcher, start_date=start_date, end_date=end_date
            )
    else:
        cfg = make_cfg("historical", tickers, args.top, as_of)

        def _run() -> BacktestResult:
            return run_backtest(cfg, fetcher)

    if args.cprofile_out:
        # Engine-only profile: data already built above, so the stats file
        # contains pure engine work with no synthetic-data-generation noise.
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.repeat):
            result = _run()
        pr.disable()
        pr.dump_stats(args.cprofile_out)
        elapsed = float("nan")
    else:
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            result = _run()
        elapsed = time.perf_counter() - t0

    print(
        f"[{args.path}] tickers={args.tickers} years={args.years} "
        f"bars/ticker={n_bars} top={args.top} repeat={args.repeat}"
    )
    print(f"  window: {start_date} -> {end_date}")
    print(f"  trades: {len(result.trades)}  warnings: {len(result.warnings)}")
    print(f"  WALL CLOCK: {elapsed:.3f}s  ({elapsed / args.repeat:.3f}s per run)")


if __name__ == "__main__":
    main()
