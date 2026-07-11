"""Deterministic CPU harness for the earnings-event backtest engine."""

from __future__ import annotations

import argparse
import cProfile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import screener.earnings_backtest.engine as engine  # noqa: E402


def _frame(rng: np.random.Generator, bars: int) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=bars)
    returns = rng.normal(0.0003, 0.018, bars)
    close = 50.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.01, bars))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + spread),
            "low": np.minimum(open_, close) * (1.0 - spread),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, bars).astype(float),
        },
        index=index,
    )


def _fixture(
    tickers: int, bars: int, events_per_ticker: int, seed: int
) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = [f"T{i:04d}" for i in range(tickers)]
    panel = {ticker: _frame(rng, bars) for ticker in names}
    rows: list[dict[str, object]] = []
    candidate_positions = np.linspace(40, bars - 2, events_per_ticker, dtype=int)
    for ticker in names:
        index = panel[ticker].index
        rows.extend(
            {"ticker": ticker, "earnings_date": index[position]}
            for position in candidate_positions
        )
    return names, panel, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=200)
    parser.add_argument("--bars", type=int, default=800)
    parser.add_argument("--events-per-ticker", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cprofile-out", type=Path)
    args = parser.parse_args()

    names, panel, events = _fixture(
        args.tickers, args.bars, args.events_per_ticker, args.seed
    )
    original_events = engine.collect_earnings_events
    original_prices = engine.fetch_price_data
    engine.collect_earnings_events = lambda *_args, **_kwargs: events.copy()
    engine.fetch_price_data = lambda batch, *_args, **_kwargs: {
        ticker: panel[ticker] for ticker in batch
    }

    def run() -> list[engine.EarningsTrade]:
        return engine.run_earnings_backtest(
            "us",
            years=3,
            strategy="combined_score",
            tickers=names,
            batch_size=50,
        )

    try:
        if args.cprofile_out:
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(args.repeat):
                trades = run()
            profiler.disable()
            profiler.dump_stats(args.cprofile_out)
            elapsed = float("nan")
        else:
            started = time.perf_counter()
            for _ in range(args.repeat):
                trades = run()
            elapsed = time.perf_counter() - started
    finally:
        engine.collect_earnings_events = original_events
        engine.fetch_price_data = original_prices

    print(
        f"[earnings] tickers={args.tickers} bars={args.bars} "
        f"events/ticker={args.events_per_ticker} events={len(events)}"
    )
    print(f"  trades: {len(trades)}")
    print(f"  WALL CLOCK: {elapsed:.3f}s  ({elapsed / args.repeat:.3f}s per run)")


if __name__ == "__main__":
    main()
