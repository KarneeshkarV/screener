"""Execution and aggregation for the research Pine runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from screener.logging_config import get_logger
from screener.research.pine_runner.constants import BENCHMARKS
from screener.research.pine_runner.data import fetch_ohlcv, load_universe
from screener.strategies.registry import STRATEGIES
from screener.strategies.trades import Trade

log = get_logger("pine_runner")


class MarketRun(BaseModel):
    market: str
    today: date
    window_start: pd.Timestamp
    benchmark_symbol: str
    benchmark_return: float | None
    per_strategy: dict[str, list[dict]]
    error_counts: dict[str, int]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("market", "benchmark_symbol")
    @classmethod
    def _normalize_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


def _compound(trades: list[Trade]) -> float:
    r = 1.0
    for t in trades:
        r *= 1 + t.ret
    return r - 1.0


def _run_ticker(
    df: pd.DataFrame, window_start: pd.Timestamp, strategy_fn
) -> dict | None:
    """Run one strategy on one ticker with pre-window indicator warmup."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        return None
    trades = strategy_fn(df)
    in_win = [t for t in trades if t.entry_date >= window_start]
    n_bars_window = int((pd.to_datetime(df["date"]) >= window_start).sum())
    exposure = sum(t.exit_idx - t.entry_idx for t in in_win)
    return {
        "n_trades": len(in_win),
        "n_bars": n_bars_window,
        "exposure": exposure,
        "total_return": _compound(in_win),
        "wins": sum(1 for t in in_win if t.ret > 0),
        "trades": in_win,
    }


def run_market(
    *,
    market: str,
    years: int,
    limit: int,
    refresh: bool,
) -> MarketRun:
    today = date.today()
    window_start_ts = pd.Timestamp(today) - pd.DateOffset(years=years)
    window_start_ts = window_start_ts.normalize()
    fetch_start = (pd.Timestamp(today) - pd.DateOffset(years=years + 4)).date()
    fetch_end = today

    tickers = load_universe(market)
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    log.info(
        "backtest.run_started",
        market=market,
        tickers=len(tickers),
        window_start=str(window_start_ts.date()),
        window_end=str(today),
        years=years,
        warmup_start=str(fetch_start),
        strategies=list(STRATEGIES),
    )

    ohlcv: dict[str, pd.DataFrame] = {}

    def _fetch(t: str):
        df = fetch_ohlcv(t, fetch_start, fetch_end, market, refresh=refresh)
        return t, df

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t, df = fut.result()
            if df is not None and not df.empty:
                ohlcv[t] = df
            if i % 50 == 0 or i == len(tickers):
                log.info(
                    "backtest.fetch_progress",
                    fetched=i,
                    total=len(tickers),
                    with_data=len(ohlcv),
                )

    bench_sym = BENCHMARKS[market]
    bench_df = fetch_ohlcv(bench_sym, fetch_start, fetch_end, market, refresh=refresh)
    bench_return: float | None = None
    if bench_df is not None and not bench_df.empty:
        b = bench_df.sort_values("date")
        b = b[pd.to_datetime(b["date"]) >= window_start_ts]
        if len(b) > 1:
            bench_return = float(b["adj_close"].iloc[-1] / b["adj_close"].iloc[0] - 1.0)
    if bench_return is None:
        log.warning("backtest.benchmark_missing", benchmark=bench_sym)

    per_strat: dict[str, list[dict]] = {n: [] for n in STRATEGIES}
    err_counts: dict[str, int] = {n: 0 for n in STRATEGIES}
    for i, (t, df) in enumerate(ohlcv.items(), 1):
        for name, fn in STRATEGIES.items():
            try:
                res = _run_ticker(df, window_start_ts, fn)
            except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as exc:
                err_counts[name] += 1
                log.debug(
                    "backtest.strategy_error",
                    ticker=t,
                    strategy=name,
                    error=str(exc),
                    exc_info=True,
                )
                continue
            if res is None:
                continue
            per_strat[name].append(res | {"ticker": t})
        if i % 100 == 0 or i == len(ohlcv):
            log.info("backtest.iter_progress", processed=i, total=len(ohlcv))

    return MarketRun(
        market=market,
        today=today,
        window_start=window_start_ts,
        benchmark_symbol=bench_sym,
        benchmark_return=bench_return,
        per_strategy=per_strat,
        error_counts=err_counts,
    )
