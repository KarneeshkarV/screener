"""Autoresearch evaluator — Karpathy-style strategy search loop.

Runs every registered strategy (stock + `autoresearch_strategies.NEW_STRATEGIES`)
on in-sample AND out-of-sample windows, builds a daily equal-weight basket
equity curve from trades, computes DSR / Sortino / Calmar via
``screener.backtester.metrics.compute_metrics``, and appends ONE line per
strategy to ``journal.jsonl``.

Composite ranking score (OOS only):
    score = dsr * exp(-|max_drawdown| / dd_tolerance)

DSR is already sample-size and multi-testing aware (``n_trials`` = number of
strategies evaluated this run), so it absorbs most of the overfitting risk of
running many variants. The drawdown haircut is a separate sanity brake: a
strategy cannot buy a high DSR with catastrophic drawdowns.

Usage:
    uv run python autoresearch.py evaluate --iteration 1
    uv run python autoresearch.py evaluate --iteration 1 --limit 50 --years 3
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import click
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from run_pinescript_strategies import (
    BENCHMARKS,
    STRATEGIES as STOCK_STRATEGIES,
    Trade as PineTrade,
    fetch_ohlcv,
    load_universe,
)
from screener.backtester.metrics import compute_metrics

try:
    from autoresearch_strategies import NEW_STRATEGIES
except Exception as e:  # noqa: BLE001
    print(f"[warn] autoresearch_strategies import failed: {e}", file=sys.stderr)
    NEW_STRATEGIES = {}


JOURNAL_PATH = Path("journal.jsonl")
CACHE_DIR = Path(".autoresearch/ohlcv")
DD_TOLERANCE = 0.25  # 25% drawdown → score halved (via exp(-1))
DEFAULT_SLOTS = 20   # concurrent-positions denominator for basket equity


def _cache_path(market: str, ticker: str, start, end) -> Path:
    safe = ticker.replace(":", "_").replace("/", "_").replace("^", "_")
    return CACHE_DIR / market / f"{safe}__{start}__{end}.parquet"


def _cached_fetch(ticker: str, start, end, market: str, refresh: bool):
    p = _cache_path(market, ticker, start, end)
    if p.exists() and not refresh:
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            p.unlink(missing_ok=True)
    df = fetch_ohlcv(ticker, start, end, market, refresh=refresh)
    if df is not None and not df.empty:
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(p)
        except Exception:  # noqa: BLE001
            pass
    return df


# ───────────────────── shim: pine trade → compute_metrics shape ─────────────

@dataclass
class _MetricTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    entry_cost: float


def _to_metric_trades(trades: list[PineTrade]) -> list[_MetricTrade]:
    out: list[_MetricTrade] = []
    for t in trades:
        cost = float(t.entry_px)
        pnl = float(t.exit_px - t.entry_px)
        out.append(_MetricTrade(
            entry_date=pd.Timestamp(t.entry_date),
            exit_date=pd.Timestamp(t.exit_date),
            pnl=pnl,
            entry_cost=cost,
        ))
    return out


# ───────────────────── daily basket equity from trades ─────────────────────

def _basket_equity(
    ohlcv: dict[str, pd.DataFrame],
    trades_by_ticker: dict[str, list[PineTrade]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    slot_count: int = DEFAULT_SLOTS,
) -> pd.Series:
    """Daily basket equity with a FIXED capital-slot denominator.

    Each held ticker gets weight ``1 / max(held, slot_count)``, so empty slots
    earn zero and sparse strategies are not implicitly levered. This mirrors
    engine.py's portfolio model: you size positions against a fixed capital
    base split across ``slot_count`` slots, not against the count of names
    currently open."""
    master = set()
    for df in ohlcv.values():
        dts = pd.to_datetime(df["date"])
        master.update(dts[(dts >= window_start) & (dts <= window_end)].tolist())
    if not master:
        return pd.Series(dtype=float)
    dates = pd.DatetimeIndex(sorted(master))
    return_sum = pd.Series(0.0, index=dates)
    held = pd.Series(0, index=dates, dtype=int)

    for ticker, trades in trades_by_ticker.items():
        if not trades:
            continue
        df = ohlcv[ticker].sort_values("date").reset_index(drop=True)
        tdates = pd.to_datetime(df["date"]).to_numpy()
        closes = df["close"].to_numpy(dtype=float)
        for tr in trades:
            # daily returns accrue from the bar AFTER entry through the exit bar
            start_i = max(tr.entry_idx + 1, 1)
            end_i = tr.exit_idx
            if end_i < start_i:
                continue
            for i in range(start_i, end_i + 1):
                d = pd.Timestamp(tdates[i])
                if d < window_start or d > window_end:
                    continue
                if closes[i - 1] <= 0:
                    continue
                r = closes[i] / closes[i - 1] - 1.0
                if d in return_sum.index:
                    return_sum.loc[d] += r
                    held.loc[d] += 1

    denom = np.maximum(held.to_numpy(dtype=float), float(max(slot_count, 1)))
    avg = pd.Series(return_sum.to_numpy() / denom, index=dates).fillna(0.0)
    return (1.0 + avg).cumprod()


# ───────────────────── strategy evaluation ─────────────────────────────────

def _run_strategy(
    strat_fn: Callable,
    ohlcv: dict[str, pd.DataFrame],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[dict[str, list[PineTrade]], int]:
    trades_by_ticker: dict[str, list[PineTrade]] = {}
    n_trades = 0
    for ticker, df in ohlcv.items():
        try:
            all_trades = strat_fn(df.sort_values("date").reset_index(drop=True))
        except Exception:
            continue
        # entries in window only (indicators warm up on pre-window bars)
        in_win = [
            t for t in all_trades
            if pd.Timestamp(t.entry_date) >= window_start
            and pd.Timestamp(t.entry_date) <= window_end
        ]
        if in_win:
            trades_by_ticker[ticker] = in_win
            n_trades += len(in_win)
    return trades_by_ticker, n_trades


def _evaluate_one(
    name: str,
    strat_fn: Callable,
    ohlcv: dict[str, pd.DataFrame],
    bench_df: pd.DataFrame | None,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    n_trials: int,
    slot_count: int = DEFAULT_SLOTS,
) -> dict:
    def _window(ws, we):
        trades_by_t, n_tr = _run_strategy(strat_fn, ohlcv, ws, we)
        eq = _basket_equity(ohlcv, trades_by_t, ws, we, slot_count=slot_count)
        if bench_df is not None:
            b = bench_df.sort_values("date")
            b = b[(pd.to_datetime(b["date"]) >= ws) & (pd.to_datetime(b["date"]) <= we)]
            bench = pd.Series(
                b["adj_close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(b["date"]),
            )
        else:
            bench = pd.Series(dtype=float)
        flat: list[_MetricTrade] = []
        for trs in trades_by_t.values():
            flat.extend(_to_metric_trades(trs))
        if eq.empty:
            return {"trade_count": 0}
        m = compute_metrics(eq, bench, flat, slot_count=max(len(ohlcv), 1), n_trials=n_trials)
        m["trade_count"] = n_tr
        return m

    is_m = _window(is_start, is_end)
    oos_m = _window(oos_start, oos_end)

    dsr_oos = float(oos_m.get("dsr", 0.0) or 0.0)
    mdd_oos = float(oos_m.get("max_drawdown", 0.0) or 0.0)
    score = dsr_oos * math.exp(-abs(mdd_oos) / DD_TOLERANCE)

    return {
        "strategy": name,
        "score": score,
        "is": is_m,
        "oos": oos_m,
    }


# ───────────────────── universe + benchmark caches ─────────────────────────

_OHLCV_CACHE: dict[tuple[str, int, int], dict[str, pd.DataFrame]] = {}
_BENCH_CACHE: dict[tuple[str, int, int], pd.DataFrame | None] = {}


def _load(
    market: str, years: int, limit: int, refresh: bool
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None, pd.Timestamp, pd.Timestamp]:
    today = date.today()
    oos_end = pd.Timestamp(today).normalize()
    # warmup pad = 4y (covers ema600 ≈ 2.4y)
    fetch_start = (oos_end - pd.DateOffset(years=years + 4)).date()
    fetch_end = today

    ck = (market, years, limit)
    if ck not in _OHLCV_CACHE:
        tickers = load_universe(market, None)
        if limit and limit < len(tickers):
            tickers = tickers[:limit]
        print(f"[load] universe {market}: {len(tickers)} tickers", file=sys.stderr)
        ohlcv: dict[str, pd.DataFrame] = {}

        def _fetch(t: str):
            df = _cached_fetch(t, fetch_start, fetch_end, market, refresh=refresh)
            return t, df

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_fetch, t): t for t in tickers}
            for i, fut in enumerate(as_completed(futs), 1):
                t, df = fut.result()
                if df is not None and not df.empty:
                    ohlcv[t] = df
                if i % 100 == 0 or i == len(tickers):
                    print(f"  fetched {i}/{len(tickers)} ({len(ohlcv)} with data)",
                          file=sys.stderr, flush=True)
        _OHLCV_CACHE[ck] = ohlcv

        bench_sym = BENCHMARKS[market]
        _BENCH_CACHE[ck] = _cached_fetch(bench_sym, fetch_start, fetch_end, market, refresh=refresh)

    ohlcv = _OHLCV_CACHE[ck]
    bench = _BENCH_CACHE[ck]
    return ohlcv, bench, oos_end, pd.Timestamp(fetch_start)


# ───────────────────── CLI ─────────────────────────────────────────────────

@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--market", type=click.Choice(["us", "india"]), default="us")
@click.option("--years", type=int, default=3, help="Total window length (IS + OOS).")
@click.option("--oos-years", type=float, default=1.0, help="OOS length (years).")
@click.option("--limit", type=int, default=150, help="Universe cap.")
@click.option("--slots", type=int, default=DEFAULT_SLOTS,
              help="Concurrent-position slots. Basket weights = 1/max(held, slots).")
@click.option("--refresh", is_flag=True, help="Force OHLCV re-fetch (ignores parquet cache).")
@click.option("--iteration", type=int, default=0, help="Iteration index (for journal).")
@click.option("--hypothesis", type=str, default="", help="One-line hypothesis from agent.")
@click.option("--journal", type=click.Path(), default=str(JOURNAL_PATH))
def evaluate(
    market: str,
    years: int,
    oos_years: float,
    limit: int,
    slots: int,
    refresh: bool,
    iteration: int,
    hypothesis: str,
    journal: str,
) -> None:
    """Evaluate all strategies on IS+OOS and append results to the journal."""
    t0 = time.time()
    ohlcv, bench_df, oos_end, _fetch_start = _load(market, years, limit, refresh)
    oos_start = oos_end - pd.DateOffset(years=oos_years)
    is_end = oos_start - pd.Timedelta(days=1)
    is_start = oos_end - pd.DateOffset(years=years)

    strategies: dict[str, Callable] = {}
    for k, v in STOCK_STRATEGIES.items():
        strategies[f"stock:{k}"] = v
    for k, v in NEW_STRATEGIES.items():
        strategies[f"new:{k}"] = v

    print(f"[eval] iteration={iteration}  strategies={len(strategies)}  slots={slots}  "
          f"IS {is_start.date()}→{is_end.date()}  OOS {oos_start.date()}→{oos_end.date()}",
          file=sys.stderr)

    n_trials = len(strategies)
    results: list[dict] = []
    for name, fn in strategies.items():
        r = _evaluate_one(name, fn, ohlcv, bench_df, is_start, is_end,
                          oos_start, oos_end, n_trials=n_trials, slot_count=slots)
        results.append(r)
        oos = r["oos"]
        ret = float(oos.get("total_return", 0.0) or 0.0)
        dsr = float(oos.get("dsr", 0.0) or 0.0)
        sor = float(oos.get("sortino", 0.0) or 0.0)
        mdd = float(oos.get("max_drawdown", 0.0) or 0.0)
        ntr = int(oos.get("trade_count", 0) or 0)
        print(f"  {name:<28} ret={ret:+7.1%}  dsr={dsr:4.2f}  sortino={sor:+5.2f}  "
              f"mdd={mdd:+6.1%}  trades={ntr:>5}  score={r['score']:+6.3f}",
              file=sys.stderr)

    results.sort(key=lambda r: r["score"], reverse=True)
    elapsed = time.time() - t0

    # write one journal line per strategy so all 20 iterations × N strategies
    # are fully retained
    ts = pd.Timestamp.now().isoformat()
    commit = _git_head()
    diff = _git_diff_stat()
    with open(journal, "a") as f:
        for r in results:
            f.write(json.dumps({
                "ts": ts,
                "iteration": iteration,
                "hypothesis": hypothesis,
                "git_head": commit,
                "git_diff_stat": diff,
                "market": market,
                "slots": slots,
                "is_window": [str(is_start.date()), str(is_end.date())],
                "oos_window": [str(oos_start.date()), str(oos_end.date())],
                "n_trials": n_trials,
                "strategy": r["strategy"],
                "score": r["score"],
                "is": _safe(r["is"]),
                "oos": _safe(r["oos"]),
            }) + "\n")

    print(f"\n[eval] done in {elapsed:.1f}s — top 5 by OOS score:", file=sys.stderr)
    for r in results[:5]:
        print(f"  {r['strategy']:<28} score={r['score']:+6.3f}  "
              f"dsr={float(r['oos'].get('dsr', 0) or 0):.2f}  "
              f"ret={float(r['oos'].get('total_return', 0) or 0):+.1%}", file=sys.stderr)


def _safe(d: dict) -> dict:
    """Make metric dict JSON-safe (convert NaN/inf to None)."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def _git_diff_stat() -> str:
    try:
        r = subprocess.run(["git", "diff", "--stat", "--", "autoresearch_strategies.py"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    cli()
