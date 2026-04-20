"""Exit-combo sweep for backtest-historical.

Runs the heavy screening ONCE per market (cached OHLCV, rank top-20), then
iterates the exit-policy grid calling ``_forward_test_from_matrix`` directly
so the expensive screening + ranking + matrix build doesn't repeat.

Grid (per market):
    stop-loss:     [None, 0.05, 0.08, 0.12]          # 4
    take-profit:   [None, 0.15, 0.20, 0.30]          # 4
    trailing-stop: [None, 0.08, 0.12, 0.20]          # 4
    exit-signal:   [None, ema20_break, ema_stack_break,
                    macd_cross_down, rsi_overbought, bb_upper_tag]  # 6
= 384 combos; drop the all-None row -> 383 sweep rows + 1 baseline row.

Writes results to _exit_combo_sweep.txt (sorted by Sharpe desc per market)
with the no-exit baseline at the top of each market's section.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Quiet noisy third-party warnings
import warnings
warnings.filterwarnings("ignore")

from screener.backtest import BENCHMARKS, _forward_test_from_matrix
from screener.historical_criteria import HIST_CRITERIA, compute_scores
from screener.historical_exits import ExitPolicy
from screener.prices import fetch_ohlcv
from screener.sectors import bullish_sectors, sector_map
from screener.universes import load_universe


AS_OF = date(2024, 4, 15)
HOLD_DAYS = 252
TOP_N = 20
CRITERIA_NAME = "ema"

STOP_VALUES    = [None, 0.05, 0.08, 0.12]
TAKE_VALUES    = [None, 0.15, 0.20, 0.30]
TRAIL_VALUES   = [None, 0.08, 0.12, 0.20]
SIGNAL_VALUES  = [None, "ema20_break", "ema_stack_break",
                  "macd_cross_down", "rsi_overbought", "bb_upper_tag"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def prepare_market(market: str) -> dict:
    """Screening preamble: fetch, screen, rank, build forward matrix.

    Returns a dict carrying everything ``_forward_test_from_matrix`` needs so
    downstream combos reuse it.
    """
    _log(f"\n=== Preparing market={market} ===")
    eval_fn = HIST_CRITERIA[CRITERIA_NAME]
    as_of_ts = pd.Timestamp(AS_OF).normalize()
    fetch_start = (as_of_ts - pd.DateOffset(years=2)).date()
    fetch_end = (as_of_ts + pd.DateOffset(days=int(HOLD_DAYS * 1.5) + 30)).date()

    all_tickers = load_universe(market, None)
    _log(f"Universe size: {len(all_tickers)}")

    # Fetch OHLCV for every ticker (cache hits should make this quick).
    ohlcv_data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i, t in enumerate(all_tickers, 1):
        df = fetch_ohlcv(t, fetch_start, fetch_end, market, refresh=False)
        if df is None or df.empty:
            failed.append(t)
        else:
            ohlcv_data[t] = df
        if i % 100 == 0 or i == len(all_tickers):
            _log(f"  fetched {i}/{len(all_tickers)} (failed {len(failed)})")

    # Screen — evaluate ema criterion at as_of.
    matches: list[tuple[str, dict]] = []
    for ticker, df in ohlcv_data.items():
        result = eval_fn(df, as_of_ts, None)
        if result is not None and result["passes"]:
            matches.append((ticker, result))
    _log(f"Matches: {len(matches)} / {len(ohlcv_data)}")
    if not matches:
        raise RuntimeError(f"No matches for {market}")

    # Bull-sector filter (matches run_historical_backtest default).
    smap = sector_map(market, list(ohlcv_data.keys()))
    bull_set, sector_mean = bullish_sectors(smap, ohlcv_data, as_of_ts)
    if bull_set:
        before = len(matches)
        matches = [(t, r) for t, r in matches if smap.get(t) in bull_set]
        _log(f"Bull-sector filter: kept {len(matches)}/{before}")

    if not matches:
        raise RuntimeError(f"All matches filtered out by bull-sector gate for {market}")

    # Rank by score, take top-N.
    score_list = compute_scores([r["score_inputs"] for _, r in matches])
    ranked = sorted(
        zip([t for t, _ in matches], score_list),
        key=lambda x: x[1],
        reverse=True,
    )
    selected = [t for t, _ in ranked[:TOP_N]]
    _log(f"Selected top {len(selected)}: {', '.join(selected)}")

    # Entry date: first trading day strictly after as_of.
    entry_date: Optional[date] = None
    for t in selected:
        df = ohlcv_data[t]
        dd = pd.to_datetime(df["date"]).dt.normalize()
        after = dd[dd > as_of_ts]
        if not after.empty:
            candidate = after.iloc[0].date()
            if entry_date is None or candidate < entry_date:
                entry_date = candidate
    if entry_date is None:
        raise RuntimeError(f"No trading day after {AS_OF} for {market}")

    # Wide forward matrix (adj_close per ticker, aligned on trading days).
    series: dict[str, pd.Series] = {}
    for t in selected:
        df = ohlcv_data[t].copy()
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        forward = df[df["_date"].dt.date >= entry_date]
        if forward.empty:
            continue
        series[t] = pd.Series(
            forward["adj_close"].values,
            index=forward["_date"].values,
            name=t,
        )
    wide = pd.DataFrame(series).sort_index()
    _log(f"Forward matrix: {wide.shape[0]} rows × {wide.shape[1]} tickers  entry={entry_date}")

    return {
        "market": market,
        "wide": wide,
        "ohlcv_full": {t: ohlcv_data[t] for t in selected if t in ohlcv_data},
        "benchmark": BENCHMARKS.get(market, "AMEX:SPY"),
        "selected": selected,
        "entry_date": entry_date,
    }


def run_combo(ctx: dict, policy: ExitPolicy) -> dict:
    """Run a single exit-policy combo against the cached wide matrix."""
    ft = _forward_test_from_matrix(
        ctx["wide"],
        HOLD_DAYS,
        ctx["benchmark"],
        ctx["market"],
        refresh=False,
        exit_policy=policy,
        ohlcv_full=ctx["ohlcv_full"],
    )
    basket = ft["summary"]["basket"]
    pt = ft["per_ticker"]
    mean_days = float(pt["trading_days"].mean()) if not pt.empty else float("nan")
    return {
        "total_return": basket["total_return"],
        "sharpe": basket["sharpe"],
        "max_drawdown": basket["max_drawdown"],
        "hit_rate": basket["hit_rate"],
        "mean_days": mean_days,
    }


def build_grid() -> list[tuple[Optional[float], Optional[float], Optional[float], Optional[str]]]:
    grid = []
    for s in STOP_VALUES:
        for t in TAKE_VALUES:
            for tr in TRAIL_VALUES:
                for sig in SIGNAL_VALUES:
                    if s is None and t is None and tr is None and sig is None:
                        continue
                    grid.append((s, t, tr, sig))
    return grid


def _policy(s, t, tr, sig) -> ExitPolicy:
    return ExitPolicy(
        stop_loss=s,
        take_profit=t,
        trailing_stop=tr,
        signals=((sig,) if sig else ()),
        time_stop_bars=HOLD_DAYS,
    )


def _fmt_val(v: Optional[float], pct: bool = False) -> str:
    if v is None:
        return "none"
    if pct:
        return f"{v:.0%}"
    return str(v)


def combo_label(s, t, tr, sig) -> str:
    return (
        f"stop={_fmt_val(s, pct=True)}|"
        f"tp={_fmt_val(t, pct=True)}|"
        f"trail={_fmt_val(tr, pct=True)}|"
        f"sig={sig or 'none'}"
    )


def sweep_market(market: str) -> tuple[dict, list[dict]]:
    ctx = prepare_market(market)

    _log(f"\nRunning baseline (no-exit) for {market}…")
    baseline_policy = ExitPolicy(time_stop_bars=HOLD_DAYS)  # noop
    baseline = run_combo(ctx, baseline_policy)
    baseline["combo"] = "baseline (no exits)"

    grid = build_grid()
    _log(f"Sweeping {len(grid)} combos for {market}…")
    rows: list[dict] = []
    t0 = time.time()
    for i, (s, t, tr, sig) in enumerate(grid, 1):
        policy = _policy(s, t, tr, sig)
        try:
            metrics = run_combo(ctx, policy)
        except Exception as e:  # noqa: BLE001
            _log(f"  combo {i} error: {e}")
            continue
        metrics["combo"] = combo_label(s, t, tr, sig)
        rows.append(metrics)
        if i % 50 == 0 or i == len(grid):
            elapsed = time.time() - t0
            _log(f"  {i}/{len(grid)} combos ({elapsed:.1f}s)")

    # Sort by Sharpe desc (NaNs last).
    rows.sort(key=lambda r: (r["sharpe"] if pd.notna(r["sharpe"]) else -1e9), reverse=True)
    return baseline, rows


def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "    -"
    return f"{v * 100:+6.2f}%"


def _fmt_num(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "    -"
    return f"{v:6.{digits}f}"


def write_results(path: Path, all_results: dict[str, tuple[dict, list[dict]]]) -> None:
    lines: list[str] = []
    header = (
        "EXIT-COMBO SWEEP — backtest-historical -c ema --as-of 2024-04-15 "
        "--hold 252 --top 20"
    )
    lines.append("=" * 100)
    lines.append(header)
    lines.append("Grid: stop×take×trail×signal = 4×4×4×6 = 384  (all-none excluded → 383 rows)")
    lines.append("=" * 100)

    col_header = (
        f"{'Combo':<58} "
        f"{'Total':>8} {'Sharpe':>7} {'MaxDD':>8} {'Hit%':>7} {'AvgDays':>8}"
    )

    for market, (baseline, rows) in all_results.items():
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"MARKET: {market.upper()}")
        lines.append("-" * 100)
        lines.append(col_header)
        lines.append("-" * 100)

        # Baseline first for easy comparison.
        lines.append(
            f"{'>> ' + baseline['combo']:<58} "
            f"{_fmt_pct(baseline['total_return'])} "
            f"{_fmt_num(baseline['sharpe']):>7} "
            f"{_fmt_pct(baseline['max_drawdown'])} "
            f"{_fmt_pct(baseline['hit_rate'])} "
            f"{_fmt_num(baseline['mean_days'], 1):>8}"
        )
        lines.append("-" * 100)

        for r in rows:
            lines.append(
                f"{r['combo']:<58} "
                f"{_fmt_pct(r['total_return'])} "
                f"{_fmt_num(r['sharpe']):>7} "
                f"{_fmt_pct(r['max_drawdown'])} "
                f"{_fmt_pct(r['hit_rate'])} "
                f"{_fmt_num(r['mean_days'], 1):>8}"
            )

    path.write_text("\n".join(lines) + "\n")
    _log(f"\nWrote {path}")


def main() -> None:
    out = Path(__file__).parent / "_exit_combo_sweep.txt"
    all_results: dict[str, tuple[dict, list[dict]]] = {}
    for market in ("us", "india"):
        baseline, rows = sweep_market(market)
        all_results[market] = (baseline, rows)
    write_results(out, all_results)

    # Surface top 5 + baseline to stdout for immediate consumption.
    print()
    print("=" * 100)
    print("SUMMARY — top 5 by Sharpe per market (with baseline)")
    print("=" * 100)
    for market, (baseline, rows) in all_results.items():
        print(f"\n-- {market.upper()} --")
        print(f"{'Combo':<58} {'Total':>8} {'Sharpe':>7} {'MaxDD':>8} {'Hit%':>7} {'AvgDays':>8}")
        print(
            f"{'>> ' + baseline['combo']:<58} "
            f"{_fmt_pct(baseline['total_return'])} "
            f"{_fmt_num(baseline['sharpe']):>7} "
            f"{_fmt_pct(baseline['max_drawdown'])} "
            f"{_fmt_pct(baseline['hit_rate'])} "
            f"{_fmt_num(baseline['mean_days'], 1):>8}"
        )
        for r in rows[:5]:
            print(
                f"{r['combo']:<58} "
                f"{_fmt_pct(r['total_return'])} "
                f"{_fmt_num(r['sharpe']):>7} "
                f"{_fmt_pct(r['max_drawdown'])} "
                f"{_fmt_pct(r['hit_rate'])} "
                f"{_fmt_num(r['mean_days'], 1):>8}"
            )


if __name__ == "__main__":
    main()
