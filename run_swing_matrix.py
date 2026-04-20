"""Short-term swing-trading matrix: 10-day hold across 12 monthly windows.

Runs every (market, as_of, criteria) combination and then prints a
per-(market, criteria) aggregation averaging basket return, alpha, Sharpe
and hit-rate across the rolling windows — the real deliverable for
comparing short-term strategies between India and US.

Usage:
    uv run python run_swing_matrix.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from statistics import mean

from screener.historical_backtest import run_historical_backtest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

# 12 rolling entry dates (roughly mid-month) covering the last year of both
# markets. Each is held for 10 trading days (≈ 2 weeks).
_AS_OF_DATES = [
    "2025-04-15", "2025-05-15", "2025-06-16", "2025-07-15",
    "2025-08-15", "2025-09-15", "2025-10-15", "2025-11-17",
    "2025-12-15", "2026-01-15", "2026-02-16", "2026-03-16",
]
HOLD_DAYS = 10
TOP = 20

_CRITERIA = [
    # Existing short-term friendly setups (sanity baselines).
    "pullback", "oversold_rsi", "breakout", "ema+pullback",
    # New additions driven by 2-week swing research.
    "rsi2_oversold", "bb_bounce", "macd_cross", "sma_cross",
    # One combo to test layering a momentum flip with a pullback filter.
    "macd_cross+pullback",
]

RUNS: list[tuple[str, str, str, int]] = [
    (mkt, as_of, crit, HOLD_DAYS)
    for mkt in ("us", "india")
    for as_of in _AS_OF_DATES
    for crit in _CRITERIA
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _fmt(v, pct=False, decimals=2):
    if v is None or v != v:
        return "-"
    if pct:
        return f"{v:+.1%}"
    return f"{v:.{decimals}f}"


PER_RUN_HDR = (
    f"{'Mkt':<4} {'As-of':<12} {'Criteria':<22} "
    f"{'Match':>5} {'Top':>4} "
    f"{'Basket':>8} {'Bench':>8} {'Alpha':>7} "
    f"{'Sharpe':>7} {'MaxDD':>7} {'Hit%':>6}"
)
PER_RUN_SEP = "-" * len(PER_RUN_HDR)

AGG_HDR = (
    f"{'Mkt':<4} {'Criteria':<22} "
    f"{'AvgBasket':>10} {'AvgAlpha':>9} "
    f"{'AvgSharpe':>10} {'AvgHit%':>8} {'N':>3}"
)
AGG_SEP = "-" * len(AGG_HDR)

rows: list[dict] = []

total = len(RUNS)
for i, (market, as_of_str, criteria, hold) in enumerate(RUNS, 1):
    as_of = date.fromisoformat(as_of_str)
    label = f"[{i}/{total}] {market.upper()} {as_of_str} {criteria}"
    print(f"{label}...", file=sys.stderr, flush=True)

    try:
        result = run_historical_backtest(
            market=market,
            criteria_name=criteria,
            as_of=as_of,
            hold_days=hold,
            top=TOP,
            universe_path=None,
            benchmark_override=None,
            refresh=False,
        )
        b = result.summary["basket"]
        bm = result.summary["benchmark"]
        rows.append({
            "mkt": market.upper(), "as_of": as_of_str, "crit": criteria,
            "matches": result.matches_total, "top": len(result.tickers),
            "basket": b["total_return"], "bench": bm["total_return"],
            "alpha": b["alpha"], "sharpe": b["sharpe"],
            "mdd": b["max_drawdown"], "hit": b["hit_rate"],
            "err": None,
        })
        print(
            f"  basket={b['total_return']:+.1%}  bench={bm['total_return']:+.1%}  "
            f"alpha={b['alpha']:+.1%}  sharpe={b['sharpe']:.2f}",
            file=sys.stderr, flush=True,
        )
    except Exception as e:
        rows.append({
            "mkt": market.upper(), "as_of": as_of_str, "crit": criteria,
            "err": str(e)[:80],
        })
        print(f"  FAILED: {e}", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Per-run table
# ---------------------------------------------------------------------------
print()
print("=" * (len(PER_RUN_HDR) + 2))
print("PER-RUN RESULTS (10-day hold)")
print(PER_RUN_HDR)
print(PER_RUN_SEP)
for r in rows:
    if r.get("err"):
        print(
            f"{r['mkt']:<4} {r['as_of']:<12} {r['crit']:<22} "
            f"  ERR  {r['err']}"
        )
    else:
        print(
            f"{r['mkt']:<4} {r['as_of']:<12} {r['crit']:<22} "
            f"{r['matches']:>5} {r['top']:>4} "
            f"{_fmt(r['basket'], pct=True):>8} "
            f"{_fmt(r['bench'],  pct=True):>8} "
            f"{_fmt(r['alpha'],  pct=True):>7} "
            f"{_fmt(r['sharpe']):>7} "
            f"{_fmt(r['mdd'],    pct=True):>7} "
            f"{_fmt(r['hit'],    pct=True):>6}"
        )

# ---------------------------------------------------------------------------
# Aggregation: average across rolling windows per (market, criteria)
# ---------------------------------------------------------------------------
groups: dict[tuple[str, str], list[dict]] = {}
for r in rows:
    if r.get("err"):
        continue
    groups.setdefault((r["mkt"], r["crit"]), []).append(r)


def _avg(items: list[dict], key: str) -> float | None:
    vals = [x[key] for x in items if x.get(key) is not None and x[key] == x[key]]
    return mean(vals) if vals else None


print()
print("=" * (len(AGG_HDR) + 2))
print(f"AGGREGATION: average of {HOLD_DAYS}-day holds across rolling windows")
print(AGG_HDR)
print(AGG_SEP)

agg_rows = []
for (mkt, crit), items in groups.items():
    agg_rows.append({
        "mkt": mkt, "crit": crit, "n": len(items),
        "avg_basket": _avg(items, "basket"),
        "avg_alpha":  _avg(items, "alpha"),
        "avg_sharpe": _avg(items, "sharpe"),
        "avg_hit":    _avg(items, "hit"),
    })

agg_rows.sort(key=lambda r: (r["mkt"], -(r["avg_alpha"] or -9.9)))
for r in agg_rows:
    print(
        f"{r['mkt']:<4} {r['crit']:<22} "
        f"{_fmt(r['avg_basket'], pct=True):>10} "
        f"{_fmt(r['avg_alpha'],  pct=True):>9} "
        f"{_fmt(r['avg_sharpe']):>10} "
        f"{_fmt(r['avg_hit'],    pct=True):>8} "
        f"{r['n']:>3}"
    )
print()
