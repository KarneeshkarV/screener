"""Run a matrix of historical backtests: market × date × criteria.

Calls run_historical_backtest() directly (no subprocess) and prints a compact
results table.

Usage:
    uv run python run_matrix.py
"""
from __future__ import annotations

import sys
from datetime import date

import click

# Suppress Rich / click echo noise from the library while we run
import warnings
warnings.filterwarnings("ignore")

from screener.historical_backtest import run_historical_backtest

# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

# Single filters + interesting combos (use "+" to combine).
# Layered technical+fundamental combos are informed by multi-factor research
# (value+momentum, quality+trend, cheap-quality composite).
_CRITERIA = [
    # --- pure technical (baseline for comparison) ---
    "ema", "breakout", "pullback", "oversold_rsi",
    "ema+breakout",           # EMA stack + near-52w-high
    "ema+pullback",           # EMA stack + dip to EMA20
    "pullback+golden_cross",  # buy-the-dip right after golden cross
    "ema+breakout+pullback",  # triple: trend + strength + dip

    # --- pure fundamental composites ---
    "cheap_quality",          # Greenblatt-style: value + quality + trend-lite
    "momentum_value",         # value + RSI 50-70 + EMA stack

    # --- layered technical + fundamental (research-backed) ---
    "value+ema",              # cheap stock in uptrend
    "quality+ema",             # high-ROE in uptrend
    "quality+breakout",        # high-ROE breaking out
    "cheap_quality+breakout",  # best composite + momentum
    "value+oversold_rsi",      # buy-the-dip on cheap names
]
# (years, as_of, hold_days) — each as_of chosen so exit ≈ today (2026-04-15).
_HOLDS = [
    (1, "2025-04-15", 252),
    (3, "2023-04-17", 756),
    (5, "2021-04-15", 1260),
]

RUNS: list[tuple[str, str, str, int]] = [
    (mkt, as_of, crit, hold)
    for mkt in ("us", "india")
    for _years, as_of, hold in _HOLDS
    for crit in _CRITERIA
]

TOP = 20

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _fmt(v, pct=False, decimals=2):
    if v is None or v != v:  # None or NaN
        return "-"
    if pct:
        return f"{v:+.1%}"
    return f"{v:.{decimals}f}"

HDR = (
    f"{'Mkt':<6} {'As-of':<12} {'Criteria':<16} "
    f"{'Match':>5} {'Top':>4} "
    f"{'Basket':>8} {'Bench':>8} {'Alpha':>7} "
    f"{'Sharpe':>7} {'MaxDD':>7} {'Hit%':>6}"
)
SEP = "-" * len(HDR)

rows = []

for market, as_of_str, criteria, hold in RUNS:
    as_of = date.fromisoformat(as_of_str)
    label = f"{market.upper()} {as_of_str} {criteria}"
    print(f"{label}...", file=sys.stderr, flush=True)

    try:
        # Redirect click.echo to stderr so it doesn't pollute stdout
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
            "err": str(e)[:60],
        })
        print(f"  FAILED: {e}", file=sys.stderr, flush=True)

print()
print("=" * (len(HDR) + 2))
print(HDR)
print(SEP)
for r in rows:
    if r.get("err"):
        print(
            f"{r['mkt']:<6} {r['as_of']:<12} {r['crit']:<16} "
            f"  ERR  {r['err']}"
        )
    else:
        print(
            f"{r['mkt']:<6} {r['as_of']:<12} {r['crit']:<16} "
            f"{r['matches']:>5} {r['top']:>4} "
            f"{_fmt(r['basket'], pct=True):>8} "
            f"{_fmt(r['bench'],  pct=True):>8} "
            f"{_fmt(r['alpha'],  pct=True):>7} "
            f"{_fmt(r['sharpe']):>7} "
            f"{_fmt(r['mdd'],    pct=True):>7} "
            f"{_fmt(r['hit'],    pct=True):>6}"
        )
print()
