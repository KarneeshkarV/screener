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

RUNS: list[tuple[str, str, str, int]] = [
    # (market, as_of, criteria, hold_days)
    # OHLCV-only — go back 2 years so there is a full forward window
    ("us",    "2023-04-15", "ema",           252),
    ("us",    "2023-04-15", "breakout",       252),
    ("us",    "2023-04-15", "ema_breakout",   252),
    ("us",    "2024-04-15", "ema",           252),
    ("us",    "2024-04-15", "breakout",       252),
    ("us",    "2024-04-15", "ema_breakout",   252),
    # Fundamentals — yfinance quarterly data only covers ~last 5 quarters,
    # so use a recent as_of where 2+ quarters fall inside the 45-day lag window.
    # as_of 2025-09-01: Q4-2024 (avail 2025-02-14) and Q1-2025 (avail 2025-05-15) ✓
    # hold=180 keeps exit well within available data (exits ~2026-03)
    ("us",    "2025-09-01", "value",         180),
    ("us",    "2025-09-01", "quality",        180),
    ("us",    "2025-09-01", "cheap_quality",  180),
    ("us",    "2025-09-01", "dividend",       180),
    ("us",    "2025-09-01", "momentum_value", 180),
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
