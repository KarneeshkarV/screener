"""Matrix: enter on a technical criterion, exit when that criterion is no
longer valid.

One row per (market, criterion).  Each row maps the entry criterion to its
matching invalidation exit signal and runs `run_historical_backtest` — same
style as run_matrix.py.

Usage:
    uv run python run_exit_invalid_matrix.py
"""
from __future__ import annotations

import sys
from datetime import date

import warnings
warnings.filterwarnings("ignore")

from screener.historical_backtest import run_historical_backtest


# Entry criterion -> invalidation exit signal.  Each exit fires when the
# criterion's defining state no longer holds (see historical_exits.py).
CRITERION_TO_EXIT: dict[str, str] = {
    "ema":           "ema_stack_full_break",
    "breakout":      "breakout_invalid",
    "ema_breakout":  "ema_breakout_invalid",
    "pullback":      "pullback_invalid",
    "oversold_rsi":  "oversold_rsi_invalid",
    "golden_cross":  "golden_cross_invalid",
    "rsi2_oversold": "rsi2_oversold_invalid",
    "macd_cross":    "macd_cross_invalid",
    "sma_cross":     "sma_cross_invalid",
}

AS_OF = date(2024, 4, 15)
HOLD_DAYS = 252
TOP = 20

MARKETS = ("us", "india")


def _fmt(v, pct=False, decimals=2):
    if v is None or v != v:  # None or NaN
        return "-"
    if pct:
        return f"{v:+.1%}"
    return f"{v:.{decimals}f}"


HDR = (
    f"{'Mkt':<6} {'Criteria':<15} {'Exit':<22} "
    f"{'Match':>5} {'Top':>4} "
    f"{'Basket':>8} {'Bench':>8} {'Alpha':>7} "
    f"{'Sharpe':>7} {'MaxDD':>7} {'Hit%':>6} {'AvgDays':>8}"
)
SEP = "-" * len(HDR)


def main() -> None:
    rows: list[dict] = []
    runs = [(m, c) for m in MARKETS for c in CRITERION_TO_EXIT]

    for market, criteria in runs:
        exit_sig = CRITERION_TO_EXIT[criteria]
        label = f"{market.upper()} {criteria:<15} -> exit on {exit_sig}"
        print(f"{label}...", file=sys.stderr, flush=True)
        try:
            result = run_historical_backtest(
                market=market,
                criteria_name=criteria,
                as_of=AS_OF,
                hold_days=HOLD_DAYS,
                top=TOP,
                universe_path=None,
                benchmark_override=None,
                refresh=False,
                exit_signals=(exit_sig,),
            )
            b = result.summary["basket"]
            bm = result.summary["benchmark"]
            pt = result.per_ticker
            avg_days = float(pt["trading_days"].mean()) if not pt.empty else float("nan")
            rows.append({
                "mkt": market.upper(), "crit": criteria, "exit": exit_sig,
                "matches": result.matches_total, "top": len(result.tickers),
                "basket": b["total_return"], "bench": bm["total_return"],
                "alpha": b["alpha"], "sharpe": b["sharpe"],
                "mdd": b["max_drawdown"], "hit": b["hit_rate"],
                "avg_days": avg_days,
                "err": None,
            })
            print(
                f"  basket={b['total_return']:+.1%}  bench={bm['total_return']:+.1%}  "
                f"alpha={_fmt(b['alpha'], pct=True)}  sharpe={b['sharpe']:.2f}  "
                f"days={avg_days:.1f}",
                file=sys.stderr, flush=True,
            )
        except Exception as e:  # noqa: BLE001
            rows.append({
                "mkt": market.upper(), "crit": criteria, "exit": exit_sig,
                "err": str(e)[:60],
            })
            print(f"  FAILED: {e}", file=sys.stderr, flush=True)

    print()
    print("=" * (len(HDR) + 2))
    print(f"Exit-on-invalidation matrix  as_of={AS_OF}  hold={HOLD_DAYS}  top={TOP}")
    print("=" * (len(HDR) + 2))
    print(HDR)
    print(SEP)
    for r in rows:
        if r.get("err"):
            print(
                f"{r['mkt']:<6} {r['crit']:<15} {r['exit']:<22}   ERR  {r['err']}"
            )
        else:
            print(
                f"{r['mkt']:<6} {r['crit']:<15} {r['exit']:<22} "
                f"{r['matches']:>5} {r['top']:>4} "
                f"{_fmt(r['basket'], pct=True):>8} "
                f"{_fmt(r['bench'],  pct=True):>8} "
                f"{_fmt(r['alpha'],  pct=True):>7} "
                f"{_fmt(r['sharpe']):>7} "
                f"{_fmt(r['mdd'],    pct=True):>7} "
                f"{_fmt(r['hit'],    pct=True):>6} "
                f"{_fmt(r['avg_days'], decimals=1):>8}"
            )
    print()


if __name__ == "__main__":
    main()
