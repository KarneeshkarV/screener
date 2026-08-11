"""Build the lever-sweep report (FMP data source) from the sweep/validate CSVs.

Usage: uv run python scripts/build_sweep_report.py
Writes findings/RESEARCH_LEVER_SWEEP_REPORT.md: per strategy x market, baseline
(no levers) vs best tuned config on the 3y sweep window, then tuned-vs-baseline
on 5/2/1y validation windows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "findings" / "research_study"
REPORT = ROOT / "findings" / "RESEARCH_LEVER_SWEEP_REPORT.md"

MARKET_LABEL = {"india": "India (Nifty 500)", "us": "US (S&P 500)"}

STRATEGY_ORDER = [
    "golden_cross_50_200",
    "fifty_two_week_high",
    "bll_trading_range_break",
    "keltner_breakout",
    "adx_trend",
    "long_term_reversal",
    "macd_signal_cross",
    "stochastic_cross",
    "connors_rsi2",
    "connors_rsi2_bull",
    "bollinger_mean_reversion",
    "williams_percent_r",
    "cci_reversion",
    "short_term_reversal",
    "turn_of_month",
]


def pct(v) -> str:
    if v is None or v == "":
        return "—"
    return f"{float(v):+.1f}%"


def num(v) -> str:
    if v is None or v == "":
        return "—"
    return f"{float(v):+.2f}"


def load(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def baseline(rows: list[dict], strategy: str, market: str) -> dict | None:
    for r in rows:
        if (
            r["strategy"] == strategy
            and r["market"] == market
            and r["regime"] == "none"
            and r["sl"] == "none"
            and r["tp"] == "none"
            and r["trail"] == "none"
            and r.get("sizing", "equal_slot") == "equal_slot"
        ):
            return r
    return None


def cfg_text(best: dict) -> str:
    parts = [f"regime={best['regime']}", f"SL {best['sl']}", f"TP {best['tp']}", f"trail {best['trail']}"]
    sizing = best.get("sizing", "equal_slot")
    if sizing != "equal_slot":
        parts.append(f"sizing={sizing}")
    return " ".join(parts)


def main() -> None:
    rows = load(OUT / "sweep_results_fmp.csv")
    best = json.loads((OUT / "sweep_best_fmp.json").read_text()) if (OUT / "sweep_best_fmp.json").exists() else {}
    validate_rows = load(OUT / "sweep_validate_fmp.csv")

    lines = [
        "# Lever Sweep Report — execution levers per strategy (FMP price data)\n",
        "**Method.** For every strategy x market, a 3-year window (ending 2026-08-11) "
        "was swept over regime filter (none / bull / bull+pullback), stop loss "
        "(none/8%/15%/25%), take profit (none/25%), trailing stop (none/15%/25%), and "
        "sizing (equal_slot + atr_risk/fixed_fraction/inverse_vol on the grid-best). "
        "Best config chosen by Sharpe with >= 8 trades. Price data: **FMP** "
        "(historical-price-full, dividend/split adjusted). Costs: India NSE statutory "
        "+ 10bps slippage; US flat 1bp + 5bps slippage. Tuned-vs-baseline validated on "
        "5/2/1y windows. Research, not financial advice.\n",
    ]

    for market in ("india", "us"):
        lines.append(f"\n## {MARKET_LABEL[market]}\n")
        lines.append(
            "| Strategy | Baseline 3y CAGR / Sharpe | Best 3y CAGR / Sharpe / MDD | Best config | Tuned 5y/2y/1y CAGR (baseline) |"
        )
        lines.append("|---|---|---|---|---|")
        for strategy in STRATEGY_ORDER:
            key = f"{market}/{strategy}"
            best_row = best.get(key)
            base = baseline(rows, strategy, market)
            base_cagr = pct(base["cagr"]) if base else "—"
            base_sharpe = num(base["sharpe"]) if base else "—"
            if best_row is None or best_row.get("sharpe") is None:
                lines.append(f"| `{strategy}` | {base_cagr} / {base_sharpe} | — | — | — |")
                continue
            best_cagr = pct(best_row["cagr"])
            best_sharpe = num(best_row["sharpe"])
            best_mdd = pct(best_row["max_drawdown"])
            cfg_txt = cfg_text(best_row)
            val_tuned = [r for r in validate_rows if r["strategy"] == strategy and r["market"] == market and r.get("kind") == "tuned"]
            val_base = [r for r in validate_rows if r["strategy"] == strategy and r["market"] == market and r.get("kind") == "baseline"]
            val_tuned.sort(key=lambda r: -int(r["years"]))
            val_base.sort(key=lambda r: -int(r["years"]))
            if val_tuned:
                cells = []
                for t, b in zip(val_tuned, val_base):
                    cells.append(f"{pct(t['cagr'])} ({pct(b['cagr'])})")
                val_txt = " / ".join(cells)
            else:
                val_txt = "—"
            lines.append(
                f"| `{strategy}` | {base_cagr} / {base_sharpe} | "
                f"{best_cagr} / {best_sharpe} / {best_mdd} | `{cfg_txt}` | {val_txt} |"
            )

    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
