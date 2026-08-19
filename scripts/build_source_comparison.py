"""Cross-source comparison: yfinance vs FMP baselines and tuned configs.

Reads the yfinance study results.csv (baseline 5/3/2/1y), the FMP sweep
baseline 3y, and the FMP validation (baseline + tuned 5/2/1y) to produce
findings/RESEARCH_SOURCE_COMPARISON.md showing how much the price source
moves the numbers, and what the levers add on FMP.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "findings" / "research_study"
REPORT = ROOT / "findings" / "RESEARCH_SOURCE_COMPARISON.md"

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
    if v in (None, ""):
        return "—"
    return f"{float(str(v).replace('%', '')):+.1f}%"


def load(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def main() -> None:
    yf_study = load(OUT / "results.csv")  # yfinance baseline, 5/3/2/1y
    fmp_sweep = load(OUT / "sweep_results_fmp.csv")  # FMP baseline + combos, 3y
    fmp_val = load(OUT / "sweep_validate_fmp.csv")  # FMP baseline + tuned, 5/2/1y

    lines = [
        "# Data-Source Comparison — yfinance vs FMP\n",
        "Baseline (no levers) numbers for the 15 strategies, same universes, same costs, "
        "same windows — only the price provider changes (yfinance auto-adjusted OHLCV vs "
        "FMP historical-price-full adjClose). The last column shows the tuned lever "
        "config on FMP. Research, not financial advice.\n",
    ]
    for market, label in (("india", "India (Nifty 500)"), ("us", "US (S&P 500)")):
        lines.append(f"\n## {label}\n")
        lines.append(
            "| Strategy | 3y yf CAGR | 3y FMP CAGR | 5y yf | 5y FMP base | 5y FMP tuned | 1y yf | 1y FMP base | 1y FMP tuned |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in STRATEGY_ORDER:

            def cag(rows, years, kind=None, sweep=False):
                for r in rows:
                    if r["strategy"] != s or r["market"] != market:
                        continue
                    if not sweep and int(r["years"]) != years:
                        continue
                    if sweep and not (
                        r.get("regime") == "none"
                        and r.get("sl") == "none"
                        and r.get("tp") == "none"
                        and r.get("trail") == "none"
                    ):
                        continue
                    if kind is None or r.get("kind") == kind:
                        return r.get("cagr", r.get("CAGR"))
                return None

            lines.append(
                f"| `{s}` | {pct(cag(yf_study, 3))} | {pct(cag(fmp_sweep, 3, sweep=True))} | "
                f"{pct(cag(yf_study, 5))} | {pct(cag(fmp_val, 5, 'baseline'))} | "
                f"{pct(cag(fmp_val, 5, 'tuned'))} | {pct(cag(yf_study, 1))} | "
                f"{pct(cag(fmp_val, 1, 'baseline'))} | {pct(cag(fmp_val, 1, 'tuned'))} |"
            )
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
