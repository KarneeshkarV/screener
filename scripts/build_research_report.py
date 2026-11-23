"""Build the final research report markdown from findings/research_study/results.csv.

Usage: uv run python scripts/build_research_report.py
Reads results.csv (written incrementally by run_research_study.py) and writes
findings/RESEARCH_STRATEGY_REPORT.md with per-market comparison tables.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "findings" / "research_study" / "results.csv"
OUT_PATH = ROOT / "findings" / "RESEARCH_STRATEGY_REPORT.md"

COLUMNS = [
    "CAGR",
    "Sharpe",
    "Sortino",
    "Calmar",
    "Max Drawdown",
    "Hit Rate",
    "Avg Exposure",
    "Trades",
    "Benchmark Return",
]

STRATEGY_LINKS = {
    "golden_cross_50_200": "strategy_golden_cross_50_200.md",
    "fifty_two_week_high": "strategy_fifty_two_week_high.md",
    "connors_rsi2": "strategy_connors_rsi2.md",
    "connors_rsi2_bull": "strategy_connors_rsi2.md",
    "bollinger_mean_reversion": "strategy_bollinger_mean_reversion.md",
    "macd_signal_cross": "strategy_macd_signal_cross.md",
    "bll_trading_range_break": "strategy_bll_trading_range_break.md",
    "stochastic_cross": "strategy_stochastic_cross.md",
    "williams_percent_r": "strategy_williams_percent_r.md",
    "keltner_breakout": "strategy_keltner_breakout.md",
    "cci_reversion": "strategy_cci_reversion.md",
    "adx_trend": "strategy_adx_trend.md",
    "short_term_reversal": "strategy_short_term_reversal.md",
    "long_term_reversal": "strategy_long_term_reversal.md",
    "turn_of_month": "strategy_turn_of_month.md",
}

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

MARKETS = {"india": "India (Nifty 500)", "us": "US (S&P 500)"}


def load() -> list[dict[str, str]]:
    return list(csv.DictReader(CSV_PATH.open()))


def num(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "").strip().replace("%", "").replace(",", "")
    if not val or val in {"nan", "N/A", "-"}:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fmt(row: dict[str, str], key: str) -> str:
    n = num(row, key)
    if n is None:
        return "—"
    s = f"{n:+.2f}" if abs(n) < 100 else f"{n:+.0f}"
    if key == "Trades":
        return f"{int(n)}"
    if key == "Avg Exposure":
        return f"{n:.0f}%"
    if key in {"Max Drawdown", "Benchmark Return"}:
        return f"{n:+.1f}%"
    return s


def table(rows: list[dict[str, str]], horizon: int) -> str:
    lines = [
        f"### {horizon}-year window",
        "",
        "| Strategy | " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * (len(COLUMNS) + 1),
    ]
    for name in STRATEGY_ORDER:
        row = next(
            (r for r in rows if r["strategy"] == name and int(r["years"]) == horizon),
            None,
        )
        if row is None:
            continue
        cells = [fmt(row, c) for c in COLUMNS]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = [r for r in load() if r.get("status") == "ok"]
    if not rows:
        raise SystemExit("no results yet — run scripts/run_research_study.py first")

    parts: list[str] = []
    parts.append("# Research Strategy Backtest Report\n")
    parts.append(
        "**Methodology.** 15 research-backed strategies (see the linked per-strategy "
        "markdown files) backtested with `screener backtest-rolling` on **India "
        "Nifty 500** and **US S&P 500** over trailing 5/3/2/1-year windows ending "
        "2026-08-11. **Price data: yfinance** (auto-adjusted OHLCV) — the FMP-based "
        "lever sweep and cross-source comparison live in "
        "`RESEARCH_LEVER_SWEEP_REPORT.md` / `RESEARCH_SOURCE_COMPARISON.md`. "
        "Realistic costs: India — NSE statutory fee model "
        "(`--cost-model india`) + 10 bps slippage; US — flat 1 bp commission + 5 bps "
        "slippage. Portfolio size is strategy-dependent: trend followers run 10 slots "
        "with long holds (100-500 days), momentum oscillators 15 slots, mean reversion "
        "and reversal factors 20 slots with ~20-day holds. Initial capital 100,000. "
        "Benchmark: ^NSEI (India) / SPY (US). "
        "Survivorship caveat: universes are today's index members applied to history "
        "(no point-in-time membership). Research, not financial advice.\n"
    )

    for market, label in MARKETS.items():
        market_rows = [r for r in rows if r["market"] == market]
        parts.append(f"\n## {label}\n")
        for horizon in (5, 3, 2, 1):
            parts.append(table(market_rows, horizon))
            parts.append("")

    # Worst/best summary across all horizons
    parts.append("\n## Headline takeaways\n")
    for market, label in MARKETS.items():
        market_rows = [r for r in rows if r["market"] == market]
        parts.append(f"### {label}\n")
        best = max(market_rows, key=lambda r: num(r, "CAGR") or -999)
        worst = min(market_rows, key=lambda r: num(r, "CAGR") or 999)
        parts.append(
            f"- Best CAGR: **`{best['strategy']}`** {best['years']}y → "
            f"{fmt(best, 'CAGR')} (Sharpe {fmt(best, 'Sharpe')}, MaxDD {fmt(best, 'Max Drawdown')})"
        )
        parts.append(
            f"- Worst CAGR: **`{worst['strategy']}`** {worst['years']}y → "
            f"{fmt(worst, 'CAGR')} (Sharpe {fmt(worst, 'Sharpe')})"
        )
        for name in STRATEGY_ORDER:
            strat_rows = [r for r in market_rows if r["strategy"] == name]
            avg = sum(num(r, "CAGR") or 0 for r in strat_rows) / len(strat_rows)
            cagars = " / ".join(
                fmt(r, "CAGR")
                for r in sorted(strat_rows, key=lambda r: -int(r["years"]))
            )
            parts.append(f"- `{name}` average CAGR {avg:+.1f}% — 5/3/2/1y: {cagars}")

    OUT_PATH.write_text("\n".join(parts))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
