"""Run the research-strategy backtest study (India nifty500 + US sp500, 5/3/2/1y).

For each strategy × market × horizon it runs `screener backtest-rolling` with
realistic costs and per-strategy sizing, captures the agent-mode digest, parses
headline metrics, and writes:

  findings/research_study/<market>__<strategy>__<horizon>y.log   full digest
  findings/research_study/results.csv                             parsed metrics
  findings/research_study/results.json                            same, JSON

Run from the repo root:  uv run python scripts/run_research_study.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "findings" / "research_study"
OUT.mkdir(parents=True, exist_ok=True)

# market -> (cli market, universe, cost flags)
MARKETS = {
    "india": {
        "universe": "nifty500",
        "costs": ["--cost-model", "india", "--slippage-bps", "10"],
    },
    "us": {
        "universe": "sp500",
        "costs": [
            "--cost-model",
            "flat",
            "--commission-bps",
            "1",
            "--slippage-bps",
            "5",
        ],
    },
}

# strategy -> (top slots, hold days). Portfolio size is strategy-dependent:
# trend followers get few concentrated slots + long holds; mean reversion gets
# many diversified slots + short holds; MACD sits in between.
STRATEGIES = {
    # trend following: few concentrated slots + long holds
    "golden_cross_50_200": {"top": 10, "hold": 500},
    "fifty_two_week_high": {"top": 10, "hold": 500},
    "bll_trading_range_break": {"top": 10, "hold": 500},
    "keltner_breakout": {"top": 10, "hold": 100},
    "adx_trend": {"top": 10, "hold": 250},
    "long_term_reversal": {"top": 10, "hold": 250},
    # momentum oscillator: middle
    "macd_signal_cross": {"top": 15, "hold": 60},
    "stochastic_cross": {"top": 15, "hold": 20},
    # mean reversion: many diversified slots + short holds
    "connors_rsi2": {"top": 20, "hold": 20},
    "connors_rsi2_bull": {"top": 20, "hold": 20},
    "bollinger_mean_reversion": {"top": 20, "hold": 20},
    "williams_percent_r": {"top": 20, "hold": 20},
    "cci_reversion": {"top": 20, "hold": 20},
    "short_term_reversal": {"top": 20, "hold": 20},
    "turn_of_month": {"top": 20, "hold": 20},
}

HORIZONS = [5, 3, 2, 1]

METRICS = [
    "Total Return",
    "CAGR",
    "Volatility (ann.)",
    "Sharpe",
    "Sortino",
    "Calmar",
    "Probabilistic Sharpe",
    "Max Drawdown",
    "Hit Rate",
    "Alpha (ann.)",
    "Beta",
    "Benchmark Return",
    "Trades",
    "Unique Tickers",
    "Avg Trade Return",
    "Median Trade Return",
    "Best Trade",
    "Worst Trade",
    "Profit Factor",
    "Winning Trades",
    "Losing Trades",
]


def parse_metrics(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^(\w[\w ()\.,/'-]*?)\s+([+-]?[\d.,]+%?)\s*$", line.strip())
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key in METRICS:
            out[key] = val
    return out


def run_one(market: str, strategy: str, years: int) -> dict[str, str]:
    cfg = MARKETS[market]
    sizing = STRATEGIES[strategy]
    cmd = [
        str(ROOT / ".venv" / "bin" / "screener"),
        "backtest-rolling",
        "-m",
        market,
        "--years",
        str(years),
        "--strategy",
        strategy,
        "--top",
        str(sizing["top"]),
        "--hold",
        str(sizing["hold"]),
        "--universe",
        cfg["universe"],
        *cfg["costs"],
    ]
    log_path = OUT / f"{market}__{strategy}__{years}y.log"
    print(f"[{datetime.now():%H:%M:%S}] running {' '.join(cmd[2:])}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=2400)
    text = proc.stdout + "\n" + proc.stderr
    log_path.write_text(text)
    if proc.returncode != 0:
        print(f"  !! exit {proc.returncode}: {text[-600:]}", flush=True)
    metrics = parse_metrics(text)
    metrics["market"] = market
    metrics["strategy"] = strategy
    metrics["years"] = str(years)
    metrics["top"] = str(sizing["top"])
    metrics["hold"] = str(sizing["hold"])
    metrics["status"] = "ok" if proc.returncode == 0 else "error"
    return metrics


def main() -> None:
    results: list[dict[str, str]] = []
    csv_path = OUT / "results.csv"
    if csv_path.exists():
        csv_path.unlink()
    fieldnames = None
    for market in MARKETS:
        for strategy in STRATEGIES:
            for years in HORIZONS:
                row = run_one(market, strategy, years)
                results.append(row)
                if fieldnames is None:
                    fieldnames = list(row.keys())
                    with csv_path.open("w", newline="") as fh:
                        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
                with csv_path.open("a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
