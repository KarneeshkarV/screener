#!/usr/bin/env python
"""Print a compact leaderboard from the PIT mid/small matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path("findings/pit_midsmall/results.csv")


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}")
        return 1
    df = pd.read_csv(CSV)
    print(f"rows={len(df)} errors={(df.get('error', '') != '').sum() if 'error' in df else 0}")
    if df.empty:
        return 0
    work = df.copy()
    if "error" in work:
        work = work[work["error"].fillna("") == ""]
    for col in ("sharpe", "cagr", "max_drawdown", "hit_rate", "n_trades"):
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    print("\nby universe x years (count)")
    print(work.groupby(["universe", "years"]).size().unstack(fill_value=0))
    print("\ntop Sharpe by universe (mean across windows, min 2 windows)")
    g = (
        work.dropna(subset=["sharpe"])
        .groupby(["universe", "strategy"])
        .agg(mean_sharpe=("sharpe", "mean"), min_sharpe=("sharpe", "min"), n=("sharpe", "size"), trades=("n_trades", "sum"))
        .reset_index()
    )
    g = g[g["n"] >= 2]
    for univ in ("mid", "small", "midsmall"):
        block = g[g["universe"] == univ].sort_values("mean_sharpe", ascending=False).head(10)
        print(f"\n=== {univ} ===")
        print(block.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
