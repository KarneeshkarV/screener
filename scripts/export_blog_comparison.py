"""Publish compact backtest tables and a PR-ready result digest, without price data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def export_blog_comparison() -> Path:
    """Save all available study metrics and matched reference differences."""
    base = ROOT / "reports/blog_momentum"
    rows = []
    for path in sorted(base.glob("*/summary.json")):
        if path.parent.name.endswith("_repeat"):
            continue
        window = (
            path.parent.name.removeprefix("window_")
            if path.parent.name.startswith("window_")
            else "6y"
        )
        for row in json.loads(path.read_text()):
            if "strategy" in row:
                rows.append({"window": window, **row})
    table = pd.DataFrame(rows)
    for strategy in ("momentum_12_1", "mark_minervini"):
        refs = table.loc[
            table.strategy.eq(strategy),
            ["window", "market", "asset", "calendar_cagr", "status"],
        ].rename(
            columns={
                "calendar_cagr": f"{strategy}_cagr",
                "status": f"{strategy}_status",
            }
        )
        table = table.merge(
            refs, on=["window", "market", "asset"], how="left", validate="many_to_one"
        )
        table[f"delta_{strategy}_pp"] = (
            table.calendar_cagr - table[f"{strategy}_cagr"]
        ) * 100
    destination = ROOT / "findings/blog_momentum_results"
    destination.mkdir(exist_ok=True)
    table.to_csv(destination / "metrics.csv", index=False)
    shutil.copyfile(base / "index.html", destination / "index.html")
    lines = [
        "# Blog momentum backtest results",
        "",
        f"Saved cells: **{len(table)}**; provisional: **{table.status.eq('provisional').sum()}**.",
        "",
        "Each matched comparison uses the same frozen bars, universe, costs, dates, and portfolio sizing.",
        "The 1/2/3/5-year windows all end on 2025-12-31.",
        "The original six-year window remains available in the HTML.",
        "",
        "## Results",
        "",
        "| Window | Market | Assets | Rule | CAGR | Max drawdown | Δ momentum12_1 | Δ Minervini | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for window in ("6y", "5y", "3y", "2y", "1y"):
        subset = table.loc[table.window.eq(window)]
        for (market, asset), group in subset.groupby(["market", "asset"]):
            best = group.loc[group.status.eq("completed")].nlargest(1, "calendar_cagr")
            references = group.loc[
                group.strategy.isin(["momentum_12_1", "mark_minervini"])
            ]
            for _, row in (
                pd.concat([best, references]).drop_duplicates("strategy").iterrows()
            ):
                def fmt(value: float) -> str:
                    return "n/a" if pd.isna(value) else f"{value:.2f} pp"
                lines.append(
                    f"| {window} | {market} | {asset} | {row.strategy} | {row.calendar_cagr * 100:.2f}% | {row.max_drawdown * 100:.2f}% | {fmt(row.delta_momentum_12_1_pp)} | {fmt(row.delta_mark_minervini_pp)} | {row.status} |"
                )
    lines += [
        "",
        "## Limits",
        "",
        "Rules are disclosed research adaptations, not exact publisher replications.",
        "Monthly rotation liquidates one session before new entries; stock ROC-HV omits the source market MA200 gate.",
        "Provisional means an early history-end sale at the last available quote.",
        "Membership gaps, missing price histories, surviving ETF baskets, and approximate India ETF fees remain limitations.",
        "Reference deltas do not remove provisional reference risk; reference statuses are included in the full CSV.",
        "The strongest in-sample result is not a recommendation or an out-of-sample validation.",
        "",
        "## Files",
        "",
        "- `findings/blog_momentum_results/index.html`: standalone HTML, light/dark themes, window filters, reference comparisons.",
        "- `findings/blog_momentum_results/metrics.csv`: all saved cells and matched-reference differences.",
        "- `findings/blog_monthly_extension.md`: monthly rule and execution assumptions.",
        "- Source inventories: Alvarez, Unger, Robot Wealth, and `findings/blog_momentum.md` for Chan.",
        "",
        "Price data and invalid first-run rotation results are not committed.",
    ]
    output = destination / "README.md"
    output.write_text("\n".join(lines) + "\n")
    print(f"Published {len(table)} cells: {output}")
    return output


if __name__ == "__main__":
    export_blog_comparison()
