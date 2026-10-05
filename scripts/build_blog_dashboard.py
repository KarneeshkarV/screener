"""Build a standalone offline HTML dashboard from saved blog backtests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_blog_dashboard() -> Path:
    """Embed results and observed monthly equity without external dependencies."""
    rows = []
    for summary in sorted((ROOT / "reports/blog_momentum").glob("*/summary.json")):
        if summary.parent.name.endswith("_repeat"):
            continue
        for row in json.loads(summary.read_text()):
            if "strategy" not in row:
                continue
            folder = summary.parent / row["cell"]
            curve = pd.read_csv(folder / "equity.csv", index_col=0).iloc[:, 0]
            curve.index = pd.to_datetime(curve.index)
            monthly = curve.resample("ME").last().dropna()
            row["curve"] = [
                [
                    str(pd.Timestamp(str(d)).date()),
                    round(float(v / row["starting_equity"] * 100), 3),
                ]
                for d, v in monthly.items()
            ]
            monthly_returns = monthly.pct_change()
            if not monthly.empty:
                monthly_returns.iloc[0] = monthly.iloc[0] / row["starting_equity"] - 1
            row["returns"] = [
                [str(pd.Timestamp(str(d)).date()), round(float(v * 100), 3)]
                for d, v in monthly_returns.dropna().items()
            ]
            manifest = (
                json.loads((summary.parent / "manifest.json").read_text())
                if (summary.parent / "manifest.json").exists()
                else {"start": "2020-01-01", "end": "2025-12-31"}
            )
            row["start"] = manifest["start"]
            row["end"] = manifest["end"]
            row["window"] = (
                summary.parent.name.removeprefix("window_")
                if summary.parent.name.startswith("window_")
                else "6y"
            )
            row["cell"] = f"{row['window']}:{row['cell']}"
            row["source"] = {
                "chan": "Ernie Chan",
                "alvarez": "Alvarez",
                "unger": "Unger",
                "robotwealth": "Robot Wealth",
                "momentum": "Reference",
                "mark": "Reference",
            }[row["strategy"].split("_")[0]]
            row["warnings"] = json.loads((folder / "warnings.json").read_text())
            row["source_url"] = {
                "Ernie Chan": "https://epchan.blogspot.com/",
                "Alvarez": "https://alvarezquanttrading.com/blog/",
                "Unger": "https://ungeracademy.com/blog",
                "Robot Wealth": "https://robotwealth.com/blog/",
                "Reference": "https://alvarezquanttrading.com/blog/different-ranking-methods-for-a-monthly-sp500-stock-rotation-strategy/",
            }[row["source"]]
            if (
                row["strategy"].startswith("alvarez_monthly")
                or "rotation_roc" in row["strategy"]
                or row["strategy"] == "alvarez_three_factor"
            ):
                row["source_url"] = (
                    "https://alvarezquanttrading.com/blog/three-factor-etf-rotation-strategy/"
                    if "factor" in row["strategy"]
                    else "https://alvarezquanttrading.com/blog/different-ranking-methods-for-a-monthly-sp500-stock-rotation-strategy/"
                    if "rotation" in row["strategy"]
                    else "https://alvarezquanttrading.com/blog/trend-following-plus-momentum-in-etfs/"
                    if "dual" in row["strategy"] or "either" in row["strategy"]
                    else "https://alvarezquanttrading.com/blog/trend-following-vs-momentum-in-etfs/"
                )
                row["warnings"].insert(
                    0,
                    "Monthly adaptation: rotation liquidates one session early, not synchronized next-open rotation. Stock ROC-HV omits the source market MA200 gate. See findings/blog_monthly_extension.md.",
                )
            row["artifact"] = str(folder.relative_to(ROOT))
            row = {
                key: None
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for key, value in row.items()
            }
            rows.append(row)
    template = (ROOT / "scripts/blog_dashboard.html").read_text()
    output = ROOT / "reports/blog_momentum/index.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(
        template.replace(
            "__RESULTS__", json.dumps(rows, allow_nan=False).replace("</", "<\\/")
        )
    )
    temporary.replace(output)
    print(f"{len(rows)} cells: {output}")
    return output


if __name__ == "__main__":
    build_blog_dashboard()
