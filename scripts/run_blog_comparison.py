"""Run matched 5/3/2/1-year windows using the original frozen study data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = (
    "chan_return_12m",
    "chan_weighted_13612",
    "chan_log_ma_7_10",
    "chan_log_ma_50_200",
    "alvarez_ma200_confirm3",
    "alvarez_ma200_confirm5",
    "unger_close_sma200",
    "unger_sma50_200",
    "unger_sma200_slope50",
    "robotwealth_close_sma100",
    "unger_donchian5",
    "alvarez_monthly_momentum10",
    "alvarez_monthly_sma10",
    "alvarez_monthly_dual10",
    "alvarez_monthly_either10",
    "alvarez_rotation_roc_hv63",
    "alvarez_rotation_roc_hv126",
    "alvarez_rotation_roc_hv252",
    "alvarez_three_factor",
    "momentum_12_1",
    "mark_minervini",
)


def run_blog_comparison(years: list[int]) -> None:
    """Preserve histories and membership, and compare fixed rule parameters."""
    base = ROOT / "reports/blog_momentum"
    for year in years:
        out = base / f"window_{year}y"
        out.mkdir(parents=True, exist_ok=True)
        for market in ("us", "india"):
            for asset in ("stocks", "etfs"):
                link = out / f"{market}_{asset}_bars"
                if not link.exists():
                    link.symlink_to(Path("../chan") / link.name)
            universe = out / f"{market}_stocks_universe.json"
            if not universe.exists():
                universe.write_bytes((base / "chan" / universe.name).read_bytes())
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_blog_momentum",
                "--start",
                f"{2026 - year}-01-01",
                "--end",
                "2025-12-31",
                "--out-dir",
                str(out),
                "--strategies",
                *STRATEGIES,
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps({"completed_windows": years}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", type=int, nargs="+", choices=(5, 3, 2, 1), default=[5, 3, 2, 1]
    )
    run_blog_comparison(parser.parse_args().years)
