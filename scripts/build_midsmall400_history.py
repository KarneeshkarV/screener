#!/usr/bin/env python
"""Construct the mid+small (ranks 101-500) point-in-time snapshot CSV.

Midcap 150 and Smallcap 250 are contiguous rank bands of the Nifty 500.
This writes their union per effective date. It does not include microcap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_MID = Path("data/universes/nifty_midcap150_history.csv")
DEFAULT_SMALL = Path("data/universes/nifty_smallcap250_history.csv")
DEFAULT_OUT = Path("data/universes/nifty_midsmall400_history.csv")


def build_combined(mid_path: Path, small_path: Path) -> pd.DataFrame:
    mid = pd.read_csv(mid_path, parse_dates=["effective_date"])
    small = pd.read_csv(small_path, parse_dates=["effective_date"])
    for frame, label in ((mid, "mid"), (small, "small")):
        missing = {"effective_date", "symbol"} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} CSV missing columns: {sorted(missing)}")
    combined = pd.concat([mid, small], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.strip().str.upper()
    combined = combined.dropna(subset=["effective_date", "symbol"])
    combined = combined.drop_duplicates(["effective_date", "symbol"])
    combined = combined.sort_values(["effective_date", "symbol"], kind="stable")
    return combined[["effective_date", "symbol"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid", type=Path, default=DEFAULT_MID)
    parser.add_argument("--small", type=Path, default=DEFAULT_SMALL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    combined = build_combined(args.mid, args.small)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    counts = combined.groupby("effective_date").size()
    print(
        f"wrote {args.out}: {combined['effective_date'].nunique()} snapshots, "
        f"{len(combined)} rows, {combined['symbol'].nunique()} unique symbols, "
        f"per-date {int(counts.min())}-{int(counts.max())} "
        f"(median {int(counts.median())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
