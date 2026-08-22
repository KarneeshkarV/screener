#!/usr/bin/env python
"""Reconstruct dated Nifty 50 membership from archived NSE constituent lists."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WAYBACK = "https://web.archive.org/web/{timestamp}id_/{url}"
NSE_ARCHIVES = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NIFTYINDICES = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

# Spaced captures that cover the 2021-2026 window used by the 1/2/3/5y study.
SNAPSHOTS: tuple[tuple[str, str, str], ...] = (
    ("20210615183523", NIFTYINDICES, "2021-06-15"),
    ("20210903113122", NSE_ARCHIVES, "2021-09-03"),
    ("20220728034149", NIFTYINDICES, "2022-07-28"),
    ("20230126060250", NSE_ARCHIVES, "2023-01-26"),
    ("20230522051228", NSE_ARCHIVES, "2023-05-22"),
    ("20240205222833", NSE_ARCHIVES, "2024-02-05"),
    ("20240524073805", NIFTYINDICES, "2024-05-24"),
    ("20241110013717", NIFTYINDICES, "2024-11-10"),
    ("20250807042704", NSE_ARCHIVES, "2025-08-07"),
)

DEFAULT_OUT = Path("data/universes/nifty50_history.csv")


def fetch_snapshot(timestamp: str, url: str) -> tuple[str, ...]:
    resp = requests.get(WAYBACK.format(timestamp=timestamp, url=url), timeout=90)
    resp.raise_for_status()
    frame = pd.read_csv(StringIO(resp.text))
    column = "Symbol" if "Symbol" in frame.columns else "SYMBOL"
    if column not in frame.columns:
        raise RuntimeError(f"archived CSV {timestamp} has no Symbol column")
    symbols = frame[column].dropna().astype(str).str.strip().str.upper()
    return tuple(dict.fromkeys(f"{symbol}.NS" for symbol in symbols))


def build(backfill_from: str, include_current: bool) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    previous: set[str] | None = None
    for index, (timestamp, url, effective) in enumerate(SNAPSHOTS):
        symbols = fetch_snapshot(timestamp, url)
        date_used = backfill_from if index == 0 else effective
        rows.extend((date_used, symbol) for symbol in symbols)
        current = set(symbols)
        churn = (
            f"+{len(current - previous)}/-{len(previous - current)}"
            if previous is not None
            else "baseline"
        )
        print(f"{date_used}  n={len(symbols):3d}  {churn}", flush=True)
        previous = current

    if include_current:
        from screener.universes import load_current_universe

        loaded = load_current_universe("nifty50", use_cache=True)
        today = pd.Timestamp.today().date().isoformat()
        rows.extend((today, symbol) for symbol in loaded.symbols)
        print(f"{today}  n={len(loaded.symbols):3d}  (live)", flush=True)

    frame = pd.DataFrame(rows, columns=["effective_date", "symbol"])
    frame = frame.drop_duplicates()
    return frame.sort_values(["effective_date", "symbol"], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backfill-from", default="2021-01-01")
    parser.add_argument("--no-current", action="store_true")
    args = parser.parse_args()

    frame = build(args.backfill_from, not args.no_current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        f"wrote {args.out}: {frame['effective_date'].nunique()} snapshots, "
        f"{len(frame)} rows, {frame['symbol'].nunique()} unique symbols"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
