#!/usr/bin/env python
"""Reconstruct dated Nifty 500 membership from archived NSE constituent lists.

``--point-in-time`` needs dated membership. The S&P 500 gets it from the
Wikipedia change log (``screener.universes._fetch_sp500_changes``); NSE
publishes only a *current* constituent CSV, so India runs silently apply
today's 500 names to all history. That bias is large: 164 of today's members
were not in the index in May 2022, and they are exactly the names that rallied
hard enough to be promoted.

This script rebuilds what it can from the Internet Archive's copies of the two
URLs NSE has served that CSV from, and writes a snapshot CSV that
``type = "snapshots"`` universes consume directly.

    uv run python scripts/build_nifty500_history.py
    uv run python scripts/build_nifty500_history.py --out data/universes/nifty500_history.csv

Re-running is safe: output is fully derived from the fetched snapshots and is
sorted, so an unchanged archive produces a byte-identical file.

Known limits, all of which leave residual upward bias:

* Snapshot dates are whatever the Archive happens to hold, not NSE's actual
  semi-annual rebalance dates, so changes inside a gap collapse onto the next
  observed date.
* The earliest snapshot is backdated to ``--backfill-from`` so runs can start
  before the Archive's first copy. Membership is assumed constant before then.
* Names that merged, were renamed, or delisted often have no price history
  left upstream, so they still cannot be traded even when membership is right.

For history from here on, prefer ``screener universes sync`` on a cron: it
records real membership changes as they happen and needs no archaeology.
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Archived copies of NSE's constituent CSV. NSE moved the file between hosts,
# so neither URL alone covers the full period: archives.nseindia.com holds
# 2022-2024 and niftyindices.com holds 2024-2026.
WAYBACK = "https://web.archive.org/web/{timestamp}id_/{url}"
NSE_ARCHIVES = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTYINDICES = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

# (wayback timestamp, source url, effective date). The effective date is the
# archive capture date, except for the first entry -- see --backfill-from.
SNAPSHOTS: tuple[tuple[str, str, str], ...] = (
    ("20220504103923", NSE_ARCHIVES, "2022-05-04"),
    ("20221009160959", NSE_ARCHIVES, "2022-10-09"),
    ("20230404164710", NSE_ARCHIVES, "2023-04-04"),
    ("20240226224931", NSE_ARCHIVES, "2024-02-26"),
    ("20250616113621", NIFTYINDICES, "2025-06-16"),
    ("20250815235304", NIFTYINDICES, "2025-08-15"),
    ("20260107050129", NIFTYINDICES, "2026-01-07"),
    ("20260502114023", NIFTYINDICES, "2026-05-02"),
)

DEFAULT_OUT = Path("data/universes/nifty500_history.csv")


def fetch_snapshot(timestamp: str, url: str) -> tuple[str, ...]:
    """Return the constituent symbols of one archived capture, in NSE form."""
    resp = requests.get(WAYBACK.format(timestamp=timestamp, url=url), timeout=60)
    resp.raise_for_status()
    frame = pd.read_csv(StringIO(resp.text))
    column = "Symbol" if "Symbol" in frame.columns else "SYMBOL"
    if column not in frame.columns:
        raise RuntimeError(f"archived CSV {timestamp} has no Symbol column")
    symbols = frame[column].dropna().astype(str).str.strip().str.upper()
    # yfinance form, matching the built-in nifty500 loader.
    return tuple(dict.fromkeys(f"{symbol}.NS" for symbol in symbols))


def build(backfill_from: str, include_current: bool) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    previous: set[str] | None = None
    for index, (timestamp, url, effective) in enumerate(SNAPSHOTS):
        symbols = fetch_snapshot(timestamp, url)
        # The first capture stands in for everything before it, so a 5-year
        # backtest is not left with an empty universe for its opening months.
        date_used = backfill_from if index == 0 else effective
        rows.extend((date_used, symbol) for symbol in symbols)
        current = set(symbols)
        churn = (
            f"+{len(current - previous)}/-{len(previous - current)}"
            if previous is not None
            else "baseline"
        )
        print(f"{date_used}  n={len(symbols):3d}  {churn}", file=sys.stderr)
        previous = current

    if include_current:
        from screener.universes import load_current_universe

        loaded = load_current_universe("nifty500", use_cache=True)
        today = pd.Timestamp.today().date().isoformat()
        rows.extend((today, symbol) for symbol in loaded.symbols)
        print(f"{today}  n={len(loaded.symbols):3d}  (live)", file=sys.stderr)

    frame = pd.DataFrame(rows, columns=["effective_date", "symbol"])
    frame = frame.drop_duplicates()
    return frame.sort_values(["effective_date", "symbol"], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--backfill-from",
        default="2021-01-01",
        help="effective date for the earliest archived snapshot",
    )
    parser.add_argument(
        "--no-current",
        action="store_true",
        help="skip appending today's live nifty500 membership",
    )
    args = parser.parse_args()

    frame = build(args.backfill_from, not args.no_current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        f"\nwrote {args.out}: {frame['effective_date'].nunique()} snapshots, "
        f"{len(frame)} rows, {frame['symbol'].nunique()} unique symbols"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
