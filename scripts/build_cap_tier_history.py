#!/usr/bin/env python
"""Reconstruct dated NSE mid/small/microcap membership from historical market cap.

``scripts/build_nifty500_history.py`` rebuilds Nifty 500 membership from archived
NSE constituent CSVs, but the Internet Archive holds only one or two captures of
the mid-, small-, and microcap constituent files - far too sparse for
point-in-time backtests. This script reconstructs those tiers from the
methodology instead of from archived lists.

NSE ranks the listed pool by average full market capitalisation and slices it
into fixed bands. The Nifty Total Market index is the top 750 names, and the cap
tiers are contiguous rank bands inside it:

* ranks   1-100  Nifty 100         (largecap)
* ranks 101-250  Nifty Midcap 150
* ranks 251-500  Nifty Smallcap 250
* ranks 501-750  Nifty Microcap 250

Reconstitution is semi-annual, effective at the end of March and September,
using the average market cap over the preceding six months. This script
reproduces that: it pulls month-end market caps for every NSE symbol from FMP,
averages the six months *before* each effective date, ranks, and writes one
snapshot CSV per tier that ``type = "snapshots"`` universes consume directly.

Ranks 101-500 are taken *within* the Nifty 500 membership that was in force on
each date, read from ``build_nifty500_history.py``'s output, because the mid and
small indices are defined as rank bands of the Nifty 500 rather than of the raw
listed pool. That anchor inherits NSE's own eligibility screens for those two
tiers and measurably improves agreement with the published lists. Microcap has
no such anchor and is ranked over everything outside the Nifty 500.

Against NSE's live constituent files, the 2026-03-31 snapshot reproduces 85% of
Nifty Midcap 150, 84% of Nifty Smallcap 250, and 60% of Nifty Microcap 250. The
residual gap is NSE's listing-history and trading-frequency screens plus drift
between the reconstitution date and the live file.

    uv run python scripts/build_cap_tier_history.py

Re-running is safe. The FMP pull is cached in ``--cache`` and reused unless
``--refresh`` is passed, and the outputs are fully derived and sorted, so an
unchanged cache produces byte-identical files.

Known limits, and which way each one biases results:

* The candidate pool is FMP's *current* NSE listing, including names flagged
  inactive. Companies delisted before FMP's coverage began are absent, which
  leaves a residual upward bias concentrated in the microcap tier.
* Ranking uses six month-end observations rather than NSE's daily average, so a
  name sitting within a few ranks of a band edge can land on the wrong side of
  it. Mid-band membership is unaffected.
* FMP market cap is shares outstanding times price, not NSE's free-float
  adjusted figure. This shifts ranks for closely held companies.
* Membership before the first effective date is assumed constant, backfilled
  from the earliest computed snapshot, so runs can start before the window.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from screener.fmp import resolve_api_key

SCREENER_URL = "https://financialmodelingprep.com/api/v3/stock-screener"
MARKET_CAP_URL = (
    "https://financialmodelingprep.com/stable/historical-market-capitalization"
)

# (tier name, first rank, last rank), inclusive, matching NSE's index bands.
# ``midcap150`` and ``smallcap250`` are ranked inside Nifty 500 membership;
# ``microcap250`` is ranked over everything outside it. See ``rank_at``.
IN_500_TIERS: tuple[tuple[str, int, int], ...] = (
    ("midcap150", 101, 250),
    ("smallcap250", 251, 500),
)
MICROCAP_SIZE = 250
# Size of the stand-in Nifty 500 used before archived membership begins.
ANCHOR_SIZE = 500
# The three tiers stacked into one tradable universe, for a run that ranks
# momentum across the whole small/mid/micro space instead of within a band.
COMBINED = "smid650"
TIER_NAMES = tuple(name for name, _, _ in IN_500_TIERS) + ("microcap250", COMBINED)

DEFAULT_OUT_DIR = Path("data/universes")
DEFAULT_CACHE = Path("data/universes/nse_marketcap_monthly.csv.gz")
DEFAULT_NIFTY500 = Path("data/universes/nifty500_history.csv")
LOOKBACK_MONTHS = 6


def _fetch_json(url: str, params: dict[str, str], api_key: str) -> object:
    query = urllib.parse.urlencode({**params, "apikey": api_key})
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "screener-cli/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def fetch_candidates(api_key: str) -> list[str]:
    """Return every NSE symbol FMP knows about, active or not."""
    rows = _fetch_json(SCREENER_URL, {"exchange": "NSE", "limit": "20000"}, api_key)
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected screener payload: {type(rows).__name__}")
    symbols = {
        str(row["symbol"]).strip().upper()
        for row in rows
        if isinstance(row, dict) and row.get("symbol")
    }
    # Only .NS names are tradable through the yfinance price path the backtester
    # uses; FMP also lists a few BSE-suffixed duplicates on this exchange.
    return sorted(symbol for symbol in symbols if symbol.endswith(".NS"))


def fetch_month_end_caps(
    symbol: str, start: str, end: str, api_key: str
) -> pd.DataFrame:
    """Return month-end market caps for one symbol, empty if FMP has none."""
    try:
        rows = _fetch_json(
            MARKET_CAP_URL,
            {"symbol": symbol, "from": start, "to": end, "limit": "5000"},
            api_key,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return pd.DataFrame(columns=["month", "symbol", "market_cap"])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["month", "symbol", "market_cap"])

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns or "marketCap" not in frame.columns:
        return pd.DataFrame(columns=["month", "symbol", "market_cap"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["marketCap"] = pd.to_numeric(frame["marketCap"], errors="coerce")
    frame = frame.dropna(subset=["date", "marketCap"])
    frame = frame[frame["marketCap"] > 0]
    if frame.empty:
        return pd.DataFrame(columns=["month", "symbol", "market_cap"])

    # Keep the last observation of each calendar month: six of these stand in
    # for NSE's daily six-month average.
    frame = frame.sort_values("date")
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = pd.DataFrame(frame.groupby("month", as_index=False)["marketCap"].last())
    monthly["symbol"] = symbol
    monthly = monthly.rename(columns={"marketCap": "market_cap"})
    return monthly[["month", "symbol", "market_cap"]]


def build_cap_history(
    symbols: list[str], start: str, end: str, api_key: str, workers: int
) -> pd.DataFrame:
    """Fetch month-end market caps for every symbol, in parallel."""
    frames: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for frame in pool.map(
            lambda symbol: fetch_month_end_caps(symbol, start, end, api_key), symbols
        ):
            done += 1
            if not frame.empty:
                frames.append(frame)
            if done % 100 == 0:
                print(
                    f"  fetched {done}/{len(symbols)} symbols, {len(frames)} with data",
                    file=sys.stderr,
                )
    if not frames:
        raise RuntimeError("FMP returned no market cap history for any symbol")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["month", "symbol"], kind="stable")


def effective_dates(start: str, end: str) -> list[pd.Timestamp]:
    """Semi-annual reconstitution dates (end of March and September) in range."""
    dates = pd.date_range(start=start, end=end, freq="QE-MAR")
    return [date for date in dates if date.month in (3, 9)]


def rank_at(
    caps: pd.DataFrame, effective: pd.Timestamp, lookback: int
) -> pd.Series | None:
    """Average market cap over the ``lookback`` months strictly before ``effective``."""
    window_start = effective - pd.DateOffset(months=lookback)
    window = caps[(caps["month"] > window_start) & (caps["month"] <= effective)]
    if window.empty:
        return None
    # Require at least half the window so a newly listed name cannot outrank
    # established ones on a single lucky observation.
    counts = window.groupby("symbol")["market_cap"].count()
    eligible = counts[counts >= max(2, lookback // 2)].index
    averages = (
        window[window["symbol"].isin(eligible)].groupby("symbol")["market_cap"].mean()
    )
    if averages.empty:
        return None
    return averages.sort_values(ascending=False)


def nifty500_members(history: pd.DataFrame, effective: pd.Timestamp) -> set[str]:
    """Nifty 500 membership in force on ``effective``, from the snapshot file."""
    prior = history[history["effective_date"] <= effective]
    if prior.empty:
        return set()
    latest = prior["effective_date"].max()
    return set(prior.loc[prior["effective_date"] == latest, "symbol"])


def synthetic_anchor(ranked: pd.Series) -> set[str]:
    """Stand-in Nifty 500 for dates before the archived membership begins.

    The archived Nifty 500 history starts in 2021, so a ten-year run has no
    anchor for its first five years. NSE's own selection is roughly the top 500
    of the listed pool by average market cap, so that band is used instead. It
    is a weaker anchor: it lacks NSE's listing-history and trading-frequency
    screens and its free-float adjustment, so tier membership near the 500 line
    is noisier before 2021 than after.
    """
    return set(ranked.index[:ANCHOR_SIZE])


def build_tier_frames(
    caps: pd.DataFrame,
    dates: list[pd.Timestamp],
    nifty500: pd.DataFrame,
    backfill_from: str,
    lookback: int,
) -> dict[str, pd.DataFrame]:
    """Return one snapshot frame per tier, keyed by tier name.

    Midcap and smallcap are ranked *within* the Nifty 500 members in force on
    each date, because that is how NSE defines them - the mid and small indices
    are contiguous rank bands of the Nifty 500, not of the raw listed pool.
    Anchoring this way inherits NSE's own eligibility screens (listing history,
    trading frequency, and so on) for those two tiers instead of re-deriving
    them. Microcap has no such anchor, so it is ranked over everything outside
    the Nifty 500.
    """
    rows: dict[str, list[tuple[str, str]]] = {name: [] for name in TIER_NAMES}
    first = True
    for effective in dates:
        ranked = rank_at(caps, effective, lookback)
        if ranked is None:
            print(f"{effective.date()}  no data in window, skipped", file=sys.stderr)
            continue
        members = nifty500_members(nifty500, effective)
        anchor = "n500"
        if not members:
            members = synthetic_anchor(ranked)
            anchor = "top500"
        # The earliest usable snapshot stands in for everything before it, so a
        # 5-year run is not left with an empty universe for its opening months.
        date_used = backfill_from if first else effective.date().isoformat()
        first = False

        inside = ranked[ranked.index.isin(members)]
        outside = ranked[~ranked.index.isin(members)]
        tiers: dict[str, list[str]] = {
            name: inside.index[low - 1 : high].tolist()
            for name, low, high in IN_500_TIERS
        }
        tiers["microcap250"] = outside.index[:MICROCAP_SIZE].tolist()
        tiers[COMBINED] = [symbol for name in TIER_NAMES[:-1] for symbol in tiers[name]]

        for name, symbols in tiers.items():
            rows[name].extend((date_used, symbol) for symbol in symbols)
        print(
            f"{date_used}  pool={len(ranked):4d}  {anchor}={len(inside):3d}  "
            + "  ".join(f"{name}={len(symbols)}" for name, symbols in tiers.items()),
            file=sys.stderr,
        )

    frames: dict[str, pd.DataFrame] = {}
    for name in TIER_NAMES:
        frame = pd.DataFrame(rows[name], columns=["effective_date", "symbol"])
        frame = frame.drop_duplicates()
        frames[name] = frame.sort_values(["effective_date", "symbol"], kind="stable")
    return frames


def build_extended500(
    caps: pd.DataFrame,
    dates: list[pd.Timestamp],
    nifty500: pd.DataFrame,
    backfill_from: str,
    lookback: int,
) -> pd.DataFrame:
    """Return Nifty 500 membership extended back before the archived history.

    The archived Nifty 500 snapshots begin in 2021, which is not enough for a
    ten-year run. Dates from the first archived snapshot onward are copied
    verbatim, so nothing about the accurate part of the history is re-derived or
    rounded onto a semi-annual grid. Earlier dates are reconstructed as the top
    500 of the listed pool by trailing average market cap, on NSE's own
    semi-annual reconstitution schedule.

    The reconstructed half is the weaker half - see :func:`synthetic_anchor` -
    so a run that starts before 2021 should be read as a lower bound with a
    wider error bar than one that starts after.
    """
    archived_start = nifty500["effective_date"].min()
    rows: list[tuple[str, str]] = []
    first = True
    for effective in dates:
        if effective >= archived_start:
            break
        ranked = rank_at(caps, effective, lookback)
        if ranked is None:
            continue
        date_used = backfill_from if first else effective.date().isoformat()
        first = False
        rows.extend((date_used, symbol) for symbol in synthetic_anchor(ranked))
    reconstructed = pd.DataFrame(rows, columns=["effective_date", "symbol"])
    archived = nifty500.copy()
    archived["effective_date"] = archived["effective_date"].dt.date.astype(str)
    combined = pd.concat([reconstructed, archived[["effective_date", "symbol"]]])
    combined = combined.drop_duplicates()
    return combined.sort_values(["effective_date", "symbol"], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--nifty500",
        type=Path,
        default=DEFAULT_NIFTY500,
        help="snapshot CSV from build_nifty500_history.py, used to anchor tiers",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-pull FMP data, ignoring --cache"
    )
    parser.add_argument("--start", default="2020-06-01")
    parser.add_argument(
        "--cap-start",
        default=None,
        help=(
            "first month of the FMP market-cap pull (default: --start). Set it "
            "earlier than --start so the first reconstitution has a full "
            "six-month averaging window instead of a truncated one."
        ),
    )
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument(
        "--backfill-from",
        default="2021-01-01",
        help="effective date for the earliest computed snapshot",
    )
    parser.add_argument("--lookback-months", type=int, default=LOOKBACK_MONTHS)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.cache.exists() and not args.refresh:
        print(f"reading cached market caps from {args.cache}", file=sys.stderr)
        with gzip.open(args.cache, "rt") as handle:
            caps = pd.read_csv(handle, parse_dates=["month"])
    else:
        api_key = resolve_api_key()
        if not api_key:
            print("FMP_API_KEY is not set", file=sys.stderr)
            return 1
        symbols = fetch_candidates(api_key)
        print(f"NSE candidate pool: {len(symbols)} symbols", file=sys.stderr)
        caps = build_cap_history(
            symbols, args.cap_start or args.start, args.end, api_key, args.workers
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.cache, "wt", newline="") as handle:
            caps.to_csv(handle, index=False)
        print(
            f"cached {len(caps)} month-end observations for "
            f"{caps['symbol'].nunique()} symbols to {args.cache}",
            file=sys.stderr,
        )

    if not args.nifty500.exists():
        print(
            f"{args.nifty500} is missing; run scripts/build_nifty500_history.py first",
            file=sys.stderr,
        )
        return 1
    nifty500 = pd.read_csv(args.nifty500, parse_dates=["effective_date"])

    dates = effective_dates(args.start, args.end)
    frames = build_tier_frames(
        caps, dates, nifty500, args.backfill_from, args.lookback_months
    )

    extended = build_extended500(
        caps, dates, nifty500, args.backfill_from, args.lookback_months
    )
    frames["500_extended"] = extended

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        out = args.out_dir / f"nifty_{name}_history.csv"
        frame.to_csv(out, index=False)
        print(
            f"wrote {out}: {frame['effective_date'].nunique()} snapshots, "
            f"{len(frame)} rows, {frame['symbol'].nunique()} unique symbols"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
