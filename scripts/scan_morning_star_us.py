"""Run a single named strategy on the US universe and report fresh entries
(within the last LOOKBACK_DAYS sessions).

Usage: uv run python scripts/scan_morning_star_us.py <strategy_name>
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch import _cached_fetch
from autoresearch_strategies import NEW_STRATEGIES
from run_pinescript_strategies import load_universe
from screener.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("scan_morning_star_us")

LOOKBACK_DAYS = 7
WARMUP_YEARS = 4
LIMIT = 500
MARKET = "us"

STRAT_NAME = sys.argv[1] if len(sys.argv) > 1 else "morning_star_pullback"
if STRAT_NAME not in NEW_STRATEGIES:
    print(
        f"Unknown strategy '{STRAT_NAME}'. Available example: morning_star_pullback",
        file=sys.stderr,
    )
    sys.exit(2)
STRAT_FN = NEW_STRATEGIES[STRAT_NAME]


def fetch_one(ticker, fetch_start, fetch_end):
    try:
        df = _cached_fetch(ticker, fetch_start, fetch_end, MARKET, refresh=False)
    except (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        KeyError,
        ValueError,
        OSError,
    ):
        df = None
    return ticker, df


def main():
    today = pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=LOOKBACK_DAYS)
    fetch_start = (today - pd.DateOffset(years=WARMUP_YEARS)).date()
    fetch_end = today.date()

    tickers = load_universe(MARKET, None)[:LIMIT]
    log.info("scan.universe_loaded", market="us", size=len(tickers))

    ohlcv = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_one, t, fetch_start, fetch_end): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t, df = fut.result()
            if df is not None and not df.empty and len(df) > 250:
                ohlcv[t] = df.sort_values("date").reset_index(drop=True)
            if i % 100 == 0:
                log.info(
                    "scan.fetch_progress",
                    market="us",
                    fetched=i,
                    total=len(tickers),
                    usable=len(ohlcv),
                )
    log.info("scan.usable_tickers", market="us", usable=len(ohlcv))

    hits = []
    for ticker, df in ohlcv.items():
        try:
            trades = STRAT_FN(df)
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError):
            continue
        if not trades:
            continue
        for tr in trades:
            ed = pd.Timestamp(tr.entry_date)
            if ed >= cutoff:
                last = df.iloc[-1]
                last_close = float(last["close"])
                last_date = pd.Timestamp(last["date"]).date()
                gain_since = (last_close / float(tr.entry_px) - 1.0) * 100.0
                hits.append(
                    {
                        "ticker": ticker,
                        "entry_date": str(ed.date()),
                        "entry_px": float(tr.entry_px),
                        "last_close": last_close,
                        "last_date": str(last_date),
                        "days_since": (today - ed).days,
                        "gain_since_entry_pct": gain_since,
                    }
                )

    hits.sort(key=lambda r: (r["days_since"], -r["gain_since_entry_pct"]))

    print(
        f"\n=== US {STRAT_NAME} fresh entries "
        f"(last {LOOKBACK_DAYS}d, as of {today.date()}) ==="
    )
    if not hits:
        print("  (no fresh signals)")
    else:
        print(
            f"{'#':>3}  {'TICKER':<8}  {'ENTRY':<11}  "
            f"{'ENTRY_PX':>9}  {'LAST':>9}  {'GAIN%':>7}  {'DAYS':>4}"
        )
        for i, h in enumerate(hits, 1):
            print(
                f"{i:>3}  {h['ticker']:<8}  {h['entry_date']:<11}  "
                f"{h['entry_px']:>9.2f}  {h['last_close']:>9.2f}  "
                f"{h['gain_since_entry_pct']:>+6.2f}%  {h['days_since']:>4}"
            )
    print(
        f"\n[done] {len(hits)} fresh {STRAT_NAME} signals across "
        f"{len(ohlcv)} US tickers"
    )


if __name__ == "__main__":
    main()
