"""Scan today's market for fresh buy signals across all profitable strategies.

Loads the list of "good" strategy keys from /tmp/good_strategies.json (built
upstream from journal.jsonl), runs each one against the full US + India
universes ending today, and counts how often each ticker shows up as a fresh
entry within the last LOOKBACK_DAYS sessions.

Output:
  picks_today.json   structured picks by market + strategy
  picks_today.md     human-readable summary
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

from autoresearch import _cached_fetch
from autoresearch_strategies import NEW_STRATEGIES
from run_pinescript_strategies import (
    BENCHMARKS,
    STRATEGIES as STOCK_STRATEGIES,
    load_universe,
)
from screener.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("scan_today")

LOOKBACK_DAYS = 7  # fresh signal = entry within last 7 calendar days
WARMUP_YEARS = 4
LIMIT = 500


def load_good_strategies() -> dict:
    keys = json.load(open("/tmp/good_strategies.json"))
    out: dict = {}
    for key in keys:
        kind, name = key.split(":", 1)
        if kind == "stock":
            fn = STOCK_STRATEGIES.get(name)
        elif kind == "new":
            fn = NEW_STRATEGIES.get(name)
        else:
            fn = None
        if fn is not None:
            out[key] = fn
    return out


def fetch_universe(market: str, limit: int) -> dict[str, pd.DataFrame]:
    today = date.today()
    fetch_start = (pd.Timestamp(today) - pd.DateOffset(years=WARMUP_YEARS)).date()
    fetch_end = today
    tickers = load_universe(market)[:limit]
    log.info("scan.universe_loaded", market=market, size=len(tickers))
    ohlcv: dict[str, pd.DataFrame] = {}

    def _fetch(t):
        try:
            df = _cached_fetch(t, fetch_start, fetch_end, market, refresh=False)
        except (
            requests.RequestException,
            ConnectionError,
            TimeoutError,
            KeyError,
            ValueError,
            OSError,
        ):
            df = None
        return t, df

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t, df = fut.result()
            if df is not None and not df.empty and len(df) > 250:
                ohlcv[t] = df.sort_values("date").reset_index(drop=True)
            if i % 100 == 0:
                log.info(
                    "scan.fetch_progress",
                    market=market,
                    fetched=i,
                    total=len(tickers),
                    usable=len(ohlcv),
                )
    log.info("scan.usable_tickers", market=market, usable=len(ohlcv))
    return ohlcv


def scan_market(market: str, strategies: dict, ohlcv: dict[str, pd.DataFrame]) -> dict:
    today = pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=LOOKBACK_DAYS)

    # signals[strategy] = list of (ticker, entry_date, days_since)
    signals: dict[str, list] = defaultdict(list)
    # ticker_count[ticker] = set of strategies that flagged it
    ticker_strategies: dict[str, set] = defaultdict(set)
    # latest_close[ticker] = last close
    latest_close: dict[str, float] = {}
    latest_date: dict[str, pd.Timestamp] = {}

    n_strats = len(strategies)
    for s_idx, (sname, sfn) in enumerate(strategies.items(), 1):
        if s_idx % 10 == 0:
            log.info(
                "scan.strategy_progress",
                market=market,
                index=s_idx,
                total=n_strats,
                strategy=sname,
            )
        for ticker, df in ohlcv.items():
            try:
                trades = sfn(df)
            except (ValueError, KeyError, TypeError, IndexError, RuntimeError):
                continue
            if not trades:
                continue
            # Fresh entries only — entry within the lookback window
            for tr in trades:
                ed = pd.Timestamp(tr.entry_date)
                if ed >= cutoff:
                    days_since = (today - ed).days
                    signals[sname].append(
                        (ticker, str(ed.date()), days_since, float(tr.entry_px))
                    )
                    ticker_strategies[ticker].add(sname)

            last_row = df.iloc[-1]
            latest_close[ticker] = float(last_row["close"])
            latest_date[ticker] = pd.Timestamp(last_row["date"])

    # Frequency table
    freq = Counter({t: len(s) for t, s in ticker_strategies.items()})

    return {
        "market": market,
        "as_of": str(today.date()),
        "lookback_days": LOOKBACK_DAYS,
        "n_strategies_scanned": n_strats,
        "n_tickers_scanned": len(ohlcv),
        "frequency": dict(freq.most_common()),
        "ticker_strategies": {t: sorted(ss) for t, ss in ticker_strategies.items()},
        "signals_by_strategy": {k: v for k, v in signals.items() if v},
        "latest_close": latest_close,
        "latest_date": {t: str(d.date()) for t, d in latest_date.items()},
    }


def main():
    strategies = load_good_strategies()
    log.info("scan.strategies_loaded", count=len(strategies))

    out = {}
    for market in ("us", "india"):
        ohlcv = fetch_universe(market, LIMIT)
        out[market] = scan_market(market, strategies, ohlcv)

    with open("picks_today.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Markdown summary
    lines = ["# Today's buy picks — frequency across profitable strategies", ""]
    lines.append(
        f"**Scan date:** {date.today().isoformat()}  •  **Lookback:** {LOOKBACK_DAYS}d  •  "
        f"**Strategies:** {len(strategies)}"
    )
    lines.append("")
    for market in ("us", "india"):
        m = out[market]
        bench = BENCHMARKS[market]
        lines.append(f"## {market.upper()} (benchmark {bench})")
        lines.append("")
        lines.append(f"- Tickers scanned: {m['n_tickers_scanned']}")
        lines.append(f"- Tickers with at least 1 fresh signal: {len(m['frequency'])}")
        lines.append("")
        lines.append("### Top 25 by strategy-frequency")
        lines.append("")
        lines.append(
            "| # | Ticker | # Strategies | Last close | Last bar | Strategies |"
        )
        lines.append(
            "|---|--------|--------------|-----------:|----------|------------|"
        )
        for i, (ticker, n) in enumerate(list(m["frequency"].items())[:25], 1):
            close = m["latest_close"].get(ticker, 0.0)
            ldate = m["latest_date"].get(ticker, "")
            strats = ", ".join(
                s.split(":", 1)[1] for s in m["ticker_strategies"][ticker][:6]
            )
            extra = (
                ""
                if len(m["ticker_strategies"][ticker]) <= 6
                else f" (+{len(m['ticker_strategies'][ticker]) - 6} more)"
            )
            lines.append(
                f"| {i} | **{ticker}** | {n} | {close:.2f} | {ldate} | {strats}{extra} |"
            )
        lines.append("")
    Path("picks_today.md").write_text("\n".join(lines))
    log.info("scan.outputs_written", json="picks_today.json", md="picks_today.md")
    # Print top 10 each market to stdout
    for market in ("us", "india"):
        print(f"\nTOP 10 {market.upper()}")
        for ticker, n in list(out[market]["frequency"].items())[:10]:
            print(f"  {ticker:<15} {n} strategies")


if __name__ == "__main__":
    main()
