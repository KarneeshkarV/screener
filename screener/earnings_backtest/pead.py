"""Post-earnings-announcement drift (PEAD) backtest.

Selects earnings events whose EPS surprise exceeds a threshold, enters at
the open of the first trading day after the announcement, holds for a fixed
number of trading days (entry day counts as day 1), and exits at that bar's
close. Reuses the earnings-event and price plumbing of this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.earnings_backtest.data import (
    collect_earnings_events,
    fetch_price_data,
    load_universe,
)
from screener.earnings_backtest.engine import compute_backtest_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeadTrade:
    """One executed PEAD trade (next-open entry, close exit after N sessions)."""

    ticker: str
    earnings_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    return_pct: float
    surprise_pct: float
    holding_days: int
    passed_filter: bool = True
    details: dict = field(default_factory=dict)


def run_pead_backtest(
    market: str,
    years: int = 3,
    min_surprise: float = 5.0,
    hold_days: int = 40,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    batch_size: int = 50,
    tickers: Optional[list[str]] = None,
    fetcher: Optional[PriceFetcher] = None,
) -> list[PeadTrade]:
    """Run the PEAD backtest and return the trade ledger.

    Steps:
      1. Load universe tickers.
      2. Collect earnings events (with EPS surprise) via the module's plumbing.
      3. Keep events with ``surprise_pct >= min_surprise``.
      4. Enter at the next trading day's open, exit at the close
         ``hold_days`` trading days later (entry day counts as day 1).
    """
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")

    # 1. Universe
    if tickers is None:
        tickers = load_universe(market)
    logger.info("universe_loaded", extra={"market": market, "count": len(tickers)})

    # 2. Earnings events with surprise data
    cutoff_date = date.today() - timedelta(days=years * 365)
    events_df = collect_earnings_events(
        tickers, years=years, batch_size=batch_size, market=market
    )
    if events_df.empty:
        logger.warning("no_earnings_events_found")
        return []

    events_df = events_df.copy()
    events_df["earnings_date"] = pd.to_datetime(events_df["earnings_date"])
    events_df["surprise_pct"] = pd.to_numeric(
        events_df["surprise_pct"], errors="coerce"
    )
    events_df = events_df[
        (events_df["earnings_date"] >= pd.Timestamp(cutoff_date))
        & (events_df["earnings_date"] <= pd.Timestamp(date.today()))
    ]

    # 3. Surprise filter (events without surprise data are dropped)
    events_df = events_df.dropna(subset=["surprise_pct"])
    events_df = events_df[events_df["surprise_pct"] >= min_surprise]
    logger.info("pead_events_selected", extra={"count": len(events_df)})
    if events_df.empty:
        return []

    # Price window: a few days before the first event through the drift window
    event_tickers = events_df["ticker"].unique().tolist()
    earliest = (events_df["earnings_date"].min() - pd.Timedelta(days=5)).date()
    # hold_days trading days ≈ hold_days * 7/5 calendar days, plus buffer
    latest = (
        events_df["earnings_date"].max() + pd.Timedelta(days=int(hold_days * 1.6) + 10)
    ).date()

    price_data = fetch_price_data(
        event_tickers, earliest, latest, fetcher=fetcher, batch_size=batch_size
    )
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    logger.info("price_data_fetched", extra={"tickers": len(price_data)})

    # 4. Simulate next-open entry → N-session hold → close exit
    trades: list[PeadTrade] = []
    for _, event in events_df.iterrows():
        ticker = event["ticker"]
        ed = pd.Timestamp(event["earnings_date"]).normalize()

        bars = price_data.get(ticker)
        if bars is None or bars.empty:
            continue

        post_bars = bars[bars.index > ed]
        if post_bars.empty:
            continue

        entry_idx = bars.index.get_indexer([post_bars.index[0]])[0]
        exit_idx = entry_idx + hold_days - 1
        if exit_idx >= len(bars):
            # Incomplete drift window (e.g. recent earnings): skip
            continue

        entry_price = float(bars.iloc[entry_idx]["open"])
        exit_price = float(bars.iloc[exit_idx]["close"])
        if entry_price <= 0:
            continue

        entry_price *= 1 + slippage_bps / 10_000
        exit_price *= 1 - slippage_bps / 10_000

        ret_raw = (exit_price / entry_price) - 1.0
        ret_net = ret_raw - commission_bps / 10_000

        entry_ts = bars.index[entry_idx]
        exit_ts = bars.index[exit_idx]
        trades.append(
            PeadTrade(
                ticker=ticker,
                earnings_date=ed.date(),
                entry_date=entry_ts.date(),
                exit_date=exit_ts.date(),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                return_pct=round(ret_net * 100, 4),
                surprise_pct=round(float(event["surprise_pct"]), 4),
                holding_days=hold_days,
                details={"raw_return_pct": round(ret_raw * 100, 4)},
            )
        )

    logger.info("pead_backtest_complete", extra={"trades": len(trades)})
    return trades


def compute_pead_summary(
    trades: list[PeadTrade],
    min_surprise: float,
    hold_days: int,
) -> dict:
    """Aggregate PEAD drift statistics plus a by-surprise-quintile breakdown."""
    summary = compute_backtest_summary(trades, strategy="pead")
    summary["min_surprise_pct"] = min_surprise
    summary["hold_days"] = hold_days
    summary["surprise_quintiles"] = surprise_quintiles(trades)
    return summary


def surprise_quintiles(trades: list[PeadTrade]) -> dict[str, dict[str, float]]:
    """Return drift stats per EPS-surprise quintile (Q1 lowest … Q5 highest).

    Returns an empty dict when there are too few trades or too little
    surprise dispersion to form at least two bins.
    """
    if len(trades) < 5:
        return {}
    df = pd.DataFrame(
        {
            "surprise": [t.surprise_pct for t in trades],
            "ret": [t.return_pct for t in trades],
        }
    )
    try:
        df["bin"] = pd.qcut(df["surprise"], 5, labels=False, duplicates="drop")
    except ValueError:
        return {}
    if df["bin"].nunique() < 2:
        return {}

    out: dict[str, dict[str, float]] = {}
    for bin_id, grp in df.groupby("bin"):
        out[f"Q{int(bin_id) + 1}"] = {
            "trades": int(len(grp)),
            "avg_surprise_pct": round(float(grp["surprise"].mean()), 4),
            "avg_return_pct": round(float(grp["ret"].mean()), 4),
            "median_return_pct": round(float(grp["ret"].median()), 4),
            "win_rate": round(float((grp["ret"] > 0).mean()) * 100, 2),
        }
    return out
