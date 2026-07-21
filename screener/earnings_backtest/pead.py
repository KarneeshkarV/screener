"""Post-earnings-announcement drift (PEAD) backtest.

Selects earnings events whose EPS surprise exceeds a threshold, enters at
the open of the first trading day after the announcement, holds for a fixed
number of trading days (entry day counts as day 1), and exits at that bar's
close. Reuses the earnings-event and price plumbing of this module.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional, cast

import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.backtester.execution import (
    fixed_bps_round_trip,
    net_round_trip_return,
)
from screener.earnings_backtest.data import (
    fetch_price_data,
    load_universe,
)
from screener.earnings_backtest.earnings_dates import collect_earnings_events
from screener.earnings_backtest.metrics import compute_backtest_summary
from screener.earnings_backtest.models import PeadTrade

logger = logging.getLogger(__name__)


EXIT_MODES = ("fixed", "dynamic")


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
    exit_mode: str = "fixed",
) -> list[PeadTrade]:
    """Run the PEAD backtest and return the trade ledger.

    Steps:
      1. Load universe tickers.
      2. Collect earnings events (with EPS surprise) via the module's plumbing.
         For India the surprise-capable Financial Modeling Prep source is used
         (``surprise_source="fmp"``); US already carries surprise via yfinance.
      3. Keep events with usable surprise data.
      4. Simulate, per *exit_mode*:

         * ``"fixed"`` (default): for every event with
           ``surprise_pct >= min_surprise``, enter at the next trading day's open
           and exit at the close ``hold_days`` trading days later (entry day is
           day 1). This is the original behaviour and is bit-for-bit unchanged.
         * ``"dynamic"``: process each ticker's events chronologically — enter at
           the next open after a qualifying report and hold until a subsequent
           report FAILS the ``min_surprise`` criterion (exit at the next open
           after that report), or force-exit at the last available close. The
           actual trading-day holding period and an ``exit_reason`` are recorded
           per trade. ``hold_days`` is ignored in this mode.
    """
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if exit_mode not in EXIT_MODES:
        raise ValueError(f"exit_mode must be one of {EXIT_MODES}")

    # 1. Universe
    if tickers is None:
        tickers = load_universe(market)
    logger.info("universe_loaded", extra={"market": market, "count": len(tickers)})

    # 2. Earnings events with surprise data. India's NSE/openscreener path has no
    #    EPS surprise, so route India through the FMP surprise source.
    surprise_source = "fmp" if market == "india" else None
    cutoff_date = date.today() - timedelta(days=years * 365)
    events_df = collect_earnings_events(
        tickers,
        years=years,
        batch_size=batch_size,
        market=market,
        surprise_source=surprise_source,
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

    # 3. Drop events without usable surprise data. The fixed mode also pre-filters
    #    on the threshold (its historical behaviour); the dynamic mode keeps the
    #    full known-surprise stream so it can detect the report that later fails
    #    the criterion and triggers the exit.
    events_df = events_df.dropna(subset=["surprise_pct"])
    if exit_mode == "fixed":
        events_df = events_df[events_df["surprise_pct"] >= min_surprise]
    logger.info("pead_events_selected", extra={"count": len(events_df)})
    if events_df.empty:
        return []

    # Price window: a few days before the first event through the exit window. In
    # dynamic mode a position can be held for many quarters, so the window runs
    # through today; in fixed mode it spans the bounded drift window.
    event_tickers = events_df["ticker"].unique().tolist()
    earliest = (events_df["earnings_date"].min() - pd.Timedelta(days=5)).date()
    if exit_mode == "dynamic":
        latest = date.today()
    else:
        # hold_days trading days ~= hold_days * 7/5 calendar days, plus buffer
        latest = (
            events_df["earnings_date"].max()
            + pd.Timedelta(days=int(hold_days * 1.6) + 10)
        ).date()

    price_data = fetch_price_data(
        event_tickers,
        earliest,
        latest,
        # TODO(plan-005): run_pead_backtest accepts the broad PriceFetcher
        # Protocol but fetch_price_data is typed for YFinancePriceFetcher;
        # widening fetch_price_data's param is a public-signature change.
        fetcher=fetcher,  # type: ignore[arg-type]
        batch_size=batch_size,
    )
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    logger.info("price_data_fetched", extra={"tickers": len(price_data)})

    # 4. Simulate per the requested exit mode.
    if exit_mode == "dynamic":
        trades = _simulate_dynamic_hold(
            events_df, price_data, min_surprise, commission_bps, slippage_bps
        )
    else:
        trades = _simulate_fixed_hold(
            events_df, price_data, hold_days, commission_bps, slippage_bps
        )

    logger.info("pead_backtest_complete", extra={"trades": len(trades)})
    return trades


def _simulate_fixed_hold(
    events_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    hold_days: int,
    commission_bps: float,
    slippage_bps: float,
) -> list[PeadTrade]:
    """Fixed next-open entry -> ``hold_days``-session hold -> close exit."""
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

        entry_idx = bars.index.get_indexer(pd.Index([post_bars.index[0]]))[0]
        exit_idx = entry_idx + hold_days - 1
        if exit_idx >= len(bars):
            # Incomplete drift window (e.g. recent earnings): skip.
            continue

        entry_price = float(bars.iloc[entry_idx]["open"])
        exit_price = float(bars.iloc[exit_idx]["close"])
        if entry_price <= 0:
            continue

        entry_price, exit_price = fixed_bps_round_trip(
            entry_price, exit_price, slippage_bps
        )

        ret_raw, ret_net = net_round_trip_return(
            entry_price, exit_price, commission_bps
        )

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
    return trades


def _dynamic_trade(
    ticker: str,
    position: dict[str, Any],
    exit_ts: Any,
    exit_price: float,
    bars: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    exit_reason: str,
) -> PeadTrade:
    """Build one dynamic-hold trade from an open *position* and its exit fill."""
    entry_ts = position["entry_ts"]
    entry_fill, exit_fill = fixed_bps_round_trip(
        position["entry_price"], exit_price, slippage_bps
    )
    ret_raw, ret_net = net_round_trip_return(entry_fill, exit_fill, commission_bps)
    # Actual trading-day holding period: bars inclusive of entry and exit.
    holding_days = int(len(bars[(bars.index >= entry_ts) & (bars.index <= exit_ts)]))
    return PeadTrade(
        ticker=ticker,
        earnings_date=position["earnings_date"].date(),
        entry_date=entry_ts.date() if hasattr(entry_ts, "date") else entry_ts,
        exit_date=exit_ts.date() if hasattr(exit_ts, "date") else exit_ts,
        entry_price=round(entry_fill, 4),
        exit_price=round(exit_fill, 4),
        return_pct=round(ret_net * 100, 4),
        surprise_pct=round(position["surprise"], 4),
        holding_days=holding_days,
        details={
            "raw_return_pct": round(ret_raw * 100, 4),
            "exit_reason": exit_reason,
        },
    )


def _simulate_dynamic_hold(
    events_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    min_surprise: float,
    commission_bps: float,
    slippage_bps: float,
) -> list[PeadTrade]:
    """Chronological per-ticker hold: enter on a beat, exit when a report fails.

    Enters at the next open after a report with ``surprise_pct >= min_surprise``
    and stays invested until a later report FAILS the criterion (exit at the next
    open after it, ``exit_reason="criteria_failed"``) or the price history ends
    while still holding (``exit_reason="end_of_data"``).
    """
    trades: list[PeadTrade] = []
    for ticker_key, group in events_df.groupby("ticker"):
        # groupby keys are typed as Hashable; the ticker column holds strings.
        ticker = str(ticker_key)
        bars = price_data.get(ticker)
        if bars is None or bars.empty:
            continue

        group = group.sort_values("earnings_date")
        position: Optional[dict[str, Any]] = None

        for _, event in group.iterrows():
            ed = pd.Timestamp(event["earnings_date"]).normalize()
            surprise = float(event["surprise_pct"])
            meets_criteria = surprise >= min_surprise

            if position is None and meets_criteria:
                post_bars = bars[bars.index > ed]
                if post_bars.empty:
                    continue
                entry_price = float(post_bars.iloc[0]["open"])
                if entry_price <= 0:
                    continue
                position = {
                    "entry_ts": post_bars.index[0],
                    "entry_price": entry_price,
                    "earnings_date": ed,
                    "surprise": surprise,
                }
            elif position is not None and not meets_criteria:
                post_bars = bars[bars.index > ed]
                if post_bars.empty:
                    exit_ts = bars.index[-1]
                    exit_price = float(bars.iloc[-1]["close"])
                else:
                    exit_ts = post_bars.index[0]
                    exit_price = float(post_bars.iloc[0]["open"])
                trades.append(
                    _dynamic_trade(
                        ticker,
                        position,
                        exit_ts,
                        exit_price,
                        bars,
                        commission_bps,
                        slippage_bps,
                        exit_reason="criteria_failed",
                    )
                )
                position = None

        # Still holding at the end of the price history: force-exit at last close.
        if position is not None:
            trades.append(
                _dynamic_trade(
                    ticker,
                    position,
                    bars.index[-1],
                    float(bars.iloc[-1]["close"]),
                    bars,
                    commission_bps,
                    slippage_bps,
                    exit_reason="end_of_data",
                )
            )
    return trades


def compute_pead_summary(
    trades: list[PeadTrade],
    min_surprise: float,
    hold_days: int,
) -> dict:
    """Aggregate PEAD drift statistics plus a by-surprise-quintile breakdown."""
    summary: dict[str, object] = dict(compute_backtest_summary(trades, strategy="pead"))
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
        out[f"Q{int(cast(Any, bin_id)) + 1}"] = {
            "trades": int(len(grp)),
            "avg_surprise_pct": round(float(grp["surprise"].mean()), 4),
            "avg_return_pct": round(float(grp["ret"].mean()), 4),
            "median_return_pct": round(float(grp["ret"].median()), 4),
            "win_rate": round(float((grp["ret"] > 0).mean()) * 100, 2),
        }
    return out


__all__ = [
    "EXIT_MODES",
    "PeadTrade",
    "compute_pead_summary",
    "run_pead_backtest",
    "surprise_quintiles",
]
