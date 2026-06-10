"""Earnings-drift backtest engine.

Orchestrates data fetching, strategy evaluation, and P&L computation
for the E-1/E-2 → E entry/exit pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from screener.earnings_backtest.data import (
    collect_earnings_events,
    fetch_analyst_sentiment,
    fetch_iv_sentiment,
    fetch_price_data,
    load_universe,
)
from screener.earnings_backtest.strategies import (
    STRATEGY_FUNCS,
    combined_score,
)

logger = logging.getLogger(__name__)

# ── Trade result ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EarningsTrade:
    ticker: str
    earnings_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    return_pct: float
    strategy: str
    score: float
    passed_filter: bool
    details: dict = field(default_factory=dict)


# ── Core engine ──────────────────────────────────────────────────────────


def run_earnings_backtest(
    market: str,
    years: int = 3,
    strategy: str = "combined_score",
    days_before: int = 1,
    min_score: float = 0.55,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    batch_size: int = 50,
    tickers: Optional[list[str]] = None,
) -> list[EarningsTrade]:
    """Run the earnings-drift backtest.

    Steps:
      1. Load universe tickers.
      2. Collect earnings dates from yfinance.
      3. Fetch price data around each earnings event.
      4. Compute strategy scores for each event.
      5. Apply min_score filter.
      6. Simulate buy-close-E-N / sell-close-E trades.
      7. Return list of EarningsTrade objects.
    """
    # 1. Universe
    if tickers is None:
        tickers = load_universe(market)
    logger.info("universe_loaded", extra={"market": market, "count": len(tickers)})

    # 2. Earnings events
    cutoff_date = date.today() - timedelta(days=years * 365)
    events_df = collect_earnings_events(
        tickers, years=years, batch_size=batch_size, market=market
    )
    if events_df.empty:
        logger.warning("no_earnings_events_found")
        return []

    # Filter to past earnings (we need the exit price)
    events_df["earnings_date"] = pd.to_datetime(events_df["earnings_date"])
    events_df = events_df[
        (events_df["earnings_date"] >= pd.Timestamp(cutoff_date))
        & (events_df["earnings_date"] <= pd.Timestamp(date.today()))
    ]
    logger.info("earnings_events_collected", extra={"count": len(events_df)})

    if events_df.empty:
        return []

    # 3. Fetch price data (batched for RAM)
    # We need bars from ~25 days before E to E itself
    all_tickers_in_events = events_df["ticker"].unique().tolist()
    earliest = (events_df["earnings_date"].min() - pd.Timedelta(days=30)).date()
    latest = (events_df["earnings_date"].max() + pd.Timedelta(days=5)).date()

    # Free up: only keep events for tickers we actually have data for
    price_start = max(earliest, cutoff_date - timedelta(days=30))

    # Fetch in batches to control RAM
    price_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(all_tickers_in_events), batch_size):
        batch = all_tickers_in_events[i : i + batch_size]
        data = fetch_price_data(batch, price_start, latest)
        price_data.update(data)
        # Don't keep empty frames
        price_data = {k: v for k, v in price_data.items() if not v.empty}

    logger.info("price_data_fetched", extra={"tickers": len(price_data)})

    # 4-6. Evaluate strategies and simulate trades
    trades: list[EarningsTrade] = []
    analyzed_strategies = _resolve_strategies(strategy)

    # These live providers expose current snapshots only. Cache by entry/as-of
    # date and only use them when the snapshot is point-in-time safe.
    analyst_cache: dict[tuple[str, date], Optional[dict]] = {}
    iv_cache: dict[tuple[str, date], Optional[dict]] = {}

    # Process each earnings event
    for _, event in events_df.iterrows():
        ticker = event["ticker"]
        ed = pd.Timestamp(event["earnings_date"])

        bars = price_data.get(ticker)
        if bars is None or bars.empty:
            continue

        # Find the E-N bar (entry) and E bar (exit)
        entry_date, exit_date = _find_entry_exit(bars, ed, days_before)
        if entry_date is None or exit_date is None:
            continue

        entry_bar = bars[bars.index == pd.Timestamp(entry_date)]
        exit_bar = bars[bars.index == pd.Timestamp(exit_date)]

        if entry_bar.empty or exit_bar.empty:
            continue

        entry_price = float(entry_bar.iloc[-1]["close"])
        exit_price = float(exit_bar.iloc[-1]["close"])

        # Apply slippage and commission
        entry_price *= 1 + slippage_bps / 10_000
        exit_price *= 1 - slippage_bps / 10_000
        round_trip_commission = (
            commission_bps / 10_000
        )  # already in bps terms, applied round-trip

        # Evaluate strategies
        scores: dict[str, float] = {}
        signal_details: dict[str, dict] = {}

        for strat_name in analyzed_strategies:
            func = STRATEGY_FUNCS[strat_name]
            if strat_name == "price_momentum":
                result = func(
                    ticker,
                    ed,
                    bars,
                    threshold=0.0,
                    as_of_date=pd.Timestamp(entry_date),
                )
            elif strat_name == "volume_surge":
                result = func(
                    ticker,
                    ed,
                    bars,
                    threshold=0.0,
                    as_of_date=pd.Timestamp(entry_date),
                )
            elif strat_name == "analyst_sentiment":
                if not _can_use_current_snapshot(entry_date):
                    signal_details[strat_name] = _historical_snapshot_unavailable(
                        entry_date
                    )
                    continue
                analyst_key = (ticker, entry_date)
                if analyst_key not in analyst_cache:
                    analyst_cache[analyst_key] = fetch_analyst_sentiment(ticker, market)
                result = func(ticker, ed, analyst_cache.get(analyst_key), threshold=0.0)
            elif strat_name == "iv_sentiment":
                if not _can_use_current_snapshot(entry_date):
                    signal_details[strat_name] = _historical_snapshot_unavailable(
                        entry_date
                    )
                    continue
                iv_key = (ticker, entry_date)
                if iv_key not in iv_cache:
                    iv_cache[iv_key] = fetch_iv_sentiment(ticker, market)
                result = func(ticker, ed, iv_cache.get(iv_key), threshold=0.0)
            else:
                continue
            scores[strat_name] = result.score
            signal_details[strat_name] = result.details

        # Compute combined score if needed
        if strategy == "combined_score":
            final_score = combined_score(scores)
        elif strategy in scores:
            final_score = scores[strategy]
        else:
            final_score = combined_score(scores)

        passed_filter = final_score >= min_score

        # Only record trade if the strategy filter passes
        ret_raw = (exit_price / entry_price) - 1.0
        ret_net = ret_raw - round_trip_commission

        trade = EarningsTrade(
            ticker=ticker,
            earnings_date=ed.date() if hasattr(ed, "date") else ed,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            return_pct=round(ret_net * 100, 4),
            strategy=strategy,
            score=final_score,
            passed_filter=passed_filter,
            details={
                "scores": scores,
                "signals": signal_details,
                "raw_return_pct": round(ret_raw * 100, 4),
            },
        )
        trades.append(trade)

    logger.info("backtest_complete", extra={"trades": len(trades)})
    return trades


def _can_use_current_snapshot(as_of_date: date) -> bool:
    """Return whether current-only sentiment data is safe for this as-of date."""
    return as_of_date >= date.today()


def _historical_snapshot_unavailable(as_of_date: date) -> dict[str, str]:
    return {
        "reason": "current_snapshot_unavailable_for_historical_entry",
        "as_of_date": as_of_date.isoformat(),
    }


def _resolve_strategies(strategy: str) -> list[str]:
    """Return list of strategy names to evaluate."""
    if strategy == "combined_score":
        return list(STRATEGY_FUNCS.keys())
    if strategy in STRATEGY_FUNCS:
        return [strategy]
    raise ValueError(
        f"Unknown strategy: {strategy!r}. Known: {list(STRATEGY_FUNCS.keys()) + ['combined_score']}"
    )


def _find_entry_exit(
    bars: pd.DataFrame,
    earnings_date: pd.Timestamp,
    days_before: int,
) -> tuple[Optional[date], Optional[date]]:
    """Find the entry date (E-days_before) and exit date (E) from price bars.

    E is the earnings day: we use the bar ON or JUST BEFORE the earnings date.
    E-N is N trading days before E.

    Returns (entry_date, exit_date) or (None, None) if not found.
    """
    ed = pd.Timestamp(earnings_date).normalize()

    # Find the exit bar: the bar on or just before earnings_date
    exit_bars = bars[bars.index <= ed]
    if exit_bars.empty:
        return None, None

    exit_idx_raw = bars.index.get_loc(exit_bars.index[-1])
    if isinstance(exit_idx_raw, slice):
        # Fallback: use integer position
        exit_idx = (
            len(bars)
            - 1
            - (len(bars) - 1 - list(bars.index).index(exit_bars.index[-1]))
        )
    else:
        exit_idx = int(exit_idx_raw)

    if exit_idx < days_before:
        return None, None

    entry_idx = exit_idx - days_before
    if entry_idx < 0:
        return None, None

    exit_ts = bars.index[exit_idx]
    entry_ts = bars.index[entry_idx]

    entry_date = entry_ts.date() if hasattr(entry_ts, "date") else entry_ts
    exit_date = exit_ts.date() if hasattr(exit_ts, "date") else exit_ts
    return entry_date, exit_date


# ── Summary statistics ──────────────────────────────────────────────────


def compute_backtest_summary(trades: list[EarningsTrade], strategy: str = "") -> dict:
    """Compute aggregate backtest statistics."""
    if not trades:
        return {
            "total_events": 0,
            "trades_taken": 0,
            "strategy": strategy,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_winner_pct": 0.0,
            "max_loser_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_approx": 0.0,
        }

    taken = [t for t in trades if t.passed_filter]
    if not taken:
        return {
            "total_events": len(trades),
            "trades_taken": 0,
            "strategy": strategy,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_winner_pct": 0.0,
            "max_loser_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_approx": 0.0,
        }

    returns = np.array([t.return_pct for t in taken])
    winners = returns[returns > 0]
    losers = returns[returns < 0]

    holding_days = [(t.exit_date - t.entry_date).days for t in taken]
    avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

    # Sharpe approximation (annualised assuming avg holding period)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        avg_annualized = np.mean(returns) / avg_holding * 252 if avg_holding > 0 else 0
        std_annualized = (
            np.std(returns) / np.sqrt(avg_holding) * np.sqrt(252)
            if avg_holding > 0
            else 1
        )
        sharpe = round(
            avg_annualized / std_annualized if std_annualized > 0 else 0.0, 4
        )

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_events": len(trades),
        "trades_taken": len(taken),
        "strategy": strategy,
        "win_rate": round(float((returns > 0).mean()) * 100, 2),
        "avg_return_pct": round(float(returns.mean()), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "total_return_pct": round(float(returns.sum()), 4),
        "max_winner_pct": round(float(returns.max()), 4) if len(returns) > 0 else 0.0,
        "max_loser_pct": round(float(returns.min()), 4) if len(returns) > 0 else 0.0,
        "profit_factor": round(profit_factor, 4),
        "avg_holding_days": round(avg_holding, 2),
        "sharpe_approx": sharpe,
    }
