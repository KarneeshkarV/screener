"""Earnings-drift backtest engine.

Orchestrates data fetching, strategy evaluation, and P&L computation
for the E-1/E-2 → E entry/exit pattern.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from screener.backtester.costs import build_cost_model
from screener.backtester.execution import (
    apply_round_trip_costs,
    fixed_bps_round_trip,
)
from screener.earnings_backtest.data import (
    fetch_price_data,
    load_universe,
)
from screener.earnings_backtest.earnings_dates import collect_earnings_events
from screener.earnings_backtest.prepare import prepare_earnings_run
from screener.earnings_backtest.sentiment import (
    fetch_analyst_sentiment,
    fetch_iv_sentiment,
)
from screener.earnings_backtest.strategies import (
    STRATEGY_FUNCS,
    combined_score,
)
from screener.earnings_backtest.metrics import compute_backtest_summary
from screener.earnings_backtest.models import EarningsTrade

logger = logging.getLogger(__name__)

# ── Core engine ──────────────────────────────────────────────────────────


def run_earnings_backtest(
    market: str,
    years: int = 3,
    strategy: str = "combined_score",
    days_before: int = 1,
    min_score: float = 0.55,
    commission_bps: float = 10.0,
    cost_model: str = "flat",
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

    Fees use the shared :func:`~screener.backtester.costs.build_cost_model`
    stack (``flat`` / ``india`` / ``us_vested``). For ``cost_model="flat"``,
    ``commission_bps`` remains a **round-trip** total (legacy earnings CLI
    semantics); it is split evenly across buy and sell so
    :class:`~screener.backtester.costs.FlatCommission` per-side rates sum to
    the same drag as the previous single subtraction — bit-identical for flat.
    """
    # FlatCommission is per-fill; earnings historically treated commission_bps
    # as a single round-trip total. Split so buy+sell fractions match legacy.
    model_name = (cost_model or "flat").strip().lower()
    flat_bps = (
        float(commission_bps) / 2.0 if model_name == "flat" else float(commission_bps)
    )
    costs = build_cost_model(model_name, commission_bps=flat_bps)
    fees_paid: dict[str, float] = {}

    # Steps 1-6 (universe -> events -> price panels) are shared acquisition;
    # only the entry/exit policy below is drift-specific. ``collect_earnings_events``
    # and ``fetch_price_data`` are passed as this module's globals so their test
    # seams (monkeypatched below) stay authoritative inside the shared step.
    prepared = prepare_earnings_run(
        market=market,
        years=years,
        batch_size=batch_size,
        tickers=tickers,
        load_universe=load_universe,
        collect_events=collect_earnings_events,
        fetch_prices=fetch_price_data,
        price_window=_predrift_price_window,
    )
    events_df = prepared.events
    price_data = prepared.prices
    if events_df.empty:
        logger.warning("no_earnings_events_found")
        return []

    # Evaluate strategies and simulate the E-N -> E drift trades.
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

        if (
            entry_bar.empty or exit_bar.empty
        ):  # pragma: no cover - defensive: dates come from bars.index
            continue

        entry_price = float(entry_bar.iloc[-1]["close"])
        exit_price = float(exit_bar.iloc[-1]["close"])

        # Apply slippage and cost-model fees (fees do not move the fill price).
        entry_price, exit_price = fixed_bps_round_trip(
            entry_price, exit_price, slippage_bps
        )

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
            else:  # pragma: no cover - defensive: _resolve_strategies yields only known names
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

        ret_raw, ret_net, trade_fees = apply_round_trip_costs(
            entry_price, exit_price, costs
        )
        for name, amount in trade_fees.items():
            fees_paid[name] = fees_paid.get(name, 0.0) + amount

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
                "fees": trade_fees,
            },
        )
        trades.append(trade)

    # Stash run-level fee totals on the trade details so summary/CLI can surface
    # aggregates without changing the return type (list[EarningsTrade]).
    if trades:
        totals = dict(fees_paid)
        for trade in trades:
            trade.details["fees_paid_total"] = totals

    logger.info("backtest_complete", extra={"trades": len(trades)})
    return trades


def _predrift_price_window(
    events_df: pd.DataFrame, cutoff_date: date
) -> tuple[date, date]:
    """Drift window: ~30 sessions before the first event through E + 5 days.

    The start is clamped to ``cutoff_date - 30d`` so a stray far-past event
    cannot widen the fetch beyond the look-back the run asked for.
    """
    earliest = (events_df["earnings_date"].min() - pd.Timedelta(days=30)).date()
    latest = (events_df["earnings_date"].max() + pd.Timedelta(days=5)).date()
    price_start = max(earliest, cutoff_date - timedelta(days=30))
    return price_start, latest


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
    if isinstance(
        exit_idx_raw, slice
    ):  # pragma: no cover - defensive: bars have a unique index
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
    if (
        entry_idx < 0
    ):  # pragma: no cover - defensive: guarded by the exit_idx<days_before check above
        return None, None

    exit_ts = bars.index[exit_idx]
    entry_ts = bars.index[entry_idx]

    entry_date = entry_ts.date() if hasattr(entry_ts, "date") else entry_ts
    exit_date = exit_ts.date() if hasattr(exit_ts, "date") else exit_ts
    return entry_date, exit_date


__all__ = [
    "EarningsTrade",
    "compute_backtest_summary",
    "run_earnings_backtest",
]
