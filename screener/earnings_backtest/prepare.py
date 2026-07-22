"""Shared earnings-run preparation.

Both earnings-drift (:mod:`screener.earnings_backtest.engine`) and PEAD
(:mod:`screener.earnings_backtest.pead`) perform the same acquisition and
orchestration before they diverge into their genuinely different entry/exit
policies:

    resolve universe -> compute date cutoff -> collect earnings events ->
    normalize/filter event dates -> (policy refinement) -> resolve event
    tickers -> fetch price panels.

That common skeleton lives here in :func:`prepare_earnings_run`; the two engines
supply small policy hooks (event refinement and the price window) and keep their
own event-to-trade loops. Point-in-time semantics are preserved verbatim: the
same ``[cutoff, today]`` announcement window is applied here for both engines,
and the fetch/collect callables are injected by the caller so their module-level
seams (monkeypatched in tests) remain the authoritative ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Optional

import pandas as pd

from screener.backtester.data import PriceFetcher

# A callable that yields the earnings-event frame for a universe. Injected by the
# caller (each engine passes its own module-level ``collect_earnings_events`` so
# test monkeypatches on that name stay in effect).
CollectEvents = Callable[..., pd.DataFrame]

# A callable that fetches OHLCV panels for tickers. Injected likewise so each
# engine's module-level ``fetch_price_data`` seam remains authoritative.
FetchPrices = Callable[..., dict[str, pd.DataFrame]]

# Refines the (already date-filtered) event frame per policy — e.g. PEAD's EPS
# surprise coercion and threshold. Returns the frame to trade over.
RefineEvents = Callable[[pd.DataFrame], pd.DataFrame]

# Computes the ``(start, end)`` price window from the event frame and the run's
# announcement cutoff.
PriceWindow = Callable[[pd.DataFrame, date], tuple[date, date]]


@dataclass(frozen=True)
class EventsAndPrices:
    """The prepared inputs shared by every earnings policy."""

    events: pd.DataFrame
    prices: dict[str, pd.DataFrame]


def prepare_earnings_run(
    *,
    market: str,
    years: int,
    batch_size: int,
    tickers: Optional[list[str]],
    load_universe: Callable[[str], list[str]],
    collect_events: CollectEvents,
    fetch_prices: FetchPrices,
    price_window: PriceWindow,
    collect_kwargs: Optional[dict[str, Any]] = None,
    refine_events: Optional[RefineEvents] = None,
    fetcher: Optional[PriceFetcher] = None,
) -> EventsAndPrices:
    """Resolve events and price panels shared by both earnings engines.

    The announcement window ``[today - years*365d, today]`` is applied to keep
    only *past* earnings (we need the exit price), preserving the point-in-time
    guarantee both engines relied on. ``refine_events`` and ``price_window`` are
    the only policy-specific seams; everything else is common orchestration.
    """
    if tickers is None:
        tickers = load_universe(market)

    cutoff_date = date.today() - timedelta(days=years * 365)
    events_df = collect_events(
        tickers,
        years=years,
        batch_size=batch_size,
        market=market,
        **(collect_kwargs or {}),
    )
    if events_df.empty:
        return EventsAndPrices(events_df, {})

    events_df = events_df.copy()
    events_df["earnings_date"] = pd.to_datetime(events_df["earnings_date"])
    events_df = events_df[
        (events_df["earnings_date"] >= pd.Timestamp(cutoff_date))
        & (events_df["earnings_date"] <= pd.Timestamp(date.today()))
    ]
    if refine_events is not None:
        events_df = refine_events(events_df)
    if events_df.empty:
        return EventsAndPrices(events_df, {})

    event_tickers = events_df["ticker"].unique().tolist()
    start, end = price_window(events_df, cutoff_date)
    price_data = fetch_prices(
        event_tickers, start, end, fetcher=fetcher, batch_size=batch_size
    )
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    return EventsAndPrices(events_df, price_data)


__all__ = ["EventsAndPrices", "prepare_earnings_run"]
