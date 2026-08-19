"""A trade that opens and closes on one bar must not be marked to market.

``build_equity_curve`` applies both cash events for such a trade on the same
calendar day, so cash already carries the whole round trip. Adding the position
to that day's mark-to-market counts the notional a second time, which shows up
as the equity curve roughly doubling on the final bar of any strategy whose
rebalance lands there.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.models import Trade
from screener.backtester.portfolio import build_equity_curve


def _frame(days: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [price] * len(days)}, index=days)


def _round_trip(day: date, shares: float, price: float) -> Trade:
    notional = shares * price
    return Trade(
        entry_date=day,
        exit_date=day,
        ticker="AAA",
        rank=1,
        signal_date=day,
        entry_price=price,
        exit_price=price,
        exit_reason="end",
        shares=shares,
        entry_cost=notional,
        exit_value=notional,
        pnl=0.0,
        return_pct=0.0,
        dividend_income=0.0,
    )


def test_same_day_round_trip_leaves_equity_flat() -> None:
    days = pd.DatetimeIndex(pd.date_range("2026-01-05", periods=5, freq="B"))
    # Bought and sold at the same price on the last bar: the book is all cash
    # again, so equity must not move at all.
    trade = _round_trip(days[-1].date(), shares=500.0, price=100.0)

    equity = build_equity_curve(days, [trade], {"AAA": _frame(days, 100.0)}, 100_000.0)

    assert equity.iloc[-1] == pytest.approx(100_000.0)
    assert equity.iloc[-1] == pytest.approx(equity.iloc[-2])


def test_same_day_round_trip_mid_window_does_not_linger() -> None:
    # The stale branch marked the position for every *remaining* day, so a
    # mid-window same-day exit inflated the whole tail, not just one bar.
    days = pd.DatetimeIndex(pd.date_range("2026-01-05", periods=6, freq="B"))
    trade = _round_trip(days[1].date(), shares=500.0, price=100.0)

    equity = build_equity_curve(days, [trade], {"AAA": _frame(days, 100.0)}, 100_000.0)

    assert equity.to_numpy() == pytest.approx([100_000.0] * len(days))


def test_same_day_round_trip_still_books_its_pnl() -> None:
    # Zero mark-to-market must not mean zero effect: a same-day loss still has
    # to leave the book, via the cash legs.
    days = pd.DatetimeIndex(pd.date_range("2026-01-05", periods=4, freq="B"))
    trade = _round_trip(days[-1].date(), shares=500.0, price=100.0)
    trade = trade.model_copy(update={"exit_value": 49_000.0, "pnl": -1_000.0})

    equity = build_equity_curve(days, [trade], {"AAA": _frame(days, 100.0)}, 100_000.0)

    assert equity.iloc[-1] == pytest.approx(99_000.0)
