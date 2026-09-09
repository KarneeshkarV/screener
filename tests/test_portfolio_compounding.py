"""The per-slot budget must grow with realized equity by default.

Frozen slot capital silently de-levers any run that profits: the slot stays
at ``initial_capital / slot_count`` while equity grows, so an increasing share
of the book sits in idle cash and the measured volatility and drawdown decay
toward zero. These tests pin both modes so the difference stays deliberate.
"""

from __future__ import annotations

from datetime import date

from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio


def _round_trip(
    portfolio: Portfolio, ticker: str, entry: float, exit_price: float
) -> None:
    budget = portfolio.entry_budget()
    portfolio.assign(ticker, rank=1, signal_date=date(2024, 1, 1))
    portfolio.open(ticker, date(2024, 1, 1), entry, budget=budget)
    portfolio.close(ticker, date(2024, 2, 1), exit_price, "time")


def test_frozen_slot_budget_does_not_grow_with_profits() -> None:
    portfolio = Portfolio(100_000.0, 10, compounding=False)
    assert portfolio.entry_budget() == 10_000.0
    _round_trip(portfolio, "AAA", 10.0, 20.0)  # doubles the slot
    # Equity is now ~110k but the ceiling is still the day-one slot.
    assert portfolio.realized_equity() > 100_000.0
    assert portfolio.entry_budget() == 10_000.0


def test_compounding_slot_budget_tracks_realized_equity() -> None:
    portfolio = Portfolio(100_000.0, 10)
    assert portfolio.entry_budget() == 10_000.0
    _round_trip(portfolio, "AAA", 10.0, 20.0)
    equity = portfolio.realized_equity()
    assert equity > 100_000.0
    # The next slot gets a tenth of the grown pool, not of day-one capital.
    assert portfolio.entry_budget() == equity / 10
    assert portfolio.entry_budget() > 10_000.0


def test_compounding_shrinks_the_slot_after_a_loss() -> None:
    portfolio = Portfolio(100_000.0, 10)
    _round_trip(portfolio, "AAA", 10.0, 5.0)  # halves the slot
    assert portfolio.realized_equity() < 100_000.0
    assert portfolio.entry_budget() < 10_000.0


def test_realized_equity_counts_open_positions_at_cost() -> None:
    # An open position must not vanish from the pool the moment it is bought,
    # or every subsequent slot would be sized off a book that looks smaller
    # than it is.
    portfolio = Portfolio(100_000.0, 10)
    portfolio.assign("AAA", rank=1, signal_date=date(2024, 1, 1))
    portfolio.open("AAA", date(2024, 1, 1), 10.0, budget=portfolio.entry_budget())
    assert portfolio.realized_equity() == 100_000.0


def test_slots_stay_equal_to_each_other_under_compounding() -> None:
    # The frozen design exists to stop a lucky early trade from making one slot
    # permanently bigger. Dividing current equity by slot_count preserves that:
    # every open slot draws the same ceiling at the same moment.
    portfolio = Portfolio(100_000.0, 4)
    _round_trip(portfolio, "AAA", 10.0, 30.0)
    first = portfolio.entry_budget()
    portfolio.assign("BBB", rank=1, signal_date=date(2024, 3, 1))
    portfolio.open("BBB", date(2024, 3, 1), 10.0, budget=first)
    # Buying at the ceiling moves cash into basis, so the pool is unchanged and
    # the next slot is offered exactly the same budget.
    assert portfolio.entry_budget() == first


def test_compounding_is_on_by_default() -> None:
    assert Portfolio(100_000.0, 10).compounding is True
    assert (
        BacktestConfig(
            market="us",
            as_of=date(2024, 1, 1),
            benchmark="SPY",
            hold=5,
            top=1,
            strategy_name=None,
            entry_expr="close > 0",
            exit_expr=None,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=100_000.0,
        ).compounding
        is True
    )
