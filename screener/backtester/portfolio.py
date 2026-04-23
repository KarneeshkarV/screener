"""Explicit position + cash accounting for the backtester.

Each slot has a fixed ``slot_capital = initial_capital / slot_count`` budget
ceiling. At each ``open`` we spend up to ``min(slot_capital, current_cash)`` of
cash to fill shares (the cap prevents negative cash when a slot is reused
after a losing trade and cumulative losses have eroded the pool). At exit we
receive ``shares * exit_price - exit_commission`` back into cash.

The equity curve is cash + mark-to-market of open positions. When the engine
uses the event-driven reallocation path, closed-trade proceeds return to
``_cash`` and fund subsequent ``open`` calls on the same slot (a reserve
ticker fills the freed slot). Realized gains that exceed ``slot_capital`` stay
as idle cash within the slot — per-slot sizing is not compounded, to keep
sizing balanced across slots regardless of lucky-early-trade effects.

Concurrent positions per ticker (pyramiding) are supported internally by
keying ``_open`` on ``(ticker, open_seq)``. Legacy callers that pass ticker
only continue to work: they target the oldest-open position (FIFO) and the
``raise_if_exists=True`` flag preserves the historical invariant that a single
ticker cannot be opened twice through the legacy API.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from screener.backtester.models import ExitReason, Position, Trade


class Portfolio:
    def __init__(self, initial_capital: float, slot_count: int) -> None:
        if slot_count <= 0:
            raise ValueError("slot_count must be > 0")
        self.initial_capital = float(initial_capital)
        self.slot_count = slot_count
        self.slot_capital = self.initial_capital / slot_count
        self._cash = self.initial_capital
        # Keyed by (ticker, open_seq). Legacy callers use ticker only; helper
        # methods resolve to the FIFO-oldest open position for that ticker.
        self._open: dict[tuple[str, int], Position] = {}
        self._open_seq: dict[str, int] = {}
        self._closed: list[Trade] = []
        self._ranks: dict[str, int] = {}
        self._signal_dates: dict[str, date] = {}

    def assign(self, ticker: str, rank: int, signal_date: date) -> None:
        self._ranks[ticker] = rank
        self._signal_dates[ticker] = signal_date

    def _active_keys(self, ticker: str) -> list[tuple[str, int]]:
        return [k for k in self._open if k[0] == ticker]

    def _oldest_key(self, ticker: str) -> tuple[str, int] | None:
        keys = self._active_keys(ticker)
        if not keys:
            return None
        return min(keys, key=lambda k: k[1])

    def open(
        self,
        ticker: str,
        entry_date: date,
        entry_price: float,
        commission_bps: float,
        *,
        raise_if_exists: bool = True,
    ) -> Position:
        """Open a position for ``ticker``. By default raises if the ticker is
        already active (legacy invariant). Pass ``raise_if_exists=False`` to
        allow pyramiding: a new ``open_seq`` is allocated and the position is
        tracked as a distinct concurrent lot.
        """
        if raise_if_exists and self._active_keys(ticker):
            raise ValueError(f"Position already open for {ticker}")
        # spend up to min(slot_capital, current cash); commission reduces shares
        # acquired. Cap by current cash so reserve promotion after losing trades
        # cannot overdraw the portfolio.
        c = commission_bps / 10_000.0
        gross_per_share = entry_price * (1.0 + c)
        budget = min(self.slot_capital, max(self._cash, 0.0))
        shares = budget / gross_per_share if gross_per_share > 0 else 0.0
        notional = shares * entry_price
        commission = notional * c
        entry_cost = notional + commission  # <= budget by construction
        self._cash -= entry_cost
        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_fill=entry_price,
            shares=shares,
            slot_capital=entry_cost,
            peak_price=entry_price,
        )
        seq = self._open_seq.get(ticker, 0) + 1
        self._open_seq[ticker] = seq
        self._open[(ticker, seq)] = position
        return position

    def update_peak(self, ticker: str, high: float) -> None:
        key = self._oldest_key(ticker)
        if key is None:
            return
        pos = self._open[key]
        if high > pos.peak_price:
            pos.peak_price = high

    def credit_dividends(self, ticker: str, cash_per_share: float) -> float:
        """Credit ``shares * cash_per_share`` to portfolio cash for every open
        lot of ``ticker`` on an ex-dividend date. Returns the total dividend
        income credited across all lots.

        Each position's ``dividend_income`` accumulator is bumped so the
        ``Trade`` emitted when the lot finally closes carries the correct
        split between capital-return PnL and income-return PnL. Models the
        cash-account convention: the holder of record pockets the dividend
        as portfolio cash rather than as an implicit boost to OHLC (the
        auto_adjust regime, which conflates capital and income return).
        """
        if cash_per_share <= 0:
            return 0.0
        total = 0.0
        for key, pos in self._open.items():
            if key[0] != ticker or pos.shares <= 0:
                continue
            credit = pos.shares * cash_per_share
            self._cash += credit
            pos.dividend_income += credit
            total += credit
        return total

    def close(
        self,
        ticker: str,
        exit_date: date,
        exit_price: float,
        reason: ExitReason,
        commission_bps: float,
    ) -> Trade:
        """Fully close the oldest open position for ``ticker``."""
        key = self._oldest_key(ticker)
        if key is None:
            raise KeyError(f"No open position for {ticker}")
        position = self._open.pop(key)
        c = commission_bps / 10_000.0
        proceeds = position.shares * exit_price
        commission = proceeds * c
        exit_value = proceeds - commission
        self._cash += exit_value
        entry_cost = position.slot_capital
        pnl = exit_value - entry_cost
        return_pct = pnl / entry_cost if entry_cost else 0.0
        trade = Trade(
            ticker=ticker,
            rank=self._ranks.get(ticker, 0),
            signal_date=self._signal_dates.get(ticker, position.entry_date),
            entry_date=position.entry_date,
            entry_price=position.entry_fill,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=reason,
            shares=position.shares,
            entry_cost=entry_cost,
            exit_value=exit_value,
            pnl=pnl,
            return_pct=return_pct,
            dividend_income=position.dividend_income,
        )
        self._closed.append(trade)
        return trade

    def partial_close(
        self,
        ticker: str,
        exit_date: date,
        exit_price: float,
        reason: ExitReason,
        fraction: float,
        commission_bps: float,
    ) -> Trade:
        """Sell ``fraction`` of the ticker's oldest open position.

        The emitted Trade represents only the closed sleeve. Its ``entry_cost``
        is the pro-rata share of the original entry cost, so ``return_pct`` is
        comparable to a full-close trade. The remaining sleeve continues to
        accrue PnL against its reduced entry_cost.

        ``peak_price`` on the surviving sleeve is left unchanged — so the
        trailing-stop distance carries over from the pre-partial peak. This is
        intentional: the tranche that already ran to profit shouldn't reset
        the runner's trail.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        if fraction >= 1.0:
            return self.close(ticker, exit_date, exit_price, reason, commission_bps)
        key = self._oldest_key(ticker)
        if key is None:
            raise KeyError(f"No open position for {ticker}")
        position = self._open[key]
        close_shares = position.shares * fraction
        remaining_shares = position.shares - close_shares
        pro_rata_cost = position.slot_capital * fraction
        remaining_cost = position.slot_capital - pro_rata_cost
        pro_rata_div = position.dividend_income * fraction
        remaining_div = position.dividend_income - pro_rata_div
        c = commission_bps / 10_000.0
        proceeds = close_shares * exit_price
        commission = proceeds * c
        exit_value = proceeds - commission
        self._cash += exit_value
        pnl = exit_value - pro_rata_cost
        return_pct = pnl / pro_rata_cost if pro_rata_cost else 0.0
        trade = Trade(
            ticker=ticker,
            rank=self._ranks.get(ticker, 0),
            signal_date=self._signal_dates.get(ticker, position.entry_date),
            entry_date=position.entry_date,
            entry_price=position.entry_fill,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=reason,
            shares=close_shares,
            entry_cost=pro_rata_cost,
            exit_value=exit_value,
            pnl=pnl,
            return_pct=return_pct,
            dividend_income=pro_rata_div,
        )
        self._closed.append(trade)
        # shrink the remaining sleeve in place
        position.shares = remaining_shares
        position.slot_capital = remaining_cost
        position.dividend_income = remaining_div
        return trade

    def open_tickers(self) -> list[str]:
        return list({k[0] for k in self._open})

    def get_position(self, ticker: str) -> Position | None:
        key = self._oldest_key(ticker)
        return self._open.get(key) if key is not None else None

    def rank_of(self, ticker: str) -> int:
        """Selection rank previously assigned to ``ticker`` via :meth:`assign`,
        or ``0`` if the ticker was never assigned."""
        return self._ranks.get(ticker, 0)

    def closed_trades(self) -> list[Trade]:
        return list(self._closed)

    def cash(self) -> float:
        return self._cash


def build_equity_curve(
    calendar: pd.DatetimeIndex,
    trades: Iterable[Trade],
    price_panel: dict[str, pd.DataFrame],
    initial_capital: float,
) -> pd.Series:
    """Reconstruct the equity curve from a list of completed trades.

    On each calendar date, equity = cash + Σ shares * close for positions that
    are open that day (after applying all trade events dated <= that day, with
    entries processed before exits on the same day).
    """
    trades = list(trades)
    # Event list keyed by a monotonically-increasing trade sequence so two
    # trades on the same ticker (re-entry or pyramiding) are tracked
    # independently. Sort closes before opens on the same day so a
    # same-day close+reopen frees the slot before refilling.
    events: list[tuple[pd.Timestamp, int, int, Trade]] = []
    for seq, t in enumerate(trades):
        events.append((pd.Timestamp(t.entry_date), 1, seq, t))  # 1 = open
        events.append((pd.Timestamp(t.exit_date), 0, seq, t))   # 0 = close (first)
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    cash = float(initial_capital)
    open_positions: dict[int, Trade] = {}
    equity = pd.Series(0.0, index=calendar, dtype=float)
    ev_idx = 0

    for day in calendar:
        while ev_idx < len(events) and events[ev_idx][0] <= day:
            _, kind, seq, trade = events[ev_idx]
            if kind == 1:  # open
                cash -= trade.entry_cost
                open_positions[seq] = trade
            else:  # close
                open_positions.pop(seq, None)
                cash += trade.exit_value
            ev_idx += 1

        mtm = 0.0
        for trade in open_positions.values():
            frame = price_panel.get(trade.ticker)
            if frame is None or frame.empty:
                mtm += trade.shares * trade.entry_price
                continue
            if day in frame.index:
                price = float(frame.loc[day, "close"])
            else:
                prior = frame.loc[frame.index <= day]
                price = (
                    float(prior["close"].iloc[-1])
                    if not prior.empty
                    else trade.entry_price
                )
            mtm += trade.shares * price
        equity.loc[day] = cash + mtm
    return equity
