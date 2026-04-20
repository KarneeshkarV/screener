"""Explicit position + cash accounting for the backtester.

Each slot gets ``initial_capital / slot_count`` cash at t=0. At the entry bar
we spend ``shares * entry_price + entry_commission`` of cash to open a
position. At exit we receive ``shares * exit_price - exit_commission`` back.

The equity curve is cash + mark-to-market of open positions. Closed-trade
proceeds stay as cash and are NOT redeployed.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

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
        self._open: dict[str, Position] = {}
        self._closed: list[Trade] = []
        self._ranks: dict[str, int] = {}
        self._signal_dates: dict[str, date] = {}

    def assign(self, ticker: str, rank: int, signal_date: date) -> None:
        self._ranks[ticker] = rank
        self._signal_dates[ticker] = signal_date

    def open(
        self,
        ticker: str,
        entry_date: date,
        entry_price: float,
        commission_bps: float,
    ) -> Position:
        if ticker in self._open:
            raise ValueError(f"Position already open for {ticker}")
        # spend up to slot_capital; commission reduces shares acquired
        # Solve: shares * entry_price * (1 + c) = slot_capital - slack
        # For simplicity: shares = slot_capital / (entry_price * (1 + c))
        c = commission_bps / 10_000.0
        gross_per_share = entry_price * (1.0 + c)
        shares = self.slot_capital / gross_per_share if gross_per_share > 0 else 0.0
        notional = shares * entry_price
        commission = notional * c
        entry_cost = notional + commission  # <= slot_capital by construction
        self._cash -= entry_cost
        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_fill=entry_price,
            shares=shares,
            slot_capital=entry_cost,  # actual capital deployed
            peak_price=entry_price,
        )
        self._open[ticker] = position
        return position

    def update_peak(self, ticker: str, high: float) -> None:
        pos = self._open.get(ticker)
        if pos is not None and high > pos.peak_price:
            pos.peak_price = high

    def close(
        self,
        ticker: str,
        exit_date: date,
        exit_price: float,
        reason: ExitReason,
        commission_bps: float,
    ) -> Trade:
        position = self._open.pop(ticker)
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
        )
        self._closed.append(trade)
        return trade

    def open_tickers(self) -> list[str]:
        return list(self._open.keys())

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._open.get(ticker)

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
    events: list[tuple[pd.Timestamp, int, Trade]] = []
    for t in trades:
        events.append((pd.Timestamp(t.entry_date), 0, t))  # 0 = open
        events.append((pd.Timestamp(t.exit_date), 1, t))   # 1 = close
    events.sort(key=lambda e: (e[0], e[1]))

    cash = float(initial_capital)
    open_positions: dict[str, Trade] = {}
    equity = pd.Series(0.0, index=calendar, dtype=float)
    ev_idx = 0

    for day in calendar:
        while ev_idx < len(events) and events[ev_idx][0] <= day:
            _, kind, trade = events[ev_idx]
            if kind == 0:
                cash -= trade.entry_cost
                open_positions[trade.ticker] = trade
            else:
                open_positions.pop(trade.ticker, None)
                cash += trade.exit_value
            ev_idx += 1

        mtm = 0.0
        for ticker, trade in open_positions.items():
            frame = price_panel.get(ticker)
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
