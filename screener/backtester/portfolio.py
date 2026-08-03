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
from datetime import date, datetime
from typing import Any, Union, cast

import numpy as np
import pandas as pd

from screener.backtester.costs import CostModel, FlatCommission, Side
from screener.backtester.models import Position, Trade
from screener.ledger import ExitReason

# Trade/position stamps are date-only for daily runs and full datetimes for
# intraday runs (see ``screener.backtester.core._bar_label``).
_Stamp = Union[date, datetime]


class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        slot_count: int,
        cost_model: CostModel | None = None,
    ) -> None:
        if slot_count <= 0:
            raise ValueError("slot_count must be > 0")
        self.initial_capital = float(initial_capital)
        self.slot_count = slot_count
        self.slot_capital = self.initial_capital / slot_count
        self.cost_model = cost_model or FlatCommission()
        # Running attribution of statutory/broker fees actually charged, keyed
        # by cost-model component name (e.g. "brokerage", "stt", "sec_fee",
        # "taf"). Populated on every buy/sell fill; see ``total_fees_paid``.
        self.fees_paid: dict[str, float] = {}
        self._cash = self.initial_capital
        # Keyed by (ticker, open_seq). Legacy callers use ticker only; helper
        # methods resolve to the FIFO-oldest open position for that ticker.
        self._open: dict[tuple[str, int], Position] = {}
        self._open_seq: dict[str, int] = {}
        self._closed: list[Trade] = []
        self._ranks: dict[str, int] = {}
        self._signal_dates: dict[str, _Stamp] = {}

    def assign(self, ticker: str, rank: int, signal_date: _Stamp) -> None:
        self._ranks[ticker] = rank
        self._signal_dates[ticker] = signal_date

    def entry_budget(self) -> float:
        """Cash available to the next slot, before entry price/commission."""
        return min(self.slot_capital, max(self._cash, 0.0))

    def _charge_fees(
        self,
        side: Side,
        notional: float,
        shares: float | None = None,
    ) -> float:
        """Return the total fee for a ``side`` fill and record its breakdown.

        Uses the cost model's per-component ``side_cost_breakdown`` when
        available (so per-share/capped components are exact), falling back to
        the fraction API for legacy cost models that only expose
        ``side_cost_fraction``. Negative amounts are clamped to 0. The returned
        total is the sum of the accumulated components.
        """
        breakdown_fn = getattr(self.cost_model, "side_cost_breakdown", None)
        if callable(breakdown_fn):
            breakdown = breakdown_fn(side, notional, shares)
        else:
            frac = float(self.cost_model.side_cost_fraction(side, notional))
            frac = max(frac, 0.0)
            breakdown = {"commission": notional * frac}
        total = 0.0
        for name, amount in breakdown.items():
            amt = float(amount)
            if amt <= 0.0:
                # Skip inapplicable components (e.g. sell-only fees on a buy)
                # so the attribution map lists only what was actually charged.
                continue
            self.fees_paid[name] = self.fees_paid.get(name, 0.0) + amt
            total += amt
        return total

    @property
    def total_fees_paid(self) -> float:
        """Sum of all statutory/broker fees charged over the run."""
        return float(sum(self.fees_paid.values()))

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
        entry_date: _Stamp,
        entry_price: float,
        *,
        raise_if_exists: bool = True,
        budget: float | None = None,
        shares: float | None = None,
    ) -> Position:
        """Open a position for ``ticker``. By default raises if the ticker is
        already active (legacy invariant). Pass ``raise_if_exists=False`` to
        allow pyramiding: a new ``open_seq`` is allocated and the position is
        tracked as a distinct concurrent lot.

        ``budget`` lets a sizing rule spend less than the slot budget; it is
        always clamped to ``entry_budget()`` so a rule can never exceed the
        slot ceiling or overdraw cash. When ``shares`` is supplied, it is the
        pre-impact count quoted by ``FillModel`` and is used unchanged. The
        portfolio remains the authority for the applicable fees and cash debit.

        Fees come from the cost model owned by this portfolio.
        """
        if raise_if_exists and self._active_keys(ticker):
            raise ValueError(f"Position already open for {ticker}")
        # spend up to min(slot_capital, current cash); fees reduce shares
        # acquired. Cap by current cash so reserve promotion after losing trades
        # cannot overdraw the portfolio.
        # Proportional fee models do not depend on notional; pass budget as a
        # stable reference for any future notional-dependent schedules.
        cap = self.entry_budget()
        budget = cap if budget is None else min(max(float(budget), 0.0), cap)
        if shares is None:
            c = float(self.cost_model.side_cost_fraction("buy", budget))
            c = max(c, 0.0)
            gross_per_share = entry_price * (1.0 + c)
            shares = budget / gross_per_share if gross_per_share > 0 else 0.0
        else:
            shares = max(float(shares), 0.0)
        notional = shares * entry_price
        commission = self._charge_fees("buy", notional, shares)
        entry_cost = notional + commission
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
        pos.peak_price = max(pos.peak_price, high)

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
        exit_date: _Stamp,
        exit_price: float,
        reason: ExitReason,
    ) -> Trade:
        """Fully close the oldest open position for ``ticker``."""
        key = self._oldest_key(ticker)
        if key is None:
            raise KeyError(f"No open position for {ticker}")
        position = self._open.pop(key)
        proceeds = position.shares * exit_price
        commission = self._charge_fees("sell", proceeds, position.shares)
        exit_value = proceeds - commission
        self._cash += exit_value
        entry_cost = position.slot_capital
        # Total PnL = capital return + dividend income. ``dividend_income`` is
        # 0 in full mode (dividends are baked into adjusted prices there), so
        # full-mode pnl is unchanged; in splits_only/none it adds the cash
        # dividends credited while the lot was held. ``return_pct`` keeps its
        # capital-only definition for report continuity.
        pnl = exit_value - entry_cost + position.dividend_income
        return_pct = (exit_value - entry_cost) / entry_cost if entry_cost else 0.0
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
        exit_date: _Stamp,
        exit_price: float,
        reason: ExitReason,
        fraction: float,
    ) -> Trade:
        """Sell ``fraction`` of the ticker's oldest open position.

        The emitted Trade represents only the closed sleeve. Its ``entry_cost``
        is the pro-rata share of the original entry cost, so ``return_pct`` is
        comparable to a full-close trade. The remaining sleeve continues to
        accrue PnL against its reduced entry_cost.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        if fraction >= 1.0:
            return self.close(
                ticker,
                exit_date,
                exit_price,
                reason,
            )
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
        proceeds = close_shares * exit_price
        commission = self._charge_fees("sell", proceeds, close_shares)
        exit_value = proceeds - commission
        self._cash += exit_value
        # Total PnL includes the pro-rata dividend income for the closed sleeve
        # (0 in full mode). ``return_pct`` stays capital-only for continuity.
        pnl = exit_value - pro_rata_cost + pro_rata_div
        return_pct = (
            (exit_value - pro_rata_cost) / pro_rata_cost if pro_rata_cost else 0.0
        )
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

    def closed_trades(self) -> list[Trade]:
        return list(self._closed)

    def cash(self) -> float:
        return self._cash


def build_equity_curve(
    calendar: pd.DatetimeIndex,
    trades: Iterable[Trade],
    price_panel: dict[str, pd.DataFrame],
    initial_capital: float,
    price_adjustment: str = "full",
) -> pd.Series:
    """Reconstruct the equity curve from a list of completed trades.

    On each calendar date, equity = cash + Σ shares * close for positions that
    are open that day (after applying all trade events dated <= that day, with
    entries processed before exits on the same day).

    In non-``full`` price-adjustment regimes the per-share ``dividend`` column
    of each frame is credited as cash on/after the ex-date for the shares held
    that day, mirroring ``Portfolio.credit_dividends``. ``full`` mode bakes
    dividends into adjusted prices, so the dividend stream is skipped there and
    the curve is unchanged.
    """
    credit_dividends = price_adjustment != "full"
    trades = list(trades)
    # Event list keyed by a monotonically-increasing trade sequence so two
    # trades on the same ticker (re-entry or pyramiding) are tracked
    # independently. Sort closes before opens on the same day so a
    # same-day close+reopen frees the slot before refilling.
    events: list[tuple[pd.Timestamp, int, int, Trade]] = []
    for seq, t in enumerate(trades):
        events.append((pd.Timestamp(t.entry_date), 1, seq, t))  # 1 = open
        events.append((pd.Timestamp(t.exit_date), 0, seq, t))  # 0 = close (first)
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    # Per-trade ex-date dividend cash, keyed by the calendar day on which it is
    # credited. The engine credits dividends on every ex-date bar strictly
    # after entry up to and including the exit bar (see core._maybe_credit_
    # dividends + _close_slot_at_day ordering), so mirror that window here so
    # the curve carries exactly the dividend stream behind each trade's
    # ``dividend_income``. Skipped entirely in full mode.
    dividend_cash_by_day: dict[pd.Timestamp, float] = {}
    if credit_dividends:
        for t in trades:
            frame = price_panel.get(t.ticker)
            if frame is None or frame.empty or "dividend" not in frame.columns:
                continue
            entry_ts = pd.Timestamp(t.entry_date)
            exit_ts = pd.Timestamp(t.exit_date)
            window = frame.loc[
                (frame.index > entry_ts) & (frame.index <= exit_ts), "dividend"
            ]
            for ex_day, div in window.items():
                div = float(cast(Any, div))
                if pd.isna(div) or div <= 0:
                    continue
                ex_ts = pd.Timestamp(cast(Any, ex_day))
                dividend_cash_by_day[ex_ts] = (
                    dividend_cash_by_day.get(ex_ts, 0.0) + t.shares * div
                )

    # Vectorised mark-to-market: pre-align each traded ticker's valid closes
    # to the calendar once, forward-filled — a calendar day with no valid
    # close for the ticker (holiday mismatch, trading halt, delisting tail)
    # carries the most recent prior close, and days before the first valid
    # close fall back to the trade's entry price, exactly like the per-day
    # scalar lookups this replaces. Each trade contributes ``shares * close``
    # over its open calendar span [entry_date, exit_date). Trades accumulate
    # in the order the open events were applied — (entry date, sequence) — so
    # per-day float summation order matches the original open-positions loop
    # bit for bit.
    aligned_close: dict[str, np.ndarray] = {}
    mtm = np.zeros(len(calendar), dtype=float)
    for _seq, trade in sorted(
        enumerate(trades), key=lambda item: (pd.Timestamp(item[1].entry_date), item[0])
    ):
        arr = aligned_close.get(trade.ticker)
        if arr is None:
            frame = price_panel.get(trade.ticker)
            valid = (
                frame["close"].dropna()
                if frame is not None and not frame.empty
                else None
            )
            if valid is None or valid.empty:
                arr = np.full(len(calendar), np.nan)
            else:
                # Equivalent to ``valid.reindex(calendar, method="ffill")``
                # without constructing a pandas Series and indexer result for
                # every traded ticker.
                valid_index = valid.index
                positions = valid_index.searchsorted(calendar, side="right") - 1
                arr = np.full(len(calendar), np.nan)
                has_value = positions >= 0
                if has_value.any():
                    values = valid.to_numpy(dtype=float)
                    arr[has_value] = values[positions[has_value]]
            aligned_close[trade.ticker] = arr
        entry_ts = pd.Timestamp(trade.entry_date)
        exit_ts = pd.Timestamp(trade.exit_date)
        lo = int(calendar.searchsorted(entry_ts, side="left"))
        if exit_ts <= entry_ts:
            # Same-day entry and exit (e.g. a force-close of a position opened
            # on the window's last bar): the event loop orders closes before
            # opens on the same day, so the close never pops this position and
            # it stays marked-to-market for every remaining calendar day.
            # Preserved exactly for bit-identical curves.
            hi = len(calendar)
        else:
            hi = int(calendar.searchsorted(exit_ts, side="left"))
        if lo >= hi:
            continue
        prices = arr[lo:hi]
        if np.isnan(prices).any():
            prices = np.where(np.isnan(prices), trade.entry_price, prices)
        mtm[lo:hi] += trade.shares * prices

    cash = float(initial_capital)
    values = np.empty(len(calendar), dtype=float)
    ev_idx = 0
    dividend_events = sorted(dividend_cash_by_day.items())
    dividend_idx = 0

    for day_idx, day in enumerate(calendar):
        while ev_idx < len(events) and events[ev_idx][0] <= day:
            _, kind, _seq2, trade = events[ev_idx]
            if kind == 1:  # open
                cash -= trade.entry_cost
            else:  # close
                cash += trade.exit_value
            ev_idx += 1

        # Credit every ex-date dividend whose date has been reached. A sorted
        # cursor avoids scanning all remaining dividend dates on every calendar
        # row while preserving the "credit on the next curve point" behaviour.
        while (
            dividend_idx < len(dividend_events)
            and dividend_events[dividend_idx][0] <= day
        ):
            cash += dividend_events[dividend_idx][1]
            dividend_idx += 1

        values[day_idx] = cash + mtm[day_idx]
    return pd.Series(values, index=calendar, dtype=float)
