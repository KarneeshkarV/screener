"""Book knowledge: the mutable half of a run - capital, slots and fills.

Everything here is rebuilt from scratch for every simulation, which is exactly
why a :class:`~screener.backtester.rolling_simulation.PreparedRollingBacktest`
can be reused across a parameter sweep: changing a book field changes no bar
and no signal.

:data:`BOOK_CONFIG_FIELDS` is that claim written down. The reuse fingerprint
subtracts it, and treats every config field that no module claims as
panel-affecting, so forgetting to classify a new field costs a rebuild rather
than silently reusing stale panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from screener.backtester.core import (
    PendingEntryOrder,
    _make_slot_state,
    _RunCaches,
    _SlotState,
)
from screener.backtester.costs import cost_model_from_config
from screener.backtester.day_loop import DayLoop
from screener.backtester.fills import FillModel, _resolve_entry_fill
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.sizing import (
    entry_budget_for,
    entry_opens_no_shares,
    marked_portfolio_equity,
    sizing_allows_slot_growth,
)


def fill_pending_entries(
    *,
    day: pd.Timestamp,
    at_close: bool,
    pending_orders: dict[int, PendingEntryOrder],
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    portfolio: Portfolio,
    slot_states: dict[int, _SlotState | None],
    slot_bars: dict[int, pd.DataFrame],
    fill_model: FillModel,
    caches: _RunCaches,
    exit_ast,
    warnings: list[str],
) -> list[PendingEntryOrder]:
    """Fill only current-bar orders, sharing cash among fillable orders.

    Market-on-close orders run after exits; open and limit orders run before.
    Untouched limits reserve slots, but do not reduce this batch's cash share.
    """
    if at_close != (cfg.entry_order_type == "moc"):
        return []
    fillable: list[tuple[int, PendingEntryOrder, int]] = []
    for slot_id, order in list(pending_orders.items()):
        bars = bars_by_ticker[order.ticker]
        if day > bars.index[-1]:
            del pending_orders[slot_id]
            continue
        entry_idx = int(bars.index.get_indexer(pd.DatetimeIndex([day]))[0])
        if day <= pd.Timestamp(order.signal_date) or entry_idx <= order.signal_idx:
            continue
        index, reference, _warn = _resolve_entry_fill(
            bars, order.signal_idx, cfg, caches.frame(order.ticker, bars), entry_idx
        )
        if index is not None and reference is not None and 0 < reference < float("inf"):
            fillable.append((slot_id, order, entry_idx))
        elif cfg.entry_order_type != "limit":
            del pending_orders[slot_id]
    grows_slots = sizing_allows_slot_growth(cfg.sizing_rule)
    current_equity = (
        marked_portfolio_equity(portfolio, bars_by_ticker, day, inclusive=at_close)
        if grows_slots
        else None
    )
    filled: list[PendingEntryOrder] = []
    for offset, (slot_id, order, entry_idx) in enumerate(fillable):
        bars = bars_by_ticker[order.ticker]
        budget = entry_budget_for(
            cfg,
            portfolio,
            bars,
            order.signal_idx,
            current_equity=current_equity,
            free_slots=len(fillable) - offset,
            series_cache=caches.frame(order.ticker, bars).sizing_series,
        )
        state, warn = _make_slot_state(
            order.ticker,
            bars,
            order.signal_idx,
            cfg,
            exit_ast,
            order.rank,
            fill_model,
            caches=caches,
            entry_budget=budget,
            limit_bar_idx=entry_idx,
        )
        if state is None or entry_opens_no_shares(budget, state.entry_shares):
            if warn:
                warnings.append(f"{order.ticker}: {warn}")
            if cfg.entry_order_type != "limit":
                del pending_orders[slot_id]
            continue
        portfolio.assign(order.ticker, order.rank, order.signal_date)
        position = portfolio.open(
            order.ticker,
            state.entry_date,
            state.entry_fill,
            budget=budget,
            shares=state.entry_shares,
            allow_slot_growth=grows_slots,
        )
        if position.shares <= 0:
            raise RuntimeError("Pending entry fill created an empty position")
        slot_states[slot_id] = state
        slot_bars[slot_id] = bars
        del pending_orders[slot_id]
        filled.append(order)
    return filled


BOOK_CONFIG_FIELDS = frozenset(
    {
        # Where the bar timestamps and session boundaries come from. Shared
        # with the price panel, which is what pins their values in the
        # fingerprint.
        "market",
        "interval",
        "price_adjustment",
        # Capital and slots.
        "top",
        "initial_capital",
        # Metrics-only input; changing it does not rebuild price/signal panels.
        "risk_free_rate",
        "reserve_multiple",
        "reinvest",
        "allow_reentry",
        "max_reentries",
        "max_concurrent_per_ticker",
        # Holding rules.
        "hold",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "partial_exits",
        "intraday_only",
        # Rank-based exit rebalance.
        "rank_exit_every",
        "rank_universe_size",
        # Fills and costs.
        "slippage_bps",
        "commission_bps",
        "slippage_model",
        "cost_model",
        "spread_proxy",
        "gap_fills",
        "entry_order_type",
        "entry_limit_bps",
        # Per-entry sizing.
        "sizing_rule",
        "compounding",
        "sizing_risk_pct",
        "sizing_position_pct",
        "sizing_atr_window",
        "sizing_atr_multiple",
        "sizing_vol_window",
    }
)


@dataclass(frozen=True)
class Book:
    """Fresh portfolio, slot state and fill machinery for one simulation."""

    portfolio: Portfolio
    slot_states: dict[int, _SlotState | None]
    slot_bars: dict[int, pd.DataFrame]
    selection_rows: list[dict]
    fill_model: FillModel
    day_loop: DayLoop


def open_book(cfg: BacktestConfig) -> Book:
    """Build the per-simulation portfolio, slots, fill model and day loop."""
    portfolio = Portfolio(
        cfg.initial_capital,
        max(cfg.top, 1),
        cost_model=cost_model_from_config(cfg),
        compounding=cfg.compounding,
        whole_shares=cfg.market.lower() == "india",
    )
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    fill_model = FillModel(cfg, cost_model=portfolio.cost_model)
    return Book(
        portfolio=portfolio,
        slot_states=slot_states,
        slot_bars=slot_bars,
        selection_rows=[],
        fill_model=fill_model,
        day_loop=DayLoop(
            portfolio=portfolio,
            cfg=cfg,
            slot_states=slot_states,
            slot_bars=slot_bars,
            fill_model=fill_model,
        ),
    )
