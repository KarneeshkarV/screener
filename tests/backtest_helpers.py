"""Test helpers that drive the production single-position execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from screener.backtester.core import _make_slot_state
from screener.backtester.costs import cost_model_from_config
from screener.backtester.day_loop import DayLoop, _force_close_open_slots
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.portfolio import Portfolio


@dataclass(frozen=True)
class SingleTickerResult:
    """The production ledger and any entry-construction warning."""

    trades: tuple[Trade, ...]
    warning: str | None
    cash: float

    @property
    def trade(self) -> Trade | None:
        """Return the final closed trade for single-full-exit test cases."""
        return self.trades[-1] if self.trades else None


def simulate_single_ticker(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast: Any = None,
    *,
    ticker: str = "TEST",
    rank: int = 1,
) -> SingleTickerResult:
    """Run one position through the production ``DayLoop`` and force-close path."""
    portfolio = Portfolio(
        cfg.initial_capital,
        slot_count=max(cfg.top, 1),
        cost_model=cost_model_from_config(cfg),
    )
    fill_model = FillModel(cfg, cost_model=portfolio.cost_model)
    entry_budget = portfolio.entry_budget()
    state, warning = _make_slot_state(
        ticker=ticker,
        bars=bars,
        signal_idx=signal_idx,
        cfg=cfg,
        exit_ast=exit_ast,
        rank=rank,
        fill_model=fill_model,
        entry_budget=entry_budget,
    )
    if state is None:
        return SingleTickerResult(trades=(), warning=warning, cash=portfolio.cash())

    portfolio.assign(ticker, state.rank, state.signal_date)
    portfolio.open(
        ticker=ticker,
        entry_date=state.entry_date,
        entry_price=state.entry_fill,
        budget=entry_budget,
    )
    slot_states = {0: state}
    slot_bars = {0: bars}
    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )
    for day in bars.index[state.entry_idx + 1 :]:
        day_loop.process_exits_for_day(day)
    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=bars.index[-1],
        fill_model=fill_model,
    )
    trades = tuple(portfolio.closed_trades())
    return SingleTickerResult(trades=trades, warning=None, cash=portfolio.cash())
