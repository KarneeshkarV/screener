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

from screener.backtester.core import _SlotState
from screener.backtester.costs import cost_model_from_config
from screener.backtester.day_loop import DayLoop
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio

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
        "sizing_risk_pct",
        "sizing_position_pct",
        "sizing_atr_window",
        "sizing_atr_multiple",
        "sizing_vol_window",
        "sizing_ema_fast",
        "sizing_ema_slow",
        "sizing_ema_spread_cap",
        "sizing_ema_spread_floor",
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
