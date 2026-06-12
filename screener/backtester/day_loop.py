"""Unified per-day exit orchestration shared by both backtest flows.

Two backtest engines run a near-identical per-day skeleton:

* ``historical`` (:mod:`screener.backtester.historical`) — an event-driven sim
  that selects candidates once (at ``as_of``) into an active set plus a reserve
  queue, then walks forward crediting dividends, firing partial exits, checking
  stop/target/trail/time/exit_expr, and rotating reserves (or re-entering the
  same ticker) into freed slots.
* ``rolling`` (:mod:`screener.backtester.rolling`) — the same per-day skeleton,
  but candidates are precomputed as matrices and freed slots are refilled from
  that day's ranking.

The *exit* half of each day is identical between the two and is owned here by
:class:`DayLoop`. The *fill* half genuinely differs (reserve queue + re-entry
vs. daily candidate refill) and is therefore left to each engine as a
candidate-source adapter — see the module docstrings of ``historical`` and
``rolling``. Modelling the difference at the candidate seam (rather than
branching on a ``mode`` flag inside the day-loop) keeps each path's exact
ordering and semantics intact.

The exit sequence per slot, per day, is invariant:

    dividends → partial exits → (full-close-by-partial check) → exit check

This mirrors the original inline historical loop and ``_close_slot_at_day``
exactly; :class:`DayLoop` is the single home for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from screener.backtester.core import _SlotState, _close_slot_at_day
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio


@dataclass(frozen=True)
class FreedSlot:
    """A slot that became free during a day's exit processing.

    ``state`` is the slot state *as it was when it closed* — engines use it to
    decide on re-entry (historical) or simply to know the slot is available
    (rolling).
    """

    slot_id: int
    state: _SlotState


class DayLoop:
    """Owns the invariant per-day exit sequence for one portfolio.

    The loop holds references to the shared mutable structures (``portfolio``,
    ``slot_states``, ``slot_bars``) and the immutable ``cfg``. Engines drive it
    one day at a time via :meth:`process_exits_for_day`, then run their own
    candidate-fill logic against the returned freed slots.
    """

    def __init__(
        self,
        *,
        portfolio: Portfolio,
        cfg: BacktestConfig,
        slot_states: dict[int, _SlotState | None],
        slot_bars: dict[int, pd.DataFrame],
        fill_model: FillModel | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.cfg = cfg
        self.slot_states = slot_states
        self.slot_bars = slot_bars
        self.fill_model = fill_model if fill_model is not None else FillModel(cfg)

    def process_exits_for_day(self, day: pd.Timestamp) -> list[FreedSlot]:
        """Run dividends → partial exits → exit checks for every live slot.

        Returns the slots that freed on ``day`` (with the state they held at
        close), in slot-id iteration order. Slots whose bars do not include
        ``day``, or that have not yet reached their entry bar, are skipped — the
        original loops short-circuit identically.
        """
        freed: list[FreedSlot] = []
        for slot_id, state in list(self.slot_states.items()):
            if state is None:
                continue
            bars = self.slot_bars[slot_id]
            if _close_slot_at_day(
                slot_id=slot_id,
                state=state,
                bars=bars,
                day=day,
                cfg=self.cfg,
                portfolio=self.portfolio,
                slot_states=self.slot_states,
                fill_model=self.fill_model,
            ):
                freed.append(FreedSlot(slot_id=slot_id, state=state))
        return freed
