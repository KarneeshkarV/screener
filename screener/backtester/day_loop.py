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
vs. daily candidate refill) and is therefore expressed as the explicit
:class:`CandidateSource` interface, with the historical and rolling behaviours
as its two adapters — see the module docstrings of ``historical`` and
``rolling``. Modelling the difference at the candidate seam (rather than
branching on a ``mode`` flag inside the day-loop) keeps each path's exact
ordering and semantics intact.

The per-day skeleton shared by both engines is :func:`run_day_loop`, which
drives one ``CandidateSource`` and one ``DayLoop`` over a calendar:

    for each day:
        source.before_exits(day)              # engine-specific pre-exit fill
        freed = day_loop.process_exits_for_day(day)
        source.after_exits(day, freed)         # engine-specific refill

The exit sequence per slot, per day, is invariant:

    dividends → partial exits → (full-close-by-partial check) → exit check

This mirrors the original inline historical loop and ``_close_slot_at_day``
exactly; :class:`DayLoop` is the single home for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from screener.backtester.core import (
    _bar_label,
    _check_exit_at_bar,
    _fire_partial_exits_at_bar,
    _maybe_credit_dividends,
    _SlotState,
)
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


def _close_slot_at_day(
    *,
    slot_id: int,
    state: _SlotState,
    bars: pd.DataFrame,
    day: pd.Timestamp,
    cfg: BacktestConfig,
    portfolio: Portfolio,
    slot_states: dict[int, _SlotState | None],
    fill_model: FillModel,
) -> bool:
    """Process one slot for a day. Returns True when the slot becomes free."""
    frame_cache = state.frame_cache
    if frame_cache is not None and frame_cache.index_i8 is not None:
        index_i8 = frame_cache.index_i8
        pos = int(np.searchsorted(index_i8, day.value))
        if pos >= index_i8.size or index_i8[pos] != day.value:
            return False
        i: int = pos
    else:
        if day not in bars.index:
            return False
        loc = bars.index.get_loc(day)
        if isinstance(loc, slice) or not isinstance(loc, int):
            return False
        i = loc
    if i < state.entry_idx + 1:
        return False
    _maybe_credit_dividends(portfolio, state, bars, i, cfg)
    _fire_partial_exits_at_bar(state, bars, i, cfg, portfolio, fill_model)
    position = portfolio.get_position(state.ticker)
    if position is None:
        slot_states[slot_id] = None
        return True
    exit_ = _check_exit_at_bar(
        state,
        bars,
        i,
        cfg,
        fill_model,
        shares=position.shares,
    )
    if exit_ is None:
        return False
    fill, reason = exit_
    portfolio.close(
        ticker=state.ticker,
        exit_date=_bar_label(day, cfg),
        exit_price=fill,
        reason=reason,
    )
    slot_states[slot_id] = None
    return True


def _force_close_open_slots(
    *,
    slot_states: dict[int, _SlotState | None],
    slot_bars: dict[int, pd.DataFrame],
    cfg: BacktestConfig,
    portfolio: Portfolio,
    end_ts: pd.Timestamp,
    fill_model: FillModel,
) -> None:
    for slot_id, state in list(slot_states.items()):
        if state is None:
            continue
        bars = slot_bars[slot_id]
        tail = bars.loc[
            (bars.index >= pd.Timestamp(state.entry_date)) & (bars.index <= end_ts)
        ]
        if tail.empty:
            continue
        last_bar = tail.iloc[-1]
        fill = fill_model.exit_price(
            reason="eod",
            close=float(last_bar["close"]),
            shares=(
                position.shares
                if (position := portfolio.get_position(state.ticker)) is not None
                else 0.0
            ),
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
            half_spread=state.half_spread,
        )
        portfolio.close(
            ticker=state.ticker,
            exit_date=_bar_label(tail.index[-1], cfg),
            exit_price=fill,
            reason="eod",
        )
        slot_states[slot_id] = None


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


class CandidateSource(Protocol):
    """The engine-specific *fill* half of a backtest day.

    :class:`DayLoop` owns the invariant *exit* half. Refilling slots with new
    entries is where the two engines genuinely diverge:

    * historical rotates a fixed reserve queue selected once at ``as_of`` and
      may re-enter the same ticker into the slot it just vacated;
    * rolling refills from that day's freshly ranked candidate scan.

    Each engine supplies an adapter implementing this Protocol; :func:`run_day_loop`
    interleaves it with the shared exit sequence. ``before_exits`` runs any
    pre-exit fill work (historical re-entry promotion); ``after_exits`` receives
    the slots that freed during the exit sweep and refills them. Both mutate the
    shared slot/portfolio state in place — the return value is intentionally
    ``None`` so neither engine's exact ordering leaks into the driver.
    """

    def before_exits(self, day: pd.Timestamp) -> None:
        """Fill work that must happen *before* the day's exit sweep."""
        ...

    def after_exits(self, day: pd.Timestamp, freed: list[FreedSlot]) -> None:
        """Refill slots given those that freed during the exit sweep."""
        ...


def run_day_loop(
    days: Iterable[pd.Timestamp],
    day_loop: DayLoop,
    source: CandidateSource,
) -> None:
    """Drive one ``DayLoop`` and one ``CandidateSource`` over ``days``.

    The single shared per-day skeleton: pre-exit fill, the invariant exit sweep,
    then post-exit refill. Both engines' full per-day semantics live in their
    ``CandidateSource`` adapter; this driver only fixes the ordering between the
    exit half and the fill half.
    """
    for day in days:
        source.before_exits(day)
        freed = day_loop.process_exits_for_day(day)
        source.after_exits(day, freed)
