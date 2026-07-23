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

For intraday runs the "days" are bar timestamps and the driver becomes a
*session loop*: the calendar is grouped into exchange sessions keyed on
:func:`screener.backtester.sessions.is_session_last` (positional detection, so
half-days end at their actual last bar and market hours are honoured by
construction), and each session's last bar additionally runs
:meth:`DayLoop.flatten_at_session_end` before the refill so ``intraday_only``
portfolios are guaranteed flat at close:

    for each session:
        for each bar in session:
            before_exits → process_exits_for_day
            if session-last bar: flatten_at_session_end
            after_exits

The exit sequence per slot, per day, is invariant:

    dividends → partial exits → (full-close-by-partial check) → exit check

This mirrors the original inline historical loop and ``_close_slot_at_day``
exactly; :class:`DayLoop` is the single home for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import pandas as pd

from screener.backtester.core import _SlotState, _bar_label, _close_slot_at_day
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.sessions import is_session_last, market_timezone


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

    def flatten_at_session_end(self, session_last_bar: pd.Timestamp) -> list[FreedSlot]:
        """Force-flatten stragglers at a session's last bar (``intraday_only``).

        Slot-level session exits fire inside ``_check_exit_at_bar`` whenever a
        slot's own bars include a session-last bar; this driver-level pass is
        the guarantee on top: any slot still open after the session's last
        *calendar* bar is closed at its own last bar of that session with
        reason ``"session"`` (a market-on-close flat), so an intraday-only run
        never carries a position overnight. No-op unless ``cfg.intraday_only``.
        """
        if not self.cfg.intraday_only:
            return []
        session_last_bar = pd.Timestamp(session_last_bar)
        tz = market_timezone(self.cfg.market)
        target_session = session_last_bar.tz_localize("UTC").tz_convert(tz).date()
        freed: list[FreedSlot] = []
        for slot_id, state in list(self.slot_states.items()):
            if state is None:
                continue
            bars = self.slot_bars[slot_id]
            # bars.index is a sorted DatetimeIndex; searchsorted is O(log n).
            pos = int(bars.index.searchsorted(session_last_bar, side="right")) - 1
            if pos < state.entry_idx + 1:
                continue
            bar_ts = bars.index[pos]
            if bar_ts.tz_localize("UTC").tz_convert(tz).date() != target_session:
                continue  # no bar in this session; the end-of-run pass handles it
            position = self.portfolio.get_position(state.ticker)
            if position is None:
                self.slot_states[slot_id] = None
                freed.append(FreedSlot(slot_id=slot_id, state=state))
                continue
            fill = self.fill_model.exit_price(
                reason="session",
                close=float(bars.iloc[pos]["close"]),
                shares=position.shares,
                adv_shares=state.adv_shares,
                sigma_daily=state.sigma_daily,
                half_spread=state.half_spread,
            )
            self.portfolio.close(
                ticker=state.ticker,
                exit_date=_bar_label(bar_ts, self.cfg),
                exit_price=fill,
                reason="session",
            )
            self.slot_states[slot_id] = None
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
    *,
    market_tz: str | None = None,
) -> None:
    """Drive one ``DayLoop`` and one ``CandidateSource`` over ``days``.

    With ``market_tz=None`` (daily runs) this is the original per-day skeleton:
    pre-exit fill, the invariant exit sweep, then post-exit refill. Both
    engines' full per-day semantics live in their ``CandidateSource`` adapter;
    this driver only fixes the ordering between the exit half and the fill half.

    When ``market_tz`` is given (intraday runs), the driver becomes a session
    loop: bars are grouped into exchange sessions keyed on
    :func:`~screener.backtester.sessions.is_session_last` — positional
    detection, so half-days end at their actual last bar and market hours are
    honoured by construction. Bars are processed in the same chronological
    order as the flat loop (fills are unchanged); the only addition is
    :meth:`DayLoop.flatten_at_session_end` on each session's last bar, before
    the refill, so ``intraday_only`` portfolios are guaranteed flat at close.
    """
    day_list = list(days)
    if market_tz is None:
        for day in day_list:
            source.before_exits(day)
            freed = day_loop.process_exits_for_day(day)
            source.after_exits(day, freed)
        return
    if not day_list:
        return
    index = pd.DatetimeIndex(day_list)
    session_last_mask = is_session_last(index, market_tz)
    for pos, day in enumerate(index):
        source.before_exits(day)
        freed = day_loop.process_exits_for_day(day)
        if session_last_mask[pos]:
            freed.extend(day_loop.flatten_at_session_end(day))
        source.after_exits(day, freed)
