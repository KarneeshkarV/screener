"""Signal-panel knowledge: which ticker is eligible on which bar?

This module owns expression parsing, entry/exit/filter signals, the regime
gate, the earnings blackout, sector neutralisation and the candidate matrices
they feed. Like :mod:`screener.backtester.price_panel` it never sees a
:class:`~screener.backtester.models.BacktestConfig`: it is handed a
:class:`SignalPanelInputs` listing exactly the config values eligibility
depends on, and :data:`SIGNAL_PANEL_CONFIG_FIELDS` is derived from that class
so the reuse fingerprint cannot omit one of them.

Two entry points read the same candidate matrices: :func:`build_signal_panel`
for a whole window (the rolling engine) and :func:`build_day_candidates` for
one as-of date (the screen). The one-day path adds no gate of its own; it
composes the window builder and then reads a single row out of it, so the two
callers cannot disagree about who is a candidate.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Literal

import pandas as pd

from screener.backtester.core import (
    _precompute_entry_signals,
    _precompute_filter_signals,
    _RunCaches,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import Node, evaluate_panel_many, parse, required_lookback
from screener.backtester.price_panel import PricePanel
from screener.backtester.rolling_candidates import (
    _build_rolling_candidate_matrices,
    _candidate_rows_for_day,
    _RollingCandidateMatrices,
)
from screener.regime import classify_regimes
from screener.strategies.spec import StrategyProfile


@dataclass(frozen=True)
class SignalProgram:
    """The parsed entry/exit expressions and the history they require."""

    entry_ast: Node
    exit_ast: Node | None
    lookback: int


@dataclass(frozen=True)
class SignalPanelInputs:
    """Every config value candidate eligibility depends on.

    Field names match :class:`~screener.backtester.models.BacktestConfig`
    field names so :data:`SIGNAL_PANEL_CONFIG_FIELDS` can be derived from this
    class instead of restated by hand.
    """

    market: str
    entry_expr: str
    exit_expr: str | None
    regime_filter: tuple[str, ...]
    earnings_blackout_days: int | None
    sector_neutral: bool
    min_price: float | None
    min_avg_dollar_volume: float | None
    avg_dollar_volume_window: int
    membership_added: tuple[tuple[str, date], ...]
    membership_windows: tuple[tuple[str, date, date | None], ...]
    dynamic_universe_size: int | None
    dynamic_universe_lookback: int
    dynamic_universe_rebalance: str

    @classmethod
    def from_config(cls, cfg: BacktestConfig) -> SignalPanelInputs:
        return cls(
            market=cfg.market,
            entry_expr=cfg.entry_expr,
            exit_expr=cfg.exit_expr,
            regime_filter=cfg.regime_filter,
            earnings_blackout_days=cfg.earnings_blackout_days,
            sector_neutral=cfg.sector_neutral,
            min_price=cfg.min_price,
            min_avg_dollar_volume=cfg.min_avg_dollar_volume,
            avg_dollar_volume_window=cfg.avg_dollar_volume_window,
            membership_added=cfg.membership_added,
            membership_windows=cfg.membership_windows,
            dynamic_universe_size=cfg.dynamic_universe_size,
            dynamic_universe_lookback=cfg.dynamic_universe_lookback,
            dynamic_universe_rebalance=cfg.dynamic_universe_rebalance,
        )


# ``strategy_name`` can install its own entry expression and lookback floor, so
# the signals a run produces depend on it even though it is consumed on the
# price side.
_INDIRECT_SIGNAL_PANEL_FIELDS = frozenset({"strategy_name"})

SIGNAL_PANEL_CONFIG_FIELDS = (
    frozenset(f.name for f in fields(SignalPanelInputs)) | _INDIRECT_SIGNAL_PANEL_FIELDS
)

# Every declared field name of :class:`SignalPanelInputs`, derived from the
# dataclass like :data:`SIGNAL_PANEL_CONFIG_FIELDS` but without the indirect
# price-side extras.
SIGNAL_PANEL_INPUT_FIELDS = frozenset(f.name for f in fields(SignalPanelInputs))

# Run-scoped inputs that a per-strategy profile deliberately does not carry:
# they say which names and venue a run covers, not how a candidate is judged.
# Universe selection stays a run-level decision (``--universe``, D9 in
# docs/plans/unify-screen-backtest.md); ``market`` picks the earnings and
# sector sources for the run. Every other field must be mirrored field for
# field by :class:`~screener.strategies.spec.StrategyProfile`; the partition
# below is enforced at import time so a new gate cannot be added to
# :class:`SignalPanelInputs` without a profile decision.
RUN_SCOPED_SIGNAL_PANEL_FIELDS = frozenset(
    {
        "market",
        "membership_added",
        "membership_windows",
        "dynamic_universe_size",
        "dynamic_universe_lookback",
        "dynamic_universe_rebalance",
    }
)

# The partition itself. Asserted here rather than in spec.py because this
# module owns the field list and already carries the heavy import graph;
# ``screener.strategies`` must not import the backtest engine at module scope
# (see tests/test_import_boundaries.py).
_PROFILE_FIELD_NAMES = frozenset(StrategyProfile.model_fields)
_UNCLASSIFIED_PANEL_FIELDS = (
    SIGNAL_PANEL_INPUT_FIELDS - _PROFILE_FIELD_NAMES - RUN_SCOPED_SIGNAL_PANEL_FIELDS
)
_UNKNOWN_PROFILE_FIELDS = _PROFILE_FIELD_NAMES - SIGNAL_PANEL_INPUT_FIELDS
if _UNCLASSIFIED_PANEL_FIELDS or _UNKNOWN_PROFILE_FIELDS:
    raise RuntimeError(
        "StrategyProfile drifted from SignalPanelInputs: "
        f"unclassified panel gates {sorted(_UNCLASSIFIED_PANEL_FIELDS)}, "
        f"profile fields unknown to the panel {sorted(_UNKNOWN_PROFILE_FIELDS)}. "
        "Mirror each new SignalPanelInputs gate on StrategyProfile, or move it "
        "into RUN_SCOPED_SIGNAL_PANEL_FIELDS with a reason."
    )


@dataclass(frozen=True)
class SignalPanel:
    """Precomputed exit signals and the per-day candidate matrices.

    ``candidate_matrices`` is ``None`` when the window contained no trading
    bars at all; there is nothing to rank in that case and the caller returns
    a flat, no-trade result.
    """

    exit_signals: dict[str, pd.Series | str]
    candidate_matrices: _RollingCandidateMatrices | None


def parse_signal_program(inputs: SignalPanelInputs) -> SignalProgram:
    """Parse the entry/exit expressions and the warmup history they need."""
    entry_ast = parse(inputs.entry_expr)
    exit_ast = parse(inputs.exit_expr) if inputs.exit_expr else None
    lookback = required_lookback(entry_ast)
    if exit_ast is not None:
        lookback = max(lookback, required_lookback(exit_ast))
    return SignalProgram(entry_ast=entry_ast, exit_ast=exit_ast, lookback=lookback)


def _resolve_earnings_blackout(
    inputs: SignalPanelInputs,
    panel: PricePanel,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    injected: dict[str, list[date]] | None,
) -> dict[str, list[date]] | None:
    """Resolve the earnings blackout map when the gate is configured.

    Prefers an injected mapping (tests); otherwise collects via the
    market-aware earnings date collectors (already disk-cached). Keys must be
    TV symbols to match ``bars_by_tv`` / ``signal_mat`` columns.
    """
    if inputs.earnings_blackout_days is None or injected is not None:
        return injected

    from screener.earnings_backtest.earnings_dates import load_earnings_dates_map

    yf_tickers = list(dict.fromkeys(panel.yf_by_tv.values()))
    span_years = max(
        3,
        int((end_ts.normalize() - start_ts.normalize()).days / 365) + 2,
    )
    yf_map = load_earnings_dates_map(yf_tickers, inputs.market, years=span_years)
    # Invert yf -> list[tv] so multi-mapped symbols still gate correctly.
    resolved: dict[str, list[date]] = {}
    for tv, yf_sym in panel.yf_by_tv.items():
        dates = yf_map.get(yf_sym)
        if dates:
            resolved[tv] = dates
    return resolved


def build_signal_panel(
    inputs: SignalPanelInputs,
    panel: PricePanel,
    *,
    program: SignalProgram,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warnings: list[str],
    earnings_blackout: dict[str, list[date]] | None = None,
) -> SignalPanel:
    """Evaluate signals over ``panel`` and rank them into candidate matrices."""
    bars_by_tv = panel.bars_by_tv
    exit_signals_by_tv: dict[str, pd.Series | str] = {}
    # Bool arrays: entry (and exit) results feed matrices / ndarray caches, so
    # skip per-ticker Series boxing that otherwise dominates the pine bucket.
    if program.exit_ast is None:
        evaluated_entry = evaluate_panel_many(
            (program.entry_ast,), bars_by_tv, as_bool_array=True
        )[0]
        entry_signals_by_tv = _precompute_entry_signals(
            bars_by_tv,
            program.entry_ast,
            warnings,
            evaluated=evaluated_entry,
        )
    else:
        evaluated_entry, evaluated_exit = evaluate_panel_many(
            (program.entry_ast, program.exit_ast),
            bars_by_tv,
            as_bool_array=True,
        )
        entry_signals_by_tv = _precompute_entry_signals(
            bars_by_tv,
            program.entry_ast,
            warnings,
            evaluated=evaluated_entry,
        )
        signal_cache = _RunCaches()
        signal_cache.prewarm_exit_signals(
            bars_by_tv,
            program.exit_ast,
            evaluated=evaluated_exit,
        )
        exit_signals_by_tv = signal_cache.exit_signals
    filter_signals_by_tv = _precompute_filter_signals(bars_by_tv, inputs)

    regime_allowed: pd.Series | None = None
    if inputs.regime_filter:
        regime_allowed = classify_regimes(panel.benchmark).isin(
            set(inputs.regime_filter)
        )

    if not panel.master_dates:
        return SignalPanel(exit_signals={}, candidate_matrices=None)

    sector_by_tv: dict[str, str] | None = None
    if inputs.sector_neutral:
        from screener.sectors import sector_by_ticker

        sector_by_tv = sector_by_ticker(panel.tv_symbols, inputs.market)

    resolved_earnings_blackout = _resolve_earnings_blackout(
        inputs,
        panel,
        start_ts=start_ts,
        end_ts=end_ts,
        injected=earnings_blackout,
    )

    candidate_matrices = _build_rolling_candidate_matrices(
        bars_by_tv,
        entry_signals_by_tv,
        filter_signals_by_tv,
        panel.master_dates,
        panel.lookback,
        membership_added=dict(inputs.membership_added) or None,
        membership_windows=inputs.membership_windows,
        dynamic_universe_size=inputs.dynamic_universe_size,
        dynamic_universe_lookback=inputs.dynamic_universe_lookback,
        dynamic_universe_rebalance=inputs.dynamic_universe_rebalance,
        regime_allowed=regime_allowed,
        earnings_blackout=resolved_earnings_blackout,
        earnings_blackout_days=inputs.earnings_blackout_days,
        warnings=warnings,
        sector_neutral=inputs.sector_neutral,
        sector_by_tv=sector_by_tv,
    )
    return SignalPanel(
        exit_signals=exit_signals_by_tv,
        candidate_matrices=candidate_matrices,
    )


@dataclass(frozen=True)
class Candidate:
    """One ticker that cleared every entry gate on one master-calendar bar.

    Produced by :func:`day_candidates_from_panel` from the same candidate
    matrices the rolling engine scans, so a screen and a backtest describe the
    same name the same way. ``rank`` is the 1-based position in that day's
    ranking and ``rank_basis`` names the field the ranking used: ``rank_score``
    when the strategy wrote a cross-sectional factor score into its bars,
    ``as_of_dollar_vol`` otherwise.
    """

    ticker: str
    rank: int
    rank_basis: Literal["rank_score", "as_of_dollar_vol"]
    rank_score: float | None
    as_of_close: float
    as_of_volume: float
    as_of_dollar_vol: float
    signal_idx: int
    role: Literal["active", "reserve"]


@dataclass(frozen=True)
class DayCandidates:
    """The ranked candidates on one as-of date.

    ``as_of`` is the master-calendar bar the request resolved to, which is the
    last bar at or before the requested date. It is ``None`` when the window
    holds no bar at or before that date, mirroring
    :attr:`SignalPanel.candidate_matrices`; ``candidates`` is then empty.
    """

    as_of: pd.Timestamp | None
    candidates: tuple[Candidate, ...]


def _resolve_as_of_bar(
    matrices: _RollingCandidateMatrices, as_of: date | pd.Timestamp
) -> pd.Timestamp | None:
    """Snap ``as_of`` back to the last master-calendar bar at or before it."""
    calendar = matrices.signal_mat.index
    if isinstance(as_of, datetime):
        # datetime covers pd.Timestamp too: both carry a time of day, so both
        # are instants and must not be widened to the end of their session. A
        # caller asking for 10:00 wants the 10:00 bar, not the 15:30 close.
        position = int(calendar.searchsorted(as_of, side="right")) - 1
    else:
        # A plain calendar date covers its whole session, which matters
        # intraday where a master-calendar stamp carries a time of day. Snap on
        # the next midnight rather than "end of day minus one nanosecond": the
        # master calendar is datetime64[us] under pandas 3, and searchsorted
        # refuses a nanosecond bound it cannot convert losslessly.
        next_midnight = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        position = int(calendar.searchsorted(next_midnight, side="left")) - 1
    if position < 0:
        return None
    return pd.Timestamp(calendar[position])


def _as_of_is_within(as_of: date | pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    """True when ``as_of`` falls at or before the window the panel was built on.

    An as-of past the window end is a mis-sized request, not a holiday: snapping
    back would answer with candidates dated days earlier and report it as a
    success. A plain ``date`` is compared at the end of its session, matching
    how :func:`_resolve_as_of_bar` reads one.
    """
    if isinstance(as_of, datetime):
        return pd.Timestamp(as_of) <= end_ts
    return pd.Timestamp(as_of).normalize() <= end_ts.normalize()


def day_candidates_from_panel(
    signals: SignalPanel,
    as_of: date | pd.Timestamp,
    *,
    exclude: Collection[str] = (),
    limit: int | None = None,
    warnings: list[str] | None = None,
) -> DayCandidates:
    """Read one day's ranked candidates out of an already-built signal panel.

    This is the rolling engine's own per-day scan
    (:func:`~screener.backtester.rolling_candidates._candidate_rows_for_day`)
    over the rolling engine's own matrices, reshaped into :class:`Candidate`.
    Every gate - entry signal, entry filters, regime, earnings blackout, sector
    neutralisation - was already applied when ``signals`` was built, so there
    is nothing here for a one-day caller to re-decide.
    """
    matrices = signals.candidate_matrices
    if matrices is None:
        return DayCandidates(as_of=None, candidates=())
    day = _resolve_as_of_bar(matrices, as_of)
    if day is None:
        return DayCandidates(as_of=None, candidates=())

    rows, day_warnings = _candidate_rows_for_day(
        day, matrices, exclude=set(exclude), limit=limit
    )
    if warnings is not None:
        warnings.extend(day_warnings)

    rank_score_np = matrices.rank_score_np
    rank_basis: Literal["rank_score", "as_of_dollar_vol"] = (
        "rank_score" if rank_score_np is not None else "as_of_dollar_vol"
    )
    row_position = matrices.row_by_day[day]
    candidates = tuple(
        Candidate(
            ticker=str(row["ticker"]),
            rank=int(row["rank"]),
            rank_basis=rank_basis,
            rank_score=(
                None
                if rank_score_np is None
                else float(
                    rank_score_np[row_position, matrices.col_by_ticker[row["ticker"]]]
                )
            ),
            as_of_close=float(row["as_of_close"]),
            as_of_volume=float(row["as_of_volume"]),
            as_of_dollar_vol=float(row["as_of_dollar_vol"]),
            signal_idx=int(row["signal_idx"]),
            role=row["role"],
        )
        for row in rows
    )
    return DayCandidates(as_of=day, candidates=candidates)


def build_day_candidates(
    inputs: SignalPanelInputs,
    panel: PricePanel,
    *,
    program: SignalProgram,
    as_of: date | pd.Timestamp,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warnings: list[str],
    earnings_blackout: dict[str, list[date]] | None = None,
    exclude: Collection[str] = (),
    limit: int | None = None,
) -> DayCandidates:
    """Rank the candidates for a single as-of date.

    The one-day entry point, for a caller that wants one day rather than a
    window. It builds the signal panel over ``[start_ts, end_ts]`` exactly as
    :func:`~screener.backtester.rolling_simulation.prepare_rolling_backtest`
    does and then reads the ``as_of`` row out of it, so it cannot drift from
    the rolling engine: there is one implementation of every gate and this
    calls it.

    A caller holding several as-of dates over one window should build the panel
    once with :func:`build_signal_panel` and call
    :func:`day_candidates_from_panel` per date instead.
    """
    if not _as_of_is_within(as_of, end_ts):
        # Mirrors the below-window case, which _resolve_as_of_bar already
        # answers with None rather than with the nearest bar it can find.
        return DayCandidates(as_of=None, candidates=())
    signals = build_signal_panel(
        inputs,
        panel,
        program=program,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        earnings_blackout=earnings_blackout,
    )
    return day_candidates_from_panel(
        signals,
        as_of,
        exclude=exclude,
        limit=limit,
        warnings=warnings,
    )
