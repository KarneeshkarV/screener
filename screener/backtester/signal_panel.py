"""Signal-panel knowledge: which ticker is eligible on which bar?

This module owns expression parsing, entry/exit/filter signals, the regime
gate, the earnings blackout, sector neutralisation and the candidate matrices
they feed. Like :mod:`screener.backtester.price_panel` it never sees a
:class:`~screener.backtester.models.BacktestConfig`: it is handed a
:class:`SignalPanelInputs` listing exactly the config values eligibility
depends on, and :data:`SIGNAL_PANEL_CONFIG_FIELDS` is derived from that class
so the reuse fingerprint cannot omit one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Optional

import pandas as pd

from screener.backtester.core import (
    _RunCaches,
    _precompute_entry_signals,
    _precompute_filter_signals,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import Node, evaluate_panel_many, parse, required_lookback
from screener.backtester.price_panel import PricePanel
from screener.backtester.rolling_candidates import (
    _RollingCandidateMatrices,
    _build_rolling_candidate_matrices,
)
from screener.regime import classify_regimes


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
    exit_expr: Optional[str]
    regime_filter: tuple[str, ...]
    earnings_blackout_days: int | None
    sector_neutral: bool
    min_price: Optional[float]
    min_avg_dollar_volume: Optional[float]
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
    if program.exit_ast is None:
        entry_signals_by_tv = _precompute_entry_signals(
            bars_by_tv, program.entry_ast, warnings
        )
    else:
        evaluated_entry, evaluated_exit = evaluate_panel_many(
            (program.entry_ast, program.exit_ast), bars_by_tv
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
