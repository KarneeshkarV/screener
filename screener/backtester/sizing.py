"""Rule-based per-entry position sizing.

Each rule maps entry-time context to a dollar budget for the new position.
The result is always clamped to ``Portfolio.entry_budget()`` — the equal-slot
ceiling and available cash — so a rule can size *down* from the slot budget
but never above it, and every existing cash-accounting invariant holds.

Sizing equity is deliberately the portfolio's ``initial_capital``
(non-compounding), matching the equal-slot philosophy documented in
``screener.backtester.portfolio``: per-position sizing stays balanced across
the run regardless of lucky-early-trade effects.

Rules that need bar history (``atr_risk``, ``inverse_vol``) read only data up
to and including the signal bar. Both indicators are causal, so evaluating
the full series and indexing at ``signal_idx`` matches truncated-history
evaluation — the same discipline the entry-signal evaluators rely on. When
the indicator is not yet defined at the signal bar (insufficient lookback,
zero volatility), the rule falls back to the equal-slot budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from screener.indicators.frames import wilder_atr

if TYPE_CHECKING:
    from screener.backtester.models import BacktestConfig
    from screener.backtester.portfolio import Portfolio


@dataclass(frozen=True)
class SizingContext:
    """Entry-time inputs handed to a sizing rule.

    ``base_budget`` is ``Portfolio.entry_budget()`` — the legacy equal-slot
    budget and the hard cap applied to whatever the rule returns.
    """

    equity: float
    base_budget: float
    stop_loss: float | None
    policy: BacktestConfig
    bars: pd.DataFrame
    signal_idx: int
    # Per-frame memo for the full indicator series a rule reads one scalar out
    # of, keyed by ``(rule kind, window)``. Owned by
    # ``screener.backtester.core._FrameCache`` and threaded in by
    # :func:`entry_budget_for`; ``None`` means "no run caches", and every rule
    # then recomputes as it always did.
    series_cache: dict[tuple[str, int], np.ndarray] | None = None


SizerFunc = Callable[[SizingContext], float]


def _cached_series(
    ctx: SizingContext, kind: str, window: int, build: Callable[[], pd.Series]
) -> np.ndarray:
    """``build()`` as a float ndarray, computed at most once per frame.

    Sizing rules are causal and read only ``signal_idx``, so the series for one
    ticker is the same for every entry the run opens on it. Building it per
    entry made position sizing cost O(entries * bars) instead of O(bars).
    """
    if ctx.series_cache is None:
        return build().to_numpy(dtype=float)
    key = (kind, window)
    cached = ctx.series_cache.get(key)
    if cached is None:
        cached = build().to_numpy(dtype=float)
        ctx.series_cache[key] = cached
    return cached

_SIZERS: dict[str, SizerFunc] = {}


def sizer(name: str) -> Callable[[SizerFunc], SizerFunc]:
    """Register a sizing rule under ``name`` (mirrors the criteria/strategy
    registries). A rule returns an unclamped dollar budget; ``nan`` means
    "cannot size this entry" and falls back to the equal-slot budget."""

    def decorate(func: SizerFunc) -> SizerFunc:
        if name in _SIZERS:
            raise ValueError(f"sizing rule already registered: {name}")
        _SIZERS[name] = func
        return func

    return decorate


def available_sizing_rules() -> tuple[str, ...]:
    """Known rule names, ``equal_slot`` (the default) first."""
    rest = sorted(n for n in _SIZERS if n != "equal_slot")
    return ("equal_slot", *rest)


@sizer("equal_slot")
def _equal_slot(ctx: SizingContext) -> float:
    return ctx.base_budget


@sizer("fixed_fraction")
def _fixed_fraction(ctx: SizingContext) -> float:
    return ctx.equity * ctx.policy.sizing_position_pct


@sizer("fixed_risk")
def _fixed_risk(ctx: SizingContext) -> float:
    # Risk ``sizing_risk_pct`` of equity per trade: with a stop ``stop_loss``
    # below entry, losing the stop costs budget * stop_loss, so
    # budget = equity * risk_pct / stop_loss.
    if not ctx.stop_loss or ctx.stop_loss <= 0:
        raise ValueError("sizing rule 'fixed_risk' requires a positive stop_loss")
    return ctx.equity * ctx.policy.sizing_risk_pct / ctx.stop_loss


@sizer("atr_risk")
def _atr_risk(ctx: SizingContext) -> float:
    # Volatility-normalized risk: treat ``atr_multiple * ATR`` as the expected
    # adverse excursion and risk ``sizing_risk_pct`` of equity against it.
    policy = ctx.policy
    bars = ctx.bars
    window = policy.sizing_atr_window
    atr = _cached_series(
        ctx,
        "atr",
        window,
        lambda: wilder_atr(
            bars["high"],
            bars["low"],
            bars["close"],
            window,
            min_periods=window,
        ),
    )
    atr_value = float(atr[ctx.signal_idx])
    close = float(bars["close"].iloc[ctx.signal_idx])
    if not math.isfinite(atr_value) or atr_value <= 0 or close <= 0:
        return math.nan
    stop_fraction = policy.sizing_atr_multiple * atr_value / close
    return ctx.equity * policy.sizing_risk_pct / stop_fraction


@sizer("inverse_vol")
def _inverse_vol(ctx: SizingContext) -> float:
    # Daily volatility targeting: size so the position's expected daily PnL
    # swing (budget * realized daily vol) is ``sizing_risk_pct`` of equity.
    policy = ctx.policy
    window = policy.sizing_vol_window
    vol = _cached_series(
        ctx,
        "inverse_vol",
        window,
        lambda: ctx.bars["close"]
        .pct_change()
        .rolling(window, min_periods=window)
        .std(),
    )
    vol_value = float(vol[ctx.signal_idx])
    if not math.isfinite(vol_value) or vol_value <= 0:
        return math.nan
    return ctx.equity * policy.sizing_risk_pct / vol_value


def entry_budget_for(
    cfg: BacktestConfig,
    portfolio: Portfolio,
    bars: pd.DataFrame,
    signal_idx: int,
    *,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> float:
    """Dollar budget for the next entry under ``cfg.sizing_rule``.

    ``equal_slot`` short-circuits to ``portfolio.entry_budget()`` so the
    default path is bit-identical to the pre-sizing engine. Every other rule
    is clamped to ``[0, entry_budget()]``.

    ``series_cache`` is ``_FrameCache.sizing_series`` for this ticker's frame,
    so a run opening many entries on one name computes its ATR once. Omitting
    it is correct and only slower.
    """
    base = portfolio.entry_budget()
    rule = cfg.sizing_rule
    if rule == "equal_slot":
        return base
    func = _SIZERS.get(rule)
    if func is None:
        raise ValueError(
            f"unknown sizing rule {rule!r}; expected one of "
            f"{', '.join(available_sizing_rules())}"
        )
    ctx = SizingContext(
        equity=portfolio.initial_capital,
        base_budget=base,
        stop_loss=cfg.stop_loss,
        policy=cfg,
        bars=bars,
        signal_idx=signal_idx,
        series_cache=series_cache,
    )
    raw = func(ctx)
    if not math.isfinite(raw):
        return base
    return min(max(raw, 0.0), base)


__all__ = [
    "SizingContext",
    "available_sizing_rules",
    "entry_budget_for",
    "sizer",
]
