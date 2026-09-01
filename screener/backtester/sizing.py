"""Rule-based per-entry position sizing.

Each rule maps entry-time context to a dollar budget for the new position.
Most results are clamped to ``Portfolio.entry_budget()``. The
``reinvested_equal_slot`` rule can grow above the initial slot ceiling, but it
remains capped by available cash.

Risk-rule sizing equity remains the portfolio's ``initial_capital``. The
``reinvested_equal_slot`` rule is the explicit exception and reads current
marked-to-market equity.

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

    ``base_budget`` is ``Portfolio.entry_budget()``. It is the hard cap for
    fixed and risk sizing, but not for ``reinvested_equal_slot``.
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


@sizer("reinvested_equal_slot")
def _reinvested_equal_slot(ctx: SizingContext) -> float:
    """Allocate one equal slot from current marked-to-market portfolio equity."""
    return ctx.equity / max(ctx.policy.top, 1)


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
        lambda: (
            ctx.bars["close"].pct_change().rolling(window, min_periods=window).std()
        ),
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
    current_equity: float | None = None,
    free_slots: int = 1,
    series_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> float:
    """Dollar budget for the next entry under ``cfg.sizing_rule``.

    ``equal_slot`` short-circuits to ``portfolio.entry_budget()`` so the
    default path is bit-identical to the pre-sizing engine. Risk rules are
    clamped to ``[0, entry_budget()]``. ``reinvested_equal_slot`` is clamped
    by this entry's fair share of available cash: ``cash / free_slots``, where
    ``free_slots`` is the number of slots (this one included) still to be
    filled from the same cash balance. Without that split the first slot of a
    batch could absorb the whole balance and leave the rest with a zero
    budget.

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
    compounds_slots = rule == "reinvested_equal_slot"
    sizing_equity = (
        float(current_equity)
        if compounds_slots and current_equity is not None
        else portfolio.initial_capital
    )
    ctx = SizingContext(
        equity=sizing_equity,
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
    if compounds_slots:
        ceiling = max(portfolio.cash(), 0.0) / max(int(free_slots), 1)
    else:
        ceiling = base
    return min(max(raw, 0.0), ceiling)


def marked_portfolio_equity(
    portfolio: Portfolio,
    bars_by_ticker: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
) -> float:
    """Mark open positions at the last close known on or before ``as_of``."""
    marks: dict[str, float] = {}
    for ticker in portfolio.open_tickers():
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        # bars.index is a sorted DatetimeIndex; searchsorted is O(log n) where
        # a boolean mask is O(n) per ticker per simulated day.
        pos = int(bars.index.searchsorted(as_of, side="right"))
        if pos > 0:
            marks[ticker] = float(bars["close"].iloc[pos - 1])
    return portfolio.marked_equity(marks, as_of=as_of)


def entry_opens_no_shares(entry_budget: float, entry_shares: float | None) -> bool:
    """Whether an entry quote would create a position holding no shares.

    ``_SlotState.entry_shares`` is only populated for liquidity-aware fill
    models; otherwise ``Portfolio.open`` derives the share count from the
    budget, so a non-positive budget is the empty case. A zero-share position
    still occupies its slot and consumes the candidate, so callers skip it.
    """
    if entry_shares is not None:
        return float(entry_shares) <= 0.0
    return float(entry_budget) <= 0.0


def sizing_allows_slot_growth(sizing_rule: str) -> bool:
    """Return whether a sizing rule may exceed the initial fixed slot ceiling."""
    return sizing_rule == "reinvested_equal_slot"


__all__ = [
    "SizingContext",
    "entry_opens_no_shares",
    "available_sizing_rules",
    "entry_budget_for",
    "marked_portfolio_equity",
    "sizer",
    "sizing_allows_slot_growth",
]
