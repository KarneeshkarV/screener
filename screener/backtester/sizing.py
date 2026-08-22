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

``ema_spread`` is the trend-strength rule: it reads the strategy's *own* EMA
pair out of its entry/exit expressions and gives a wider fast-minus-slow gap a
bigger slice of the slot. Because every rule is clamped to the slot ceiling,
"more weight for a wider gap" is expressed as "less weight for a narrower gap":
a run under ``ema_spread`` holds strictly less gross exposure than the same run
under ``equal_slot``, so compare the two on risk-adjusted terms, not on total
return alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

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


SizerFunc = Callable[[SizingContext], float]

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
    atr = wilder_atr(
        bars["high"],
        bars["low"],
        bars["close"],
        policy.sizing_atr_window,
        min_periods=policy.sizing_atr_window,
    )
    atr_value = float(atr.iloc[ctx.signal_idx])
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
    vol = (
        ctx.bars["close"]
        .pct_change()
        .rolling(policy.sizing_vol_window, min_periods=policy.sizing_vol_window)
        .std()
    )
    vol_value = float(vol.iloc[ctx.signal_idx])
    if not math.isfinite(vol_value) or vol_value <= 0:
        return math.nan
    return ctx.equity * policy.sizing_risk_pct / vol_value


@lru_cache(maxsize=512)
def _ma_windows_in(
    entry_expr: str, exit_expr: str | None, func: str
) -> tuple[int, ...]:
    """Sorted, de-duplicated ``func(source, N)`` windows named by the strategy.

    Parsing uses the same Pine front end the signal evaluator uses, so a window
    counts here only if it would really be computed at signal time. Only
    integer-literal lengths are collected -- ``pine`` requires that anyway.
    An unparseable expression yields no windows rather than raising: sizing must
    never be the thing that fails a run whose signals already evaluated.
    """
    from screener.backtester.pine import (
        BinOp,
        BoolOp,
        Call,
        Compare,
        Node,
        Not,
        Num,
        PineError,
        UnaryOp,
        parse,
    )

    found: set[int] = set()

    def visit(node: Node) -> None:
        if isinstance(node, Call):
            if node.name == func and len(node.args) == 2:
                length = node.args[1]
                if isinstance(length, Num) and float(length.value).is_integer():
                    found.add(int(length.value))
            for arg in node.args:
                visit(arg)
        elif isinstance(node, (BinOp, Compare, BoolOp)):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, (UnaryOp, Not)):
            visit(node.operand)

    for expr in (entry_expr, exit_expr):
        if not expr:
            continue
        try:
            visit(parse(expr))
        except PineError:
            continue
    return tuple(sorted(found))


def ma_spread_windows(cfg: BacktestConfig, func: str) -> tuple[int, int]:
    """``(fast, slow)`` ``func`` windows a spread rule weights ``cfg`` by.

    The strategy's own pair wins when its expressions name two or more windows
    of that moving average (fastest against slowest). Anything else -- one
    window, or none at all -- falls back to the configured
    ``sizing_ema_fast``/``sizing_ema_slow``, defaulting to 50/200. The fallback
    is shared on purpose: it is the pair the user asked for, and it keeps the
    ema and sma arms weighting the same lengths whenever the strategy itself is
    silent, so the two arms differ only in the average, not the lookback.
    """
    windows = _ma_windows_in(cfg.entry_expr, cfg.exit_expr, func)
    if len(windows) >= 2:
        return windows[0], windows[-1]
    return cfg.sizing_ema_fast, cfg.sizing_ema_slow


def ema_spread_windows(cfg: BacktestConfig) -> tuple[int, int]:
    """``(fast, slow)`` EMA windows the ``ema_spread`` rule weights ``cfg`` by."""
    return ma_spread_windows(cfg, "ema")


def _spread_weight(policy: BacktestConfig, spread: float) -> float:
    """Map a normalized gap onto ``[spread_floor, 1]`` slot fractions."""
    return min(
        max(spread / policy.sizing_ema_spread_cap, policy.sizing_ema_spread_floor), 1.0
    )


@sizer("ema_spread")
def _ema_spread(ctx: SizingContext) -> float:
    # Trend-strength weighting. The normalized gap ``(fast - slow) / slow`` is
    # scale-free, so one cap works across a whole cross-section; it maps
    # linearly onto ``[spread_floor, 1]`` and scales the equal-slot budget.
    # The floor keeps a flat or inverted gap from sizing a qualifying entry down
    # to zero shares, which would override the strategy's own entry criteria and
    # silently change the trade count rather than only the weights.
    policy = ctx.policy
    fast_window, slow_window = ema_spread_windows(policy)
    close = ctx.bars["close"]
    fast = close.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    slow = close.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    fast_value = float(fast.iloc[ctx.signal_idx])
    slow_value = float(slow.iloc[ctx.signal_idx])
    if (
        not math.isfinite(fast_value)
        or not math.isfinite(slow_value)
        or slow_value <= 0
    ):
        return math.nan
    spread = (fast_value - slow_value) / slow_value
    return ctx.base_budget * _spread_weight(policy, spread)


@sizer("sma_spread")
def _sma_spread(ctx: SizingContext) -> float:
    # Same shape as ``ema_spread`` on the simple average. Most strategies in
    # the research set are sma-based, so this arm reads their real moving
    # averages where ``ema_spread`` has to fall back to 50/200.
    policy = ctx.policy
    fast_window, slow_window = ma_spread_windows(policy, "sma")
    close = ctx.bars["close"]
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    fast_value = float(fast.iloc[ctx.signal_idx])
    slow_value = float(slow.iloc[ctx.signal_idx])
    if (
        not math.isfinite(fast_value)
        or not math.isfinite(slow_value)
        or slow_value <= 0
    ):
        return math.nan
    return ctx.base_budget * _spread_weight(
        policy, (fast_value - slow_value) / slow_value
    )


@sizer("ma_extension")
def _ma_extension(ctx: SizingContext) -> float:
    # Price-against-trend instead of trend-against-trend: how far the close sits
    # above the strategy's slow EMA. Same normalization and clamp as the spread
    # rules, so the three arms differ only in what they measure.
    policy = ctx.policy
    _, slow_window = ma_spread_windows(policy, "ema")
    close = ctx.bars["close"]
    slow = close.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    slow_value = float(slow.iloc[ctx.signal_idx])
    close_value = float(close.iloc[ctx.signal_idx])
    if (
        not math.isfinite(slow_value)
        or not math.isfinite(close_value)
        or slow_value <= 0
    ):
        return math.nan
    return ctx.base_budget * _spread_weight(
        policy, (close_value - slow_value) / slow_value
    )


def entry_budget_for(
    cfg: BacktestConfig,
    portfolio: Portfolio,
    bars: pd.DataFrame,
    signal_idx: int,
) -> float:
    """Dollar budget for the next entry under ``cfg.sizing_rule``.

    ``equal_slot`` short-circuits to ``portfolio.entry_budget()`` so the
    default path is bit-identical to the pre-sizing engine. Every other rule
    is clamped to ``[0, entry_budget()]``.
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
    )
    raw = func(ctx)
    if not math.isfinite(raw):
        return base
    return min(max(raw, 0.0), base)


__all__ = [
    "SizingContext",
    "available_sizing_rules",
    "ema_spread_windows",
    "entry_budget_for",
    "ma_spread_windows",
    "sizer",
]
