"""Single execution-realism seam: where every trade fill price comes from.

Historically the "what price does this trade fill at" logic was scattered across
:mod:`screener.backtester.core` — entry order-type dispatch (MOO/MOC/limit),
gap-aware stop/target reference resolution, slippage routing (legacy bps factor
vs. a pluggable :class:`~screener.backtester.slippage.SlippageModel`), and the
partial-exit pricing path each recomputed the same primitives inline.

:class:`FillModel` consolidates all of that behind two questions:

* :meth:`entry_price` — given a signal bar, which bar do we enter on and at what
  (slipped) price? Encapsulates order-type dispatch + buy-side slippage.
* :meth:`exit_price` — given an exit bar, reason and (optional) trigger level,
  what is the (slipped) sell price? Encapsulates gap-aware reference resolution
  + sell-side slippage.

The numerical behaviour is identical to the original inline code; the module is
purely a seam. The pluggable slippage layer in
:mod:`screener.backtester.slippage` is kept underneath unchanged.

The free functions ``_resolve_entry_fill``, ``_resolve_stop_fill`` and
``_resolve_target_fill`` remain importable (some are re-exported by ``engine``
and exercised directly by tests); :class:`FillModel` delegates to them so there
is a single implementation of each primitive.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.slippage import Side, apply_slippage, needs_liquidity_inputs


class PriceArrays(Protocol):
    """The float OHLC columns of a bars frame, already extracted to NumPy.

    Structural so ``fills`` stays free of a ``core`` import (``core`` imports
    ``fills``); ``core._FrameCache`` satisfies it.
    """

    open_arr: np.ndarray
    low_arr: np.ndarray
    close_arr: np.ndarray


def _resolve_entry_fill(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    arrays: Optional[PriceArrays] = None,
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    """Resolve entry bar index and reference fill price from order settings.

    When ``arrays`` is supplied the prices are read from the caller's cached
    NumPy columns instead of ``bars.iloc[i][col]``, which builds a throwaway
    row Series per lookup. Same values, same order — the exit path already
    reads the identical arrays.
    """
    if signal_idx + 1 >= len(bars):
        return None, None, "no post-signal entry bar"
    order = cfg.entry_order_type
    if order == "moo":
        entry_idx = signal_idx + 1
        if arrays is not None:
            return entry_idx, float(arrays.open_arr[entry_idx]), None
        return entry_idx, float(bars.iloc[entry_idx]["open"]), None
    if order == "moc":
        entry_idx = signal_idx + 1
        if arrays is not None:
            return entry_idx, float(arrays.close_arr[entry_idx]), None
        return entry_idx, float(bars.iloc[entry_idx]["close"]), None
    if order == "limit":
        if cfg.entry_limit_bps is None:
            return None, None, "limit order requires entry_limit_bps"
        if arrays is not None:
            signal_close = float(arrays.close_arr[signal_idx])
        else:
            signal_close = float(bars.iloc[signal_idx]["close"])
        limit_price = signal_close * (1.0 - cfg.entry_limit_bps / 10_000.0)
        if arrays is not None:
            for i in range(signal_idx + 1, len(bars)):
                if float(arrays.low_arr[i]) <= limit_price:
                    ref = min(float(arrays.open_arr[i]), limit_price)
                    return i, ref, None
            return None, None, "limit order never filled in available window"
        for i in range(signal_idx + 1, len(bars)):
            bar = bars.iloc[i]
            low = float(bar["low"])
            if low <= limit_price:
                ref = min(float(bar["open"]), limit_price)
                return i, ref, None
        return None, None, "limit order never filled in available window"
    return None, None, f"unknown entry_order_type: {order}"


def _resolve_stop_fill(bar_open: float, stop_ref: float, gap_fills: bool) -> float:
    """Reference price for a gap-aware stop-loss fill."""
    if gap_fills and bar_open <= stop_ref:
        return bar_open
    return stop_ref


def _resolve_target_fill(bar_open: float, target_ref: float, gap_fills: bool) -> float:
    """Reference price for a gap-aware take-profit fill."""
    if gap_fills and bar_open >= target_ref:
        return bar_open
    return target_ref


class FillModel:
    """The single source of truth for trade fill prices in a backtest run.

    One instance is constructed per backtest run from the immutable
    :class:`~screener.backtester.models.BacktestConfig` and threaded through the
    engines; it holds no per-bar mutable state, so a single instance is reused
    for every fill of the run.
    """

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg

    @property
    def needs_liquidity_inputs(self) -> bool:
        """Whether the configured slippage model uses size/liquidity inputs."""
        assert self.cfg.slippage_model is not None
        return needs_liquidity_inputs(self.cfg.slippage_model)

    # ── slippage routing ─────────────────────────────────────────────
    def _apply_slip(
        self,
        ref_price: float,
        side: Side,
        *,
        shares: float = 0.0,
        adv_shares: float = 0.0,
        sigma_daily: float = 0.0,
        half_spread: float = 0.0,
    ) -> float:
        """Run ``cfg.slippage_model`` over a reference price."""
        cfg = self.cfg
        assert cfg.slippage_model is not None
        return apply_slippage(
            cfg.slippage_model,
            ref_price,
            side,
            shares=shares,
            adv=adv_shares,
            sigma_daily=sigma_daily,
            half_spread=half_spread,
        )

    # ── entry side ───────────────────────────────────────────────────
    def entry_price(
        self,
        bars: pd.DataFrame,
        signal_idx: int,
        *,
        budget: Optional[float] = None,
        adv_shares: float = 0.0,
        sigma_daily: float = 0.0,
        half_spread: float = 0.0,
        arrays: Optional[PriceArrays] = None,
    ) -> tuple[Optional[int], Optional[float], Optional[str]]:
        """Resolve the entry bar index and the slipped (buy-side) fill price.

        Returns ``(entry_idx, entry_fill, warning)``. On failure the index and
        price are ``None`` and ``warning`` explains why, mirroring the original
        ``_resolve_entry_fill`` contract — slippage is applied to the resolved
        reference before returning. ``arrays``, when given, supplies the cached
        NumPy OHLC columns to read prices from instead of ``bars.iloc``.
        """
        entry_idx, entry_ref, warn = _resolve_entry_fill(
            bars, signal_idx, self.cfg, arrays
        )
        if entry_idx is None or entry_ref is None:
            return None, None, warn
        # Entry price and acquired shares are circular when impact depends on
        # order size. Use the pre-slippage reference (including commission) as
        # the deterministic sizing convention; Portfolio.open then computes
        # the actual shares from the impacted fill without a hidden iteration.
        if budget is None:
            budget = self.cfg.initial_capital / max(self.cfg.top, 1)
        commission = self.cfg.commission_bps / 10_000.0
        gross_reference = entry_ref * (1.0 + commission)
        shares = budget / gross_reference if gross_reference > 0.0 else 0.0
        fill = self._apply_slip(
            entry_ref,
            "buy",
            shares=shares,
            adv_shares=adv_shares,
            sigma_daily=sigma_daily,
            half_spread=half_spread,
        )
        return entry_idx, fill, None

    # ── exit side ────────────────────────────────────────────────────
    def exit_price(
        self,
        *,
        reason: str,
        bar_open: float = 0.0,
        level: Optional[float] = None,
        close: float = 0.0,
        side: Side = "sell",
        shares: float = 0.0,
        adv_shares: float = 0.0,
        sigma_daily: float = 0.0,
        half_spread: float = 0.0,
    ) -> float:
        """Slipped exit price for a given exit ``reason``.

        ``reason`` selects the gap-aware reference resolution:

        * ``"stop"`` / ``"trail"`` — gap-down aware against ``level`` (the stop
          or trailing-stop reference) using ``bar_open``.
        * ``"target"`` — gap-up aware against ``level`` (the target reference).
        * ``"exit_expr"`` / ``"time"`` / ``"eod"`` / ``"session"`` — fill at
          ``close`` (session-last is a market-on-close flat).

        Slippage is applied to the resolved reference before returning.
        """
        gap_fills = self.cfg.gap_fills
        if reason in ("stop", "trail"):
            assert level is not None
            ref = _resolve_stop_fill(bar_open, level, gap_fills)
        elif reason == "target":
            assert level is not None
            ref = _resolve_target_fill(bar_open, level, gap_fills)
        else:
            ref = close
        return self._apply_slip(
            ref,
            side,
            shares=shares,
            adv_shares=adv_shares,
            sigma_daily=sigma_daily,
            half_spread=half_spread,
        )
