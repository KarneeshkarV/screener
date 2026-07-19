"""Shared execution-cost arithmetic for every backtest engine."""

from __future__ import annotations

from screener.backtester.costs import CostModel
from screener.backtester.slippage import (
    FixedBpsSlippage,
    Side,
    apply_slippage,
)


def bps_fraction(bps: float) -> float:
    """Convert basis points to a decimal rate."""
    return float(bps) / 10_000.0


def fixed_bps_fill(reference_price: float, side: Side, bps: float) -> float:
    """Apply the canonical fixed-bps slippage model to one reference price."""
    return apply_slippage(FixedBpsSlippage(bps=bps), reference_price, side)


def fixed_bps_round_trip(
    entry_reference: float,
    exit_reference: float,
    slippage_bps: float,
) -> tuple[float, float]:
    """Return adverse buy and sell fills for a fixed-bps round trip."""
    return (
        fixed_bps_fill(entry_reference, "buy", slippage_bps),
        fixed_bps_fill(exit_reference, "sell", slippage_bps),
    )


def net_round_trip_return(
    entry_fill: float,
    exit_fill: float,
    commission_bps: float,
) -> tuple[float, float]:
    """Return ``(raw_return, net_return)`` using one round-trip commission rate."""
    raw = (exit_fill / entry_fill) - 1.0
    return raw, raw - bps_fraction(commission_bps)


def apply_round_trip_costs(
    entry_price: float,
    exit_price: float,
    cost_model: CostModel,
    *,
    shares: float | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Apply per-side cost-model fees to a long round-trip.

    Returns ``(raw_return, net_return, fees_breakdown)`` where fees are
    absolute currency amounts per component for a unit trade (1 share when
    ``shares`` is omitted). Return drag is buy-side fraction + sell-side
    fraction of each leg's notional, matching
    :meth:`CostModel.side_cost_fraction` on each side.

    For :class:`~screener.backtester.costs.FlatCommission` with ``bps=C``,
    each side costs ``C/10000`` so the round-trip drag is ``2C/10000``.
    """
    raw = (exit_price / entry_price) - 1.0 if entry_price else 0.0
    share_count = 1.0 if shares is None else float(shares)
    buy_notional = abs(float(entry_price) * share_count)
    sell_notional = abs(float(exit_price) * share_count)

    buy_bd = cost_model.side_cost_breakdown("buy", buy_notional, share_count)
    sell_bd = cost_model.side_cost_breakdown("sell", sell_notional, share_count)

    fees: dict[str, float] = {}
    for breakdown in (buy_bd, sell_bd):
        for name, amount in breakdown.items():
            amt = float(amount)
            if amt <= 0.0:
                continue
            fees[name] = fees.get(name, 0.0) + amt

    buy_frac = (
        sum(float(v) for v in buy_bd.values()) / buy_notional if buy_notional else 0.0
    )
    sell_frac = (
        sum(float(v) for v in sell_bd.values()) / sell_notional
        if sell_notional
        else 0.0
    )
    # Clamp negative model output the same way the portfolio does.
    if buy_frac < 0.0:
        buy_frac = 0.0
    if sell_frac < 0.0:
        sell_frac = 0.0
    return raw, raw - buy_frac - sell_frac, fees


__all__ = [
    "apply_round_trip_costs",
    "bps_fraction",
    "fixed_bps_fill",
    "fixed_bps_round_trip",
    "net_round_trip_return",
]
