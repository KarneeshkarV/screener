"""Shared execution-cost arithmetic for every backtest engine."""

from __future__ import annotations

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


__all__ = [
    "bps_fraction",
    "fixed_bps_fill",
    "fixed_bps_round_trip",
    "net_round_trip_return",
]
