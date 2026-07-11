"""Compatibility wrapper for the canonical backtest execution primitives."""

from __future__ import annotations

from screener.backtester.execution import fixed_bps_round_trip


def apply_slippage(
    entry_price: float, exit_price: float, slippage_bps: float
) -> tuple[float, float]:
    """Apply symmetric bps slippage to entry/exit fill prices.

    Entry is paid above the reference price (buy), exit below (sell). The
    arithmetic mirrors the previous inline form bit-for-bit so backtest
    numbers are unchanged.
    """
    return fixed_bps_round_trip(entry_price, exit_price, slippage_bps)
