"""Execution-cost helpers shared by the earnings backtest engines."""

from __future__ import annotations


def apply_slippage(
    entry_price: float, exit_price: float, slippage_bps: float
) -> tuple[float, float]:
    """Apply symmetric bps slippage to entry/exit fill prices.

    Entry is paid above the reference price (buy), exit below (sell). The
    arithmetic mirrors the previous inline form bit-for-bit so backtest
    numbers are unchanged.
    """
    entry_price *= 1 + slippage_bps / 10_000
    exit_price *= 1 - slippage_bps / 10_000
    return entry_price, exit_price
