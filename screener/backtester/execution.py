"""Compatibility re-exports for the former execution-cost module.

New code imports cost arithmetic from :mod:`screener.backtester.costs` and
fixed-bps price adjustments from :mod:`screener.backtester.slippage`.
"""

from screener.backtester.costs import (
    apply_round_trip_costs,
    bps_fraction,
    net_round_trip_return,
)
from screener.backtester.slippage import fixed_bps_fill, fixed_bps_round_trip

__all__ = [
    "apply_round_trip_costs",
    "bps_fraction",
    "fixed_bps_fill",
    "fixed_bps_round_trip",
    "net_round_trip_return",
]
