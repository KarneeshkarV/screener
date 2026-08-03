"""Compatibility exports for earnings event-study metrics.

The implementation is neutral because option position trades use the same
summary contract without depending on this subsystem.
"""

from screener.ledger import compute_event_trade_summary

# Public compatibility name retained for callers of the earnings subsystem.
compute_backtest_summary = compute_event_trade_summary

__all__ = ["compute_backtest_summary"]
