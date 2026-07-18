"""Compatibility wrapper for the canonical backtest execution primitives."""

from __future__ import annotations

from screener.backtester.costs import CostModel
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
