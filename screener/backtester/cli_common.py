"""Shared CLI helpers for backtest commands."""

from __future__ import annotations

import click


DEFAULT_BENCHMARK = {"us": "SPY", "india": "^NSEI"}
DEFAULT_MIN_PRICE = {"us": 1.0, "india": 10.0}
DEFAULT_MIN_ADV = {"us": 1_000.0, "india": 100_000.0}


def resolve_strategy_exprs(strategy_name, entry_expr, exit_expr):
    from screener.backtester.strategies import resolve_strategy

    if strategy_name:
        strategy = resolve_strategy(strategy_name)
        entry_expr = entry_expr or strategy.entry
        exit_expr = exit_expr or strategy.exit
    if not entry_expr:
        raise click.UsageError("--entry (or --strategy) is required.")
    return entry_expr, exit_expr


def build_slippage_model(slippage_model, slippage_bps, half_spread_bps, vol_impact_k):
    from screener.backtester.slippage import (
        CompositeSlippage,
        FixedBpsSlippage,
        HalfSpreadSlippage,
        VolumeImpactSlippage,
    )

    if slippage_model == "fixed":
        return FixedBpsSlippage(float(slippage_bps))
    if slippage_model == "half-spread":
        return HalfSpreadSlippage(float(half_spread_bps))
    if slippage_model == "vol-impact":
        return VolumeImpactSlippage(float(vol_impact_k))
    return CompositeSlippage(
        models=(
            FixedBpsSlippage(float(slippage_bps)),
            HalfSpreadSlippage(float(half_spread_bps)),
            VolumeImpactSlippage(float(vol_impact_k)),
        )
    )


def parse_partial_exits(partial_exit_args) -> tuple[tuple[float, float], ...]:
    if not partial_exit_args:
        return ()
    parsed: list[tuple[float, float]] = []
    for raw in partial_exit_args:
        try:
            profit_s, shares_s = raw.split(":", 1)
            parsed.append((float(profit_s), float(shares_s)))
        except ValueError as exc:
            raise click.UsageError(
                f"--partial-exit expects PROFIT_FRAC:SHARES_FRAC, got {raw!r}"
            ) from exc
    return tuple(parsed)


def resolve_min_filters(market, min_price, min_avg_dollar_volume):
    resolved_min_price = (
        DEFAULT_MIN_PRICE.get(market) if min_price is None else min_price
    )
    if resolved_min_price == 0:
        resolved_min_price = None
    resolved_min_adv = (
        DEFAULT_MIN_ADV.get(market)
        if min_avg_dollar_volume is None
        else min_avg_dollar_volume
    )
    if resolved_min_adv == 0:
        resolved_min_adv = None
    return resolved_min_price, resolved_min_adv
