"""Shared CLI helpers for backtest commands."""

from __future__ import annotations

import click

from screener.markets import MARKETS

DEFAULT_BENCHMARK = {name: market.benchmark for name, market in MARKETS.items()}
DEFAULT_MIN_PRICE = {name: market.min_price for name, market in MARKETS.items()}
DEFAULT_MIN_ADV = {name: market.min_adv for name, market in MARKETS.items()}


def resolve_strategy_exprs(strategy_name, entry_expr, exit_expr):
    from screener.backtester.strategies import resolve_strategy

    if strategy_name:
        try:
            strategy = resolve_strategy(strategy_name)
        except KeyError as exc:
            raise click.UsageError(str(exc)) from exc
        entry_expr = entry_expr or strategy.entry
        exit_expr = exit_expr or strategy.exit
    if not entry_expr:
        raise click.UsageError("--entry (or --strategy) is required.")
    return entry_expr, exit_expr


def referenced_fundamental_fields(entry_expr, exit_expr):
    """Return the known fundamental fields referenced by the entry/exit exprs.

    Fundamental identifiers (e.g. ``revenue_up_3q``) only resolve once a
    fundamentals provider has merged those dated columns into the bars, so
    callers use this to decide whether a strategy needs fundamentals enabled.
    """
    from screener.backtester.fundamentals import DEFAULT_FUNDAMENTAL_FIELDS
    from screener.backtester.pine import collect_names, parse

    names: set[str] = set()
    for expr in (entry_expr, exit_expr):
        if expr:
            names |= collect_names(parse(expr))
    return names & set(DEFAULT_FUNDAMENTAL_FIELDS)


def build_slippage_model(
    slippage_model,
    slippage_bps,
    half_spread_bps,
    vol_impact_k,
    *,
    spread_proxy: bool = False,
):
    from screener.backtester.slippage import (
        CompositeSlippage,
        EstimatedHalfSpreadSlippage,
        FixedBpsSlippage,
        HalfSpreadSlippage,
        SlippageModel,
        VolumeImpactSlippage,
    )

    model: SlippageModel
    if slippage_model == "fixed":
        model = FixedBpsSlippage(bps=float(slippage_bps))
    elif slippage_model == "half-spread":
        model = HalfSpreadSlippage(half_spread_bps=float(half_spread_bps))
    elif slippage_model == "vol-impact":
        model = VolumeImpactSlippage(k=float(vol_impact_k))
    else:
        model = CompositeSlippage(
            models=(
                FixedBpsSlippage(bps=float(slippage_bps)),
                HalfSpreadSlippage(half_spread_bps=float(half_spread_bps)),
                VolumeImpactSlippage(k=float(vol_impact_k)),
            )
        )
    if spread_proxy:
        return CompositeSlippage(models=(model, EstimatedHalfSpreadSlippage()))
    return model


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
