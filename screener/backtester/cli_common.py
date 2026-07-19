"""Shared CLI helpers for backtest commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from screener.markets import MARKETS

if TYPE_CHECKING:
    from screener.backtester.slippage import SlippageModel

DEFAULT_BENCHMARK = {name: market.benchmark for name, market in MARKETS.items()}
DEFAULT_MIN_PRICE = {name: market.min_price for name, market in MARKETS.items()}
DEFAULT_MIN_ADV = {name: market.min_adv for name, market in MARKETS.items()}


def resolve_strategy_exprs(
    strategy_name: str | None,
    entry_expr: str | None,
    exit_expr: str | None,
) -> tuple[str, str | None]:
    from screener.strategies.expressions import resolve_strategy

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


def referenced_fundamental_fields(
    entry_expr: str | None, exit_expr: str | None
) -> set[str]:
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
    slippage_model: str,
    slippage_bps: float,
    half_spread_bps: float,
    vol_impact_k: float,
    *,
    spread_proxy: bool = False,
) -> SlippageModel:
    from screener.backtester.slippage import (
        CompositeSlippage,
        EstimatedHalfSpreadSlippage,
        FixedBpsSlippage,
        HalfSpreadSlippage,
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


def sizing_options(command):
    """Attach the shared per-entry position-sizing options to a backtest command.

    All rules size DOWN from the equal-slot budget (never above it); the
    ``equal_slot`` default reproduces the legacy fixed-slot engine exactly.
    """
    from screener.backtester.sizing import available_sizing_rules

    options = [
        click.option(
            "--sizing",
            "sizing_rule",
            type=click.Choice(available_sizing_rules()),
            default="equal_slot",
            show_default=True,
            help=(
                "Per-entry position sizing. 'equal_slot' = legacy fixed slots; "
                "every other rule sizes down from the slot budget."
            ),
        ),
        click.option(
            "--sizing-risk-pct",
            type=float,
            default=0.01,
            show_default=True,
            help=(
                "Fraction of initial capital risked per trade (fixed_risk/atr_risk) "
                "or daily volatility target (inverse_vol)."
            ),
        ),
        click.option(
            "--sizing-position-pct",
            type=float,
            default=0.10,
            show_default=True,
            help="Fraction of initial capital per position (fixed_fraction).",
        ),
        click.option(
            "--sizing-atr-window",
            type=int,
            default=14,
            show_default=True,
            help="ATR lookback (bars) for atr_risk sizing.",
        ),
        click.option(
            "--sizing-atr-multiple",
            type=float,
            default=2.0,
            show_default=True,
            help="ATR multiple treated as the risk-per-share for atr_risk sizing.",
        ),
        click.option(
            "--sizing-vol-window",
            type=int,
            default=20,
            show_default=True,
            help="Return-volatility lookback (bars) for inverse_vol sizing.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def intraday_options(command):
    """Attach the shared intraday session-exit option to a backtest command."""
    options = [
        click.option(
            "--intraday-only",
            is_flag=True,
            default=False,
            help=(
                "Force positions flat on the last bar of each trading session "
                "(intraday intervals only; rejects --interval 1d)."
            ),
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def build_data_policy(
    *,
    interval: str,
    price_adjustment: str,
    intraday_only: bool = False,
):
    """Build a ``DataPolicy``, mapping pydantic validation errors to Click usage."""
    from pydantic import ValidationError

    from screener.backtester.models import DataPolicy

    try:
        return DataPolicy(
            interval=interval,
            price_adjustment=price_adjustment,  # type: ignore[arg-type]
            intraday_only=intraday_only,
        )
    except ValidationError as exc:
        raise click.UsageError(str(exc)) from exc


def validate_sizing(sizing_rule: str, stop_loss: float | None) -> None:
    if sizing_rule == "fixed_risk" and (stop_loss is None or stop_loss <= 0):
        raise click.UsageError("--sizing fixed_risk requires --stop-loss.")


def resolve_min_filters(
    market: str,
    min_price: float | None,
    min_avg_dollar_volume: float | None,
) -> tuple[float | None, float | None]:
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
