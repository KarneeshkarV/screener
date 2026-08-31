"""Shared CLI helpers for backtest commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import click

from screener.backtester.models import SUPPORTED_INTERVALS
from screener.gate_options import ADV_WINDOW_DEFAULT, gate_options
from screener.markets import MARKETS

if TYPE_CHECKING:
    from pathlib import Path

    from screener.backtester.slippage import SlippageModel

DEFAULT_BENCHMARK = {name: market.benchmark for name, market in MARKETS.items()}
DEFAULT_MIN_PRICE = {name: market.min_price for name, market in MARKETS.items()}
DEFAULT_MIN_ADV = {name: market.min_adv for name, market in MARKETS.items()}

RANK_EXIT_PRESETS = {"weekly": 5, "monthly": 21}

#: Re-exported from :mod:`screener.gate_options`, which owns the gate flags
#: the screen and the rolling backtest share. Kept importable from here so the
#: backtester's existing callers do not have to learn a second module name.
__all__ = ["ADV_WINDOW_DEFAULT"]


def parse_rank_exit(value: Any) -> tuple[int, bool] | None:
    """Parse ``--rank-exit`` into ``(period_in_bars, used_named_preset)``.

    ``None`` means the feature is off. Named presets count trading DAYS and
    are therefore only valid for daily bars; callers enforce that.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    preset = RANK_EXIT_PRESETS.get(text)
    if preset is not None:
        return preset, True
    try:
        parsed = int(text)
    except ValueError as exc:
        raise click.BadParameter(
            f"{value!r} is not 'weekly', 'monthly', or a positive integer"
        ) from exc
    if parsed < 1:
        raise click.BadParameter(f"{value!r} must be >= 1")
    return parsed, False


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


# ---------------------------------------------------------------------------
# Shared backtest command options
#
# ``backtest-historical`` and ``backtest-rolling`` share 31 long-form options.
# Each option is defined exactly once here as a ``mode``-aware builder so both
# commands keep byte-identical ``--help`` output (help text and defaults differ
# for a handful of options between the two modes). A command composes its own
# option order by calling ``backtest_options(mode, *names)`` around the
# mode-specific options it declares inline, so help ordering is preserved.
# ---------------------------------------------------------------------------

OptionDecorator = Callable[[Any], Any]
OptionBuilder = Callable[[str], OptionDecorator]


def _opt_hold(mode: str) -> OptionDecorator:
    return click.option(
        "--hold", type=int, default=20, help="Holding period (trading days)."
    )


def _opt_top(mode: str) -> OptionDecorator:
    help_text = (
        "Top N tickers to select."
        if mode == "historical"
        else "Concurrent portfolio slots."
    )
    return click.option("--top", type=int, default=10, help=help_text)


def _opt_entry(mode: str) -> OptionDecorator:
    return click.option(
        "--entry", "entry_expr", default=None, help="Pine-like entry expression."
    )


def _opt_exit(mode: str) -> OptionDecorator:
    return click.option(
        "--exit", "exit_expr", default=None, help="Pine-like exit expression."
    )


def _opt_strategy(mode: str) -> OptionDecorator:
    help_text = (
        "Named strategy shortcut (overrides --entry/--exit if given)."
        if mode == "historical"
        else "Named strategy shortcut."
    )
    return click.option("--strategy", "strategy_name", default=None, help=help_text)


def _opt_stop_loss(mode: str) -> OptionDecorator:
    return click.option(
        "--stop-loss", type=float, default=None, help="Stop loss (fraction, e.g. 0.08)."
    )


def _opt_take_profit(mode: str) -> OptionDecorator:
    return click.option(
        "--take-profit", type=float, default=None, help="Take profit (fraction)."
    )


def _opt_trailing_stop(mode: str) -> OptionDecorator:
    return click.option(
        "--trailing-stop", type=float, default=None, help="Trailing stop (fraction)."
    )


def _opt_slippage_bps(mode: str) -> OptionDecorator:
    return click.option(
        "--slippage-bps", type=float, default=0.0, help="Slippage per fill (bps)."
    )


def _opt_commission_bps(mode: str) -> OptionDecorator:
    return click.option(
        "--commission-bps", type=float, default=0.0, help="Commission per fill (bps)."
    )


def _opt_cost_model(mode: str) -> OptionDecorator:
    if mode == "historical":
        choices = ["flat", "india", "us_vested"]
        help_text = (
            "Statutory fee model. 'flat' applies --commission-bps on every fill "
            "(legacy). 'india' applies NSE equity delivery fees (STT, stamp duty, "
            "exchange, SEBI, GST, IPFT). 'us_vested' applies the Vested/DriveWealth "
            "US equity fee stack (brokerage cap, SEC Section 31, FINRA TAF)."
        )
    else:
        choices = ["flat", "india"]
        help_text = (
            "Statutory fee model. 'flat' applies --commission-bps on every fill "
            "(legacy). 'india' applies NSE equity delivery fees (STT, stamp duty, "
            "exchange, SEBI, GST, IPFT)."
        )
    return click.option(
        "--cost-model",
        type=click.Choice(choices),
        default="flat",
        show_default=True,
        help=help_text,
    )


def _opt_initial_capital(mode: str) -> OptionDecorator:
    return click.option("--initial-capital", type=float, default=100_000.0)


def _opt_benchmark(mode: str) -> OptionDecorator:
    return click.option(
        "--benchmark",
        default=None,
        help="Benchmark symbol (default: SPY for US, ^NSEI for India).",
    )


def _opt_tickers(mode: str) -> OptionDecorator:
    return click.option("--tickers", default=None, help="Comma-separated ticker list.")


def _opt_universe_file(mode: str) -> OptionDecorator:
    return click.option(
        "--universe-file", default=None, help="Path to newline-separated ticker file."
    )


def _opt_max_universe(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--max-universe",
            type=int,
            default=200,
            help="Cap supplied universe size before fetching prices. Pass 0 to disable.",
        )
    return click.option(
        "--max-universe",
        type=int,
        default=0,
        help="Cap universe size before fetching prices. Pass 0 to disable.",
    )


# The three liquidity gates below are the screen's gates too, so in rolling
# mode they are taken from :mod:`screener.gate_options` rather than restated:
# a flag that gates candidates must not be able to mean two things. Historical
# mode keeps its own wording - it does not share a candidate layer with the
# screen (a deliberate gap, see docs/plans/unify-screen-backtest.md).
def _opt_min_price(mode: str) -> OptionDecorator:
    if mode != "historical":
        return gate_options("min-price")
    return click.option(
        "--min-price",
        type=float,
        default=None,
        help="Minimum as-of close to admit a ticker. Default: $1 (US) / ₹10 (India). Pass 0 to disable.",
    )


def _opt_min_avg_dollar_volume(mode: str) -> OptionDecorator:
    if mode != "historical":
        return gate_options("min-avg-dollar-volume")
    return click.option(
        "--min-avg-dollar-volume",
        type=float,
        default=None,
        help="Minimum rolling-mean dollar volume (close*volume) over --adv-window. Default: $1,000 (US) / ₹100,000 (India). Pass 0 to disable.",
    )


def _opt_adv_window(mode: str) -> OptionDecorator:
    if mode != "historical":
        return gate_options("adv-window")
    return click.option(
        "--adv-window",
        type=int,
        default=ADV_WINDOW_DEFAULT,
        help="Lookback (bars) for average dollar-volume filter.",
    )


def _opt_refresh(mode: str) -> OptionDecorator:
    return click.option(
        "--refresh",
        is_flag=True,
        default=False,
        help="Bypass cached bars and re-download the price history.",
    )


def _opt_candidates(mode: str) -> OptionDecorator:
    return click.option(
        "--candidates",
        is_flag=True,
        default=False,
        help=(
            "Print the ranked candidate set for the last bar of the window and "
            "stop, running no trades. This is the same answer 'screener screen "
            "--universe' gives, so the two can be compared directly."
        ),
    )


def _opt_min_score(mode: str) -> OptionDecorator:
    return gate_options("min-score")


def _opt_regime_filter(mode: str) -> OptionDecorator:
    return gate_options("regime-filter")


def _opt_sector_neutral(mode: str) -> OptionDecorator:
    return gate_options("sector-neutral")


def _opt_earnings_blackout(mode: str) -> OptionDecorator:
    return gate_options("earnings-blackout")


def _opt_slippage_model(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--slippage-model",
            type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
            default="fixed",
            help="Slippage model. 'fixed' = constant bps (legacy); 'half-spread' adds quoted-spread cost; 'vol-impact' adds Almgren-Chriss sqrt-law impact; 'composite' sums all three.",
        )
    return click.option(
        "--slippage-model",
        type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
        default="fixed",
    )


def _opt_half_spread_bps(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--half-spread-bps",
            type=float,
            default=0.0,
            help="Half-spread charged on every fill (bps). Used by half-spread/composite.",
        )
    return click.option("--half-spread-bps", type=float, default=0.0)


def _opt_vol_impact_k(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--vol-impact-k",
            type=float,
            default=0.1,
            help="Coefficient for sqrt-law market impact (vol-impact/composite).",
        )
    return click.option("--vol-impact-k", type=float, default=0.1)


def _opt_no_gap_fills(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--no-gap-fills",
            is_flag=True,
            default=False,
            help="Disable gap-aware stop/target fills (fills always at reference price).",
        )
    return click.option("--no-gap-fills", is_flag=True, default=False)


def _opt_entry_order(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--entry-order",
            type=click.Choice(["moo", "moc", "limit"]),
            default="moo",
            help="Entry order type. moo=next-bar open (default); moc=next-bar close; limit=limit order at close*(1 - entry_limit_bps/1e4).",
        )
    return click.option(
        "--entry-order", type=click.Choice(["moo", "moc", "limit"]), default="moo"
    )


def _opt_entry_limit_bps(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--entry-limit-bps",
            type=float,
            default=None,
            help="Discount below signal-bar close for limit entries (bps).",
        )
    return click.option("--entry-limit-bps", type=float, default=None)


def _opt_partial_exit(mode: str) -> OptionDecorator:
    help_text = (
        "Scale-out tier as 'PROFIT_FRAC:SHARES_FRAC' (e.g. 0.05:0.5 = close half at +5%). Repeat to configure multiple tiers."
        if mode == "historical"
        else "Scale-out tier as PROFIT_FRAC:SHARES_FRAC."
    )
    return click.option(
        "--partial-exit", "partial_exit_args", multiple=True, help=help_text
    )


def _opt_price_adjustment(mode: str) -> OptionDecorator:
    if mode == "historical":
        return click.option(
            "--price-adjustment",
            type=click.Choice(["full", "splits_only", "none"]),
            default="full",
            help="Price-adjustment regime. full=legacy (yfinance auto_adjust=True); splits_only=split-adjust OHLC and credit dividends as cash; none=raw OHLC.",
        )
    return click.option(
        "--price-adjustment",
        type=click.Choice(["full", "splits_only", "none"]),
        default="full",
    )


def _opt_interval(mode: str) -> OptionDecorator:
    return click.option(
        "--interval",
        type=click.Choice(list(SUPPORTED_INTERVALS)),
        default="1d",
        show_default=True,
        help=(
            "Bar interval. Intraday values (1h/30m/15m/5m/1m) fetch from yfinance "
            "and are subject to its history caps (1m ~30d, 15m/30m ~60d, 1h ~730d)."
        ),
    )


def _opt_csv(mode: str) -> OptionDecorator:
    return click.option(
        "--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV."
    )


def _opt_report(mode: str) -> OptionDecorator:
    from pathlib import Path

    return click.option(
        "--report",
        "report_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Write a static, self-contained HTML tear-sheet to this file.",
    )


def _opt_open_report(mode: str) -> OptionDecorator:
    return click.option(
        "--open-report",
        is_flag=True,
        default=False,
        help="Open the generated HTML report in the default browser.",
    )


_OPTION_BUILDERS: dict[str, OptionBuilder] = {
    "hold": _opt_hold,
    "top": _opt_top,
    "entry": _opt_entry,
    "exit": _opt_exit,
    "strategy": _opt_strategy,
    "stop-loss": _opt_stop_loss,
    "take-profit": _opt_take_profit,
    "trailing-stop": _opt_trailing_stop,
    "slippage-bps": _opt_slippage_bps,
    "commission-bps": _opt_commission_bps,
    "cost-model": _opt_cost_model,
    "initial-capital": _opt_initial_capital,
    "benchmark": _opt_benchmark,
    "tickers": _opt_tickers,
    "universe-file": _opt_universe_file,
    "max-universe": _opt_max_universe,
    "min-price": _opt_min_price,
    "min-avg-dollar-volume": _opt_min_avg_dollar_volume,
    "adv-window": _opt_adv_window,
    "min-score": _opt_min_score,
    "refresh": _opt_refresh,
    "candidates": _opt_candidates,
    "regime-filter": _opt_regime_filter,
    "sector-neutral": _opt_sector_neutral,
    "earnings-blackout": _opt_earnings_blackout,
    "slippage-model": _opt_slippage_model,
    "half-spread-bps": _opt_half_spread_bps,
    "vol-impact-k": _opt_vol_impact_k,
    "no-gap-fills": _opt_no_gap_fills,
    "entry-order": _opt_entry_order,
    "entry-limit-bps": _opt_entry_limit_bps,
    "partial-exit": _opt_partial_exit,
    "price-adjustment": _opt_price_adjustment,
    "interval": _opt_interval,
    "csv": _opt_csv,
    "report": _opt_report,
    "open-report": _opt_open_report,
}


def backtest_options(mode: str, *names: str) -> OptionDecorator:
    """Return a decorator stacking the named shared backtest options in order.

    ``mode`` is ``"historical"`` or ``"rolling"``; a handful of options carry
    mode-specific help text/defaults so each command reproduces its exact
    ``--help`` output. Options render in the given ``names`` order (which the
    caller interleaves with its mode-specific ``click.option`` decorators).
    """

    def decorator(command: Any) -> Any:
        for name in reversed(names):
            command = _OPTION_BUILDERS[name](mode)(command)
        return command

    return decorator


def parse_ticker_list(tickers: str | None) -> tuple[str, ...] | None:
    """Split a comma-separated ``--tickers`` value into a tuple (or ``None``)."""
    if not tickers:
        return None
    return tuple(t.strip() for t in tickers.split(",") if t.strip())


def build_backtest_fetcher(
    ctx_obj: Any, *, price_adjustment: str, interval: str, refresh: bool = False
):
    """Resolve the shared price fetcher exactly as both commands do."""
    from screener.backtester.data import build_price_fetcher
    from screener.markets import get_price_fetcher

    return get_price_fetcher(
        ctx_obj,
        builder=build_price_fetcher,
        auto_adjust=price_adjustment == "full",
        interval=interval,
        refresh=refresh,
    )


def resolve_report_path(
    report_path: Path | None, output_csv: bool, prefix: str
) -> Path | None:
    """Return the tear-sheet path (a temp path unless CSV output is requested)."""
    if report_path is not None:
        return report_path
    if output_csv:
        return None
    from screener.reporting import temp_report_path

    return temp_report_path(prefix)


def write_tearsheet(
    result: Any, path: Path, *, title: str, extra_notes: Sequence[str]
) -> None:
    """Render the static HTML tear-sheet (thin wrapper for lazy import)."""
    from screener.backtester.tearsheet import render_tearsheet

    render_tearsheet(result, path, title=title, extra_notes=list(extra_notes))
