"""Request resolution beneath the backtest Click commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import click
from pydantic import ValidationError

from screener import history
from screener.backtester.cli_common import (
    build_backtest_fetcher,
    build_slippage_model,
    parse_partial_exits,
    parse_ticker_list,
    referenced_fundamental_fields,
    resolve_min_filters,
    resolve_strategy_exprs,
    validate_sizing,
)
from screener.backtester.data import PriceFetcher
from screener.backtester.fundamentals import (
    FundamentalFetcher,
    build_fundamental_fetcher,
    fundamental_filing_lag_days,
)
from screener.backtester.models import BacktestConfig
from screener.markets import get_market
from screener.universes import load_sp500_membership, load_universe_selection


@dataclass(frozen=True)
class BacktestRequest:
    """Raw backtest options captured by either Click command."""

    mode: Literal["rolling", "historical"]
    context_obj: Any
    market: str
    hold: int
    top: int
    entry_expr: str | None
    exit_expr: str | None
    strategy_name: str | None
    stop_loss: float | None
    take_profit: float | None
    trailing_stop: float | None
    slippage_bps: float
    commission_bps: float
    cost_model: str
    initial_capital: float
    benchmark: str | None
    tickers: str | None
    universe_file: str | None
    max_universe: int
    min_price: float | None
    min_avg_dollar_volume: float | None
    adv_window: int
    slippage_model: str
    half_spread_bps: float
    vol_impact_k: float
    no_gap_fills: bool
    entry_order: str
    entry_limit_bps: float | None
    partial_exit_args: tuple[str, ...]
    price_adjustment: str
    interval: str
    output_csv: bool
    report_path: Path | None
    open_report: bool
    sizing_rule: str
    sizing_risk_pct: float
    sizing_position_pct: float
    sizing_atr_window: int
    sizing_atr_multiple: float
    sizing_vol_window: int
    intraday_only: bool
    start_arg: datetime | None = None
    end_arg: datetime | None = None
    years: int = 1
    universe: str | None = None
    universe_config: Path | None = None
    dynamic_base: str | None = None
    universe_size: int = 100
    universe_lookback: int = 60
    universe_rebalance: str = "monthly"
    no_universe_cache: bool = False
    point_in_time: bool = False
    spread_proxy: bool = False
    regime_filter_args: tuple[str, ...] = ()
    sector_neutral: bool = False
    rank_exit: int | None = None
    rank_universe_size: int = 50
    earnings_blackout_days: int | None = None
    fundamentals_provider: str | None = None
    fundamental_field_args: tuple[str, ...] = ()
    fundamental_lag_days: int | None = None
    dashboard: bool = False
    dashboard_port: int = 8765
    dashboard_dir: Path = Path(".screener/dashboards")
    as_of: date | datetime | None = None
    from_run: str | None = None
    run_age_days: int = 0
    reserve_multiple: int = 3
    no_reinvest: bool = False
    allow_reentry: bool = False
    max_reentries: int = 0
    market_was_explicit: bool = False
    top_was_explicit: bool = False


@dataclass(frozen=True)
class BacktestRun:
    """The resolved configuration and collaborators required to execute a run."""

    config: BacktestConfig
    price_fetcher: PriceFetcher
    fundamental_fetcher: FundamentalFetcher | None
    start_date: date | None
    end_date: date | None
    universe_note: str | None = None
    replay_note: str | None = None


def _build_config(
    request: BacktestRequest,
    *,
    market: str,
    benchmark: str,
    as_of: date,
    tickers: tuple[str, ...] | None,
    min_price: float | None,
    min_avg_dollar_volume: float | None,
    entry_expr: str,
    exit_expr: str | None,
    slippage_model: Any,
    partial_exits: tuple[tuple[float, float], ...],
    **extra: Any,
) -> BacktestConfig:
    try:
        return BacktestConfig(
            market=market,
            as_of=as_of,
            benchmark=benchmark,
            tickers=tickers,
            universe_file=request.universe_file,
            max_universe=int(request.max_universe),
            strategy_name=request.strategy_name,
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            interval=request.interval,
            price_adjustment=cast(
                Literal["full", "splits_only", "none"], request.price_adjustment
            ),
            intraday_only=bool(request.intraday_only),
            hold=int(request.hold),
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            trailing_stop=request.trailing_stop,
            slippage_bps=float(request.slippage_bps),
            commission_bps=float(request.commission_bps),
            slippage_model=slippage_model,
            cost_model=cast(Literal["flat", "india", "us_vested"], request.cost_model),
            gap_fills=not request.no_gap_fills,
            entry_order_type=cast(Literal["moo", "moc", "limit"], request.entry_order),
            entry_limit_bps=request.entry_limit_bps,
            partial_exits=partial_exits,
            top=int(request.top),
            initial_capital=float(request.initial_capital),
            min_price=min_price,
            min_avg_dollar_volume=min_avg_dollar_volume,
            avg_dollar_volume_window=int(request.adv_window),
            sizing_rule=request.sizing_rule,
            sizing_risk_pct=float(request.sizing_risk_pct),
            sizing_position_pct=float(request.sizing_position_pct),
            sizing_atr_window=int(request.sizing_atr_window),
            sizing_atr_multiple=float(request.sizing_atr_multiple),
            sizing_vol_window=int(request.sizing_vol_window),
            **extra,
        )
    except ValidationError as exc:
        raise click.UsageError(str(exc)) from exc


def _resolve_rolling(request: BacktestRequest) -> BacktestRun:
    if request.output_csv and request.dashboard:
        raise click.UsageError("--csv and --dashboard cannot be used together.")
    validate_sizing(request.sizing_rule, request.stop_loss)
    if (
        request.earnings_blackout_days is not None
        and request.earnings_blackout_days < 0
    ):
        raise click.UsageError("--earnings-blackout must be >= 0.")

    entry_expr, exit_expr = resolve_strategy_exprs(
        request.strategy_name, request.entry_expr, request.exit_expr
    )
    needed_fundamentals = referenced_fundamental_fields(entry_expr, exit_expr)
    provider = request.fundamentals_provider
    if needed_fundamentals and provider is None:
        provider = "fmp" if request.market == "us" else "openscreener"

    end_date = (
        request.end_arg.date()
        if isinstance(request.end_arg, datetime)
        else (request.end_arg or date.today())
    )
    start_date = (
        request.start_arg.date()
        if isinstance(request.start_arg, datetime)
        else (
            request.start_arg or (end_date - timedelta(days=365 * int(request.years)))
        )
    )
    market_meta = get_market(request.market)
    benchmark = request.benchmark or market_meta.benchmark
    resolved_lag = (
        int(request.fundamental_lag_days)
        if request.fundamental_lag_days is not None
        else fundamental_filing_lag_days(provider)
    )
    fields = tuple(dict.fromkeys(request.fundamental_field_args))
    if fields:
        fields = tuple(dict.fromkeys((*fields, *sorted(needed_fundamentals))))
    try:
        fundamental_fetcher = build_fundamental_fetcher(
            provider,
            market=request.market,
            fields=fields or None,
            lag_days=max(resolved_lag, 0),
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    tickers = None
    universe_note = None
    membership_added: tuple[tuple[str, date], ...] = ()
    membership_windows: tuple[tuple[str, date, date | None], ...] = ()
    dynamic_universe_size: int | None = None
    dynamic_universe_lookback = int(request.universe_lookback)
    dynamic_universe_rebalance = str(request.universe_rebalance)
    if request.tickers:
        tickers = parse_ticker_list(request.tickers)
    elif not request.universe_file:
        resolved_universe = str(
            request.universe or market_meta.default_universe
        ).lower()
        try:
            selection = load_universe_selection(
                resolved_universe,
                market=request.market,
                as_of=end_date,
                config_path=request.universe_config,
                use_cache=not request.no_universe_cache,
                dynamic_base=request.dynamic_base,
                dynamic_size=int(request.universe_size),
                dynamic_lookback=int(request.universe_lookback),
                dynamic_rebalance=str(request.universe_rebalance),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise click.UsageError(str(exc)) from exc
        tickers = selection.symbols
        membership_windows = selection.membership_windows
        dynamic_universe_size = selection.dynamic_size
        dynamic_universe_lookback = selection.dynamic_lookback
        dynamic_universe_rebalance = selection.dynamic_rebalance
        if request.benchmark is None and selection.benchmark:
            benchmark = selection.benchmark
        universe_note = f"{selection.name}: {len(selection.symbols)} candidate symbols from {selection.source}"
        if membership_windows:
            universe_note += f"; point-in-time snapshots ({len(membership_windows)} membership windows)"
        if dynamic_universe_size is not None:
            universe_note += (
                f"; top {dynamic_universe_size} by prior {dynamic_universe_lookback}-bar "
                f"ADV, rebalanced {dynamic_universe_rebalance}; candidate base is an "
                "as-of-end snapshot and may retain survivorship bias"
            )
        if request.point_in_time:
            if membership_windows or dynamic_universe_size is not None:
                pass
            elif resolved_universe != "sp500":
                raise click.UsageError(
                    "--point-in-time requires snapshot history or the sp500 universe."
                )
            else:
                added_by_symbol = load_sp500_membership(
                    as_of=end_date, use_cache=not request.no_universe_cache
                )
                membership_added = tuple(
                    (symbol, added)
                    for symbol, added in added_by_symbol.items()
                    if added is not None
                )
                universe_note += (
                    f"; point-in-time entries via 'date added' "
                    f"({len(membership_added)} dated symbols; removed ex-members not reconstructed)"
                )
        elif not membership_windows and dynamic_universe_size is None:
            universe_note += (
                "; survivorship bias: today's members applied to history "
                "(pass --point-in-time to filter by 'date added')"
            )
    if (
        request.point_in_time
        and not membership_added
        and not membership_windows
        and dynamic_universe_size is None
    ):
        raise click.UsageError(
            "--point-in-time requires an index universe; it cannot be used with "
            "--tickers or --universe-file."
        )

    slippage_model = build_slippage_model(
        request.slippage_model,
        request.slippage_bps,
        request.half_spread_bps,
        request.vol_impact_k,
        spread_proxy=bool(request.spread_proxy),
    )
    min_price, min_adv = resolve_min_filters(
        request.market, request.min_price, request.min_avg_dollar_volume
    )
    config = _build_config(
        request,
        market=request.market,
        benchmark=benchmark,
        as_of=end_date,
        tickers=tickers,
        min_price=min_price,
        min_avg_dollar_volume=min_adv,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        slippage_model=slippage_model,
        partial_exits=parse_partial_exits(request.partial_exit_args),
        membership_added=membership_added,
        membership_windows=membership_windows,
        dynamic_universe_size=dynamic_universe_size,
        dynamic_universe_lookback=dynamic_universe_lookback,
        dynamic_universe_rebalance=dynamic_universe_rebalance,
        spread_proxy=bool(request.spread_proxy),
        regime_filter=tuple(dict.fromkeys(request.regime_filter_args)),
        earnings_blackout_days=request.earnings_blackout_days,
        fundamentals_provider=provider,
        fundamental_fields=fields,
        fundamental_lag_days=max(resolved_lag, 0),
        sector_neutral=bool(request.sector_neutral),
        rank_exit_every=(
            int(request.rank_exit) if request.rank_exit is not None else None
        ),
        rank_universe_size=int(request.rank_universe_size),
    )
    return BacktestRun(
        config=config,
        price_fetcher=build_backtest_fetcher(
            request.context_obj,
            price_adjustment=request.price_adjustment,
            interval=request.interval,
        ),
        fundamental_fetcher=fundamental_fetcher,
        start_date=start_date,
        end_date=end_date,
        universe_note=universe_note,
    )


def _resolve_historical(request: BacktestRequest) -> BacktestRun:
    validate_sizing(request.sizing_rule, request.stop_loss)
    market = request.market
    as_of = request.as_of
    tickers = None
    replay_note = None
    entry_expr: str | None = request.entry_expr
    if request.from_run:
        if request.tickers or request.universe_file:
            raise click.UsageError(
                "--from-run supplies the universe; drop --tickers/--universe-file."
            )
        if as_of is not None:
            raise click.UsageError(
                "--from-run derives --as-of from the stored run; drop --as-of."
            )
        try:
            snapshot = history.resolve_replay_run(
                request.from_run, min_age_days=request.run_age_days
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if request.market_was_explicit and market != snapshot.market:
            raise click.UsageError(
                f"screen run #{snapshot.run_id} is for market {snapshot.market!r}, not {market!r}."
            )
        market = snapshot.market
        as_of = snapshot.run_date
        if not request.strategy_name and not request.entry_expr:
            entry_expr = "close > 0"
        if not request.top_was_explicit:
            request = dataclass_replace(request, top=len(snapshot.tickers))
        tickers = tuple(snapshot.tickers)
        replay_note = (
            f"Replaying screen run #{snapshot.run_id} "
            f"({snapshot.market}/{snapshot.criteria} @ {snapshot.run_date.isoformat()}, "
            f"{len(snapshot.tickers)} tickers)"
        )
    elif as_of is None:
        raise click.UsageError("--as-of is required unless --from-run is used.")

    entry_expr, exit_expr = resolve_strategy_exprs(
        request.strategy_name, entry_expr, request.exit_expr
    )
    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    assert as_of_date is not None
    if tickers is None and request.tickers:
        tickers = parse_ticker_list(request.tickers)
    if not tickers and not request.universe_file:
        raise click.UsageError(
            "No universe provided: pass --tickers, --universe-file, or --from-run. "
            "The TradingView current-screener fallback was removed because it injects survivorship bias."
        )
    min_price, min_adv = resolve_min_filters(
        market, request.min_price, request.min_avg_dollar_volume
    )
    slippage_model = build_slippage_model(
        request.slippage_model,
        request.slippage_bps,
        request.half_spread_bps,
        request.vol_impact_k,
    )
    config = _build_config(
        request,
        market=market,
        benchmark=request.benchmark or get_market(market).benchmark,
        as_of=as_of_date,
        tickers=tickers,
        min_price=min_price,
        min_avg_dollar_volume=min_adv,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        slippage_model=slippage_model,
        partial_exits=parse_partial_exits(request.partial_exit_args),
        reserve_multiple=int(request.reserve_multiple),
        reinvest=not request.no_reinvest,
        allow_reentry=bool(request.allow_reentry),
        max_reentries=int(request.max_reentries),
    )
    return BacktestRun(
        config=config,
        price_fetcher=build_backtest_fetcher(
            request.context_obj,
            price_adjustment=request.price_adjustment,
            interval=request.interval,
        ),
        fundamental_fetcher=None,
        start_date=as_of_date,
        end_date=as_of_date,
        replay_note=replay_note,
    )


def dataclass_replace(request: BacktestRequest, **changes: Any) -> BacktestRequest:
    """Avoid exposing mutable request state while replay resolution adjusts defaults."""
    from dataclasses import replace

    return replace(request, **changes)


def resolve_backtest_run(request: BacktestRequest) -> BacktestRun:
    """Resolve CLI-independent policy and construct the engine collaborators."""
    if request.mode == "rolling":
        return _resolve_rolling(request)
    return _resolve_historical(request)


__all__ = ["BacktestRequest", "BacktestRun", "resolve_backtest_run"]
