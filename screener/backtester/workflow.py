"""Request resolution beneath the backtest Click commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click
from pydantic import ValidationError

from screener import history
from screener.backtester.cli_common import (
    build_backtest_fetcher,
    build_slippage_model,
    parse_partial_exits,
    parse_rank_exit,
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
from screener.gate_options import gate_overrides
from screener.markets import get_market
from screener.universes import load_sp500_membership, load_universe_selection

if TYPE_CHECKING:
    from screener.strategies.spec import StrategyProfile


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
    rank_exit: str | None = None
    rank_universe_size: int = 50
    earnings_blackout_days: int | None = None
    fundamentals_provider: str | None = None
    fundamental_field_args: tuple[str, ...] = ()
    fundamental_lag_days: int | None = None
    compare_reinvestment: bool = False
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
    # Click source state is required because a typed default value cannot be
    # distinguished from an omitted option by inspecting ``adv_window``.
    adv_window_was_explicit: bool = False
    # ``--point-in-time`` is on by default, so a universe with no membership
    # history must fall back quietly instead of failing. Only a typed flag is
    # a hard requirement the run may refuse.
    point_in_time_was_explicit: bool = False
    # Percentile floor on ``setup_score`` (0-100). Defaulted because only the
    # rolling command declares ``--min-score``; historical has no candidate
    # layer to gate.
    min_score: float | None = None
    # Bypass cached bars. Only the rolling command declares ``--refresh``; a
    # screen has always had one, and a screen compared against a backtest that
    # served stale bars is not a comparison.
    refresh: bool = False
    # Print the ranked candidate set for the last bar and stop.
    candidates: bool = False


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
    avg_dollar_volume_window: int | None = None,
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
            avg_dollar_volume_window=(
                int(request.adv_window)
                if avg_dollar_volume_window is None
                else int(avg_dollar_volume_window)
            ),
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


def _effective_gates(request: BacktestRequest) -> StrategyProfile:
    """The strategy's declared candidate gates, with typed CLI flags winning.

    ``screener.screen_candidates`` loads the same profile for the screen.
    Loading it here as well is what stops the two paths drifting by config
    instead of by code (D13 in ``docs/plans/unify-screen-backtest.md``); without
    it a strategy is screened with its declared gates and backtested without
    them. A flag left at its option default is "not given", so the profile
    supplies the value; anything the user typed becomes an override, which is
    the precedence :func:`resolve_strategy_profile` documents.

    Gates the profile deliberately does not carry - the universe and venue
    fields in ``RUN_SCOPED_SIGNAL_PANEL_FIELDS`` - stay with the request.
    """
    from screener.strategies.spec import (
        ExpressionStrategySpec,
        resolve_strategy_profile,
        resolve_strategy_spec,
    )

    spec = (
        resolve_strategy_spec(request.strategy_name) if request.strategy_name else None
    )
    overrides = gate_overrides(
        min_price=request.min_price,
        min_avg_dollar_volume=request.min_avg_dollar_volume,
        adv_window=request.adv_window,
        adv_window_was_explicit=request.adv_window_was_explicit,
        regime_filter_args=request.regime_filter_args,
        earnings_blackout_days=request.earnings_blackout_days,
        sector_neutral=request.sector_neutral,
        min_score=request.min_score,
    )
    return resolve_strategy_profile(
        spec if isinstance(spec, ExpressionStrategySpec) else None,
        overrides,
        market=request.market,
    )


def resolve_rolling_gates(request: BacktestRequest) -> StrategyProfile:
    """The candidate gates the rolling engine applies, per-market floor included.

    :func:`_effective_gates` resolves the declared profile against the flags the
    user actually typed; ``resolve_min_filters`` then applies the venue floor,
    which is what turns a profile naming no liquidity gate into the market's own
    minimum (and still reads an explicit 0 as "disabled").

    Named rather than inlined so the screen's
    :func:`screener.screen_candidates.resolve_screen_gates` has something to be
    compared against: for one strategy and one market the two must agree, and
    ``tests/correctness`` asserts exactly that.
    """
    return _effective_gates(request)


def _resolve_rolling(request: BacktestRequest) -> BacktestRun:
    if request.output_csv and request.dashboard:
        raise click.UsageError("--csv and --dashboard cannot be used together.")
    validate_sizing(request.sizing_rule, request.stop_loss)
    if (
        request.earnings_blackout_days is not None
        and request.earnings_blackout_days < 0
    ):
        raise click.UsageError("--earnings-blackout must be >= 0.")
    rank_exit = parse_rank_exit(request.rank_exit)
    if rank_exit is not None and rank_exit[1] and request.interval != "1d":
        raise click.UsageError(
            "--rank-exit weekly/monthly count trading days and require "
            "--interval 1d; pass an explicit bar count for intraday runs."
        )

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
    # Local, because the default-on flag is downgraded for universes that carry
    # no membership history; the request keeps what the user asked for.
    point_in_time = bool(request.point_in_time)
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
                point_in_time=point_in_time,
                start=start_date,
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
            first_snapshot = min(start for _symbol, start, _end in membership_windows)
            if start_date < first_snapshot:
                # Nothing is eligible before the earliest snapshot, so the run
                # would otherwise report a silent zero-trade stretch that looks
                # like a strategy result rather than missing membership history.
                universe_note += (
                    f"; membership history starts {first_snapshot.isoformat()}, so no "
                    f"symbol is eligible from {start_date.isoformat()} until then"
                )
        if dynamic_universe_size is not None:
            universe_note += (
                f"; top {dynamic_universe_size} by prior {dynamic_universe_lookback}-bar "
                f"ADV, rebalanced {dynamic_universe_rebalance}; candidate base is an "
                "as-of-end snapshot and may retain survivorship bias"
            )
        if point_in_time:
            if membership_windows or dynamic_universe_size is not None:
                pass
            elif resolved_universe != "sp500":
                if request.point_in_time_was_explicit:
                    raise click.UsageError(
                        "--point-in-time requires snapshot history or the sp500 universe."
                    )
                point_in_time = False
                universe_note += (
                    "; survivorship bias: today's members applied to history "
                    f"({resolved_universe} has no membership history, so "
                    "point-in-time is inactive)"
                )
            else:
                # The sp500 selection could not read its revision history, so
                # fall back to the weaker "date added" filter: it dates only
                # today's members, which keeps post-as-of additions out but
                # cannot bring removed ex-members back.
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
                "(point-in-time membership is off)"
            )
    if (
        point_in_time
        and not membership_added
        and not membership_windows
        and dynamic_universe_size is None
    ):
        if request.point_in_time_was_explicit:
            raise click.UsageError(
                "--point-in-time requires an index universe; it cannot be used with "
                "--tickers or --universe-file."
            )
        # An explicit ticker list carries no membership history, so the default
        # simply does not apply. Failing here would break every --tickers run.
        point_in_time = False

    slippage_model = build_slippage_model(
        request.slippage_model,
        request.slippage_bps,
        request.half_spread_bps,
        request.vol_impact_k,
        spread_proxy=bool(request.spread_proxy),
    )
    gates = resolve_rolling_gates(request)
    config = _build_config(
        request,
        market=request.market,
        benchmark=benchmark,
        as_of=end_date,
        tickers=tickers,
        min_price=gates.min_price,
        min_avg_dollar_volume=gates.min_avg_dollar_volume,
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
        avg_dollar_volume_window=gates.avg_dollar_volume_window,
        regime_filter=gates.regime_filter,
        earnings_blackout_days=gates.earnings_blackout_days,
        fundamentals_provider=provider,
        fundamental_fields=fields,
        fundamental_lag_days=max(resolved_lag, 0),
        sector_neutral=gates.sector_neutral,
        min_score=gates.min_score,
        rank_exit_every=(rank_exit[0] if rank_exit is not None else None),
        rank_universe_size=int(request.rank_universe_size),
    )
    return BacktestRun(
        config=config,
        price_fetcher=build_backtest_fetcher(
            request.context_obj,
            price_adjustment=request.price_adjustment,
            interval=request.interval,
            refresh=request.refresh,
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
