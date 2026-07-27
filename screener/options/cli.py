"""Click commands for building and inspecting options panels."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import cast

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.cache import append_panel_snapshot
from screener.earnings_backtest.metrics import compute_backtest_summary
from screener.markets import get_price_fetcher, market_option
from screener.options.bt_models import (
    FillModel,
    MarginModel,
    OptionPositionTrade,
    OptionsBacktestConfig,
    Settlement,
)
from screener.options.criteria import OPTIONS_CRITERIA, run_options_criterion
from screener.options.models import OptionsMarket
from screener.options.panels import build_india_panel, show_symbol, snapshot_us
from screener.options.participant import build_participant_panel
from screener.options.position_backtest import run_options_position_backtest
from screener.options.regime import (
    build_india_vix_panel,
    build_us_regime_panel,
    fetch_india_vix_live,
)
from screener.options.structures import available_structures


def _as_date(value: datetime | date | None, default: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value or default


def _ticker_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or None


@click.group(name="options")
def options() -> None:
    """Build, snapshot, and inspect normalized options data."""


@options.command(name="build-panel")
@market_option(
    default="india",
    choices=("india",),
    help="Market whose historical options panel should be built.",
    show_default=True,
)
@click.option(
    "--start",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First archive date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last archive date (default: today).",
)
@click.option(
    "--tickers",
    default=None,
    help="Optional comma-separated NSE symbols; default builds every optionable symbol.",
)
@click.option("--refresh", is_flag=True, help="Re-download cached archive dates.")
def build_panel(
    market: str,
    start: datetime,
    end: datetime | None,
    tickers: str | None,
    refresh: bool,
) -> None:
    """Backfill the exact historical NSE UDiff options panel."""
    del market  # Click choice intentionally leaves room for future archives.
    start_date = start.date()
    end_date = _as_date(end, date.today())
    errors: list[tuple[date, Exception]] = []
    loaded: list[tuple[date, int]] = []
    try:
        panel = build_india_panel(
            start_date,
            end_date,
            symbols=_ticker_set(tickers),
            refresh=refresh,
            on_progress=lambda day, count: loaded.append((day, count)),
            on_error=lambda day, exc: errors.append((day, exc)),
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if panel.empty:
        click.echo(
            f"No India options rows available for {start_date} through {end_date}."
        )
        return
    dates = pd.to_datetime(panel["as_of"], errors="coerce")
    in_window = panel[(dates.dt.date >= start_date) & (dates.dt.date <= end_date)]
    if tickers:
        wanted = {value.upper() for value in _ticker_set(tickers) or set()}
        in_window = in_window[in_window["SYMBOL"].astype(str).str.upper().isin(wanted)]
    symbols = int(in_window["SYMBOL"].nunique()) if not in_window.empty else 0
    source = (
        ", ".join(sorted(in_window["source"].dropna().astype(str).unique()))
        if "source" in in_window.columns and not in_window.empty
        else "n/a"
    )
    click.echo(
        f"India options panel: {len(in_window)} rows, {symbols} symbols, "
        f"{len(loaded)} archive days ({start_date} through {end_date}); source={source}."
    )
    if errors:
        click.echo(
            f"Skipped {len(errors)} unavailable trading date(s); rerun with "
            "--log-level INFO for provider details.",
            err=True,
        )


@options.command(name="signals")
@market_option(
    default="us",
    choices=("us", "india"),
    help="Options panel market.",
    show_default=True,
)
@click.option(
    "-c",
    "--criterion",
    type=click.Choice(list(OPTIONS_CRITERIA)),
    required=True,
    help="Panel-backed options signal to screen.",
)
@click.option("-n", "--limit", type=int, default=50, show_default=True)
@click.option("--csv", "output_csv", is_flag=True, help="Emit raw CSV.")
def signals(market: str, criterion: str, limit: int, output_csv: bool) -> None:
    """Screen the accumulated options panel for a single signal.

    Point-in-time evaluation of one registered options criterion (formerly
    exposed as ``screen -c <criterion>``). Run ``options snapshot`` /
    ``options build-panel`` first to accumulate the panel history.
    """
    run_options_criterion(
        criterion, market=market, limit=int(limit), output_csv=output_csv
    )


@options.command(name="show")
@market_option(
    default="india",
    choices=("us", "india"),
    help="Options panel market.",
    show_default=True,
)
@click.option("--symbol", required=True, help="Underlying symbol to inspect.")
@click.option("--days", type=int, default=20, show_default=True)
@click.option("--csv", "output_csv", is_flag=True, help="Emit raw CSV.")
def show(market: str, symbol: str, days: int, output_csv: bool) -> None:
    """Show accumulated daily metrics for one underlying."""
    rows = show_symbol(market, symbol)
    if rows.empty:
        click.echo(
            f"No {market.upper()} options panel history for {symbol.strip().upper()}."
        )
        return
    if days > 0:
        rows = rows.tail(days)
    if output_csv:
        click.echo(rows.to_csv(index=False), nl=False)
        return
    preferred = [
        "as_of",
        "SYMBOL",
        "source",
        "spot",
        "front_expiry",
        "pcr",
        "pcr_volume",
        "max_pain_strike",
        "median_iv",
        "iv_rank",
        "implied_move_pct",
        "contract_count",
        "history_days",
    ]
    columns = [column for column in preferred if column in rows.columns]
    click.echo(rows[columns].to_string(index=False))
    latest = rows.iloc[-1]
    click.echo(
        f"As of {pd.Timestamp(latest['as_of']).date()} · "
        f"source={latest.get('source', 'unknown')} · "
        f"coverage={int(latest.get('contract_count', 0))} contracts / "
        f"{int(latest.get('history_days', len(rows)))} panel day(s)."
    )


@options.command(name="snapshot")
@market_option(
    default="us",
    choices=("us",),
    help="Market to snapshot.",
    show_default=True,
)
@click.option("--tickers", default=None, help="Comma-separated US symbols.")
@click.option(
    "--universe-size",
    type=click.IntRange(min=1),
    default=None,
    help="Snapshot the first N current S&P 500 symbols instead of --tickers.",
)
@click.option(
    "--workers", type=click.IntRange(min=1, max=8), default=4, show_default=True
)
@click.option("--refresh", is_flag=True, help="Bypass intraday provider caches.")
def snapshot(
    market: str,
    tickers: str | None,
    universe_size: int | None,
    workers: int,
    refresh: bool,
) -> None:
    """Accumulate delayed CBOE chains with yfinance fallback."""
    del market
    if bool(tickers) == bool(universe_size):
        raise click.UsageError("Pass exactly one of --tickers or --universe-size.")
    if tickers:
        symbols = sorted(_ticker_set(tickers) or set())
    else:
        from screener.universes import load_current_universe

        symbols = list(
            load_current_universe("sp500").symbols[: int(universe_size or 0)]
        )
    result = snapshot_us(
        symbols,
        refresh=refresh,
        max_workers=workers,
    )
    for chain in result.chains:
        click.echo(
            f"{chain.underlying}: as_of={chain.as_of.isoformat()} "
            f"source={chain.source} contracts={len(chain.contracts)} "
            f"expiries={len(chain.expiries)}"
        )
    click.echo(
        f"US options snapshot: {len(result.chains)}/{result.requested} symbols "
        "appended to options_metrics_us."
    )
    if result.missing:
        click.echo(
            f"Unavailable: {', '.join(result.missing)} (providers degraded cleanly).",
            err=True,
        )


@options.command(name="record")
@market_option(
    default="us",
    choices=("us", "india"),
    help="Market whose watchlist chains are snapshotted.",
    show_default=True,
)
@click.option(
    "--every",
    "every",
    default=None,
    help="Snapshot cadence (e.g. 15m) for a session-bounded loop; "
    "omit for a single pass.",
)
@click.option(
    "--once",
    is_flag=True,
    default=False,
    help="Force a single pass even when --every is given (what a 15m cron calls).",
)
@click.option(
    "--watchlist",
    default=None,
    help="Comma-separated underlyings to snapshot (default: index options).",
)
@click.option(
    "--watchlist-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File of underlyings (comma/newline separated; # comments allowed).",
)
@click.option(
    "--max-underlyings",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Cap the number of underlyings (0 = no cap).",
)
@click.option("--refresh", is_flag=True, help="Bypass provider snapshot caches.")
@click.pass_context
def record(
    ctx: click.Context,
    market: str,
    every: str | None,
    once: bool,
    watchlist: str | None,
    watchlist_file: Path | None,
    max_underlyings: int,
    refresh: bool,
) -> None:
    """Snapshot delayed/live option chains into the contract store.

    Forward-capture: each pass appends every watchlist underlying's chain as a
    timestamped snapshot (idempotent). US uses delayed CBOE (yfinance
    fallback); India uses the NSE live chain API. Run one pass per invocation
    from a 15-minute cron (``--once``), or a session-bounded loop from a single
    session-open cron (``--every 15m``).
    """
    from screener.cache import parse_ttl
    from screener.options import recorder
    from screener.options.contract_store import store_health

    provider = None
    root = None
    obj = ctx.obj if ctx is not None else None
    if isinstance(obj, dict):
        provider = obj.get("provider")
        root = obj.get("root")
    if provider is None:
        provider = recorder.default_provider(market)  # type: ignore[arg-type]

    symbols = recorder.resolve_watchlist(
        market,
        watchlist=watchlist,
        watchlist_file=watchlist_file,
        max_underlyings=max_underlyings,
    )
    if not symbols:
        raise click.UsageError("no underlyings resolved for the watchlist")

    if every and not once:
        seconds = parse_ttl(every)
        if not seconds or seconds <= 0:
            raise click.UsageError(f"invalid --every interval: {every!r}")
        recorder.record_loop(
            market,  # type: ignore[arg-type]
            symbols,
            provider=provider,
            every_seconds=float(seconds),
            root=root,
            refresh=refresh,
            echo=click.echo,
        )
    else:
        result = recorder.run_pass(
            market,  # type: ignore[arg-type]
            symbols,
            provider=provider,
            root=root,
            refresh=refresh,
        )
        for symbol, count in result.recorded:
            click.echo(f"{symbol}: {count} contracts snapshotted")
        click.echo(
            f"{market.upper()} options record: "
            f"{len(result.recorded)}/{len(symbols)} underlyings, "
            f"{result.contract_count} contracts appended."
        )
        if result.missing:
            click.echo(
                f"Unavailable: {', '.join(result.missing)} "
                "(providers degraded cleanly).",
                err=True,
            )
    click.echo(store_health(market, root=root).summary(), err=True)


@options.command(name="participants")
@click.option(
    "--start",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First participant archive date (default: --end/today).",
)
@click.option(
    "--end",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last participant archive date (default: today).",
)
@click.option("--refresh", is_flag=True, help="Bypass participant archive cache.")
@click.option("--csv", "output_csv", is_flag=True, help="Emit raw CSV.")
def participants(
    start: datetime | None,
    end: datetime | None,
    refresh: bool,
    output_csv: bool,
) -> None:
    """Backfill/show NSE Client, DII, FII, and Pro derivatives positioning."""
    end_date = _as_date(end, date.today())
    start_date = _as_date(start, end_date)
    try:
        panel = build_participant_panel(start_date, end_date, refresh=refresh)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if panel.empty:
        click.echo(f"No participant OI rows for {start_date} through {end_date}.")
        return
    dates = pd.to_datetime(panel["as_of"], errors="coerce")
    rows = panel[(dates.dt.date >= start_date) & (dates.dt.date <= end_date)].copy()
    if rows.empty:
        click.echo(f"No participant OI rows for {start_date} through {end_date}.")
        return
    if output_csv:
        click.echo(rows.to_csv(index=False), nl=False)
        return
    latest_day = pd.to_datetime(rows["as_of"]).max()
    latest = rows[pd.to_datetime(rows["as_of"]) == latest_day]
    columns = [
        "as_of",
        "participant",
        "index_futures_net",
        "stock_futures_net",
        "index_call_net",
        "index_put_net",
        "total_net",
        "source",
    ]
    click.echo(
        latest[[column for column in columns if column in latest]].to_string(
            index=False
        )
    )
    click.echo(
        f"Participant OI as of {latest_day.date()} · source=nse_participant_oi · "
        f"coverage={len(latest)} participant classes."
    )


@options.command(name="regime")
@market_option(
    default="india",
    choices=("us", "india"),
    help="Market-level options/volatility regime panel.",
    show_default=True,
)
@click.option(
    "--start",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date (default: --end/today).",
)
@click.option(
    "--end",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last date (default: today).",
)
@click.option("--refresh", is_flag=True, help="Bypass provider caches.")
@click.option("--csv", "output_csv", is_flag=True, help="Emit raw CSV.")
def regime(
    market: str,
    start: datetime | None,
    end: datetime | None,
    refresh: bool,
    output_csv: bool,
) -> None:
    """Build India VIX or US CBOE PCR + VIX/VIX3M regime history."""
    end_date = _as_date(end, date.today())
    start_date = _as_date(start, end_date)
    try:
        if market == "india":
            panel = build_india_vix_panel(start_date, end_date, refresh=refresh)
            if end_date >= date.today():
                live = fetch_india_vix_live(as_of=date.today(), refresh=refresh)
                if not live.empty:
                    panel = append_panel_snapshot(
                        "india_vix", live, dedupe_keys=["as_of"]
                    )
        else:
            panel = build_us_regime_panel(start_date, end_date, refresh=refresh)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if panel.empty:
        click.echo(
            f"No {market.upper()} options regime rows for {start_date} through {end_date}."
        )
        return
    dates = pd.to_datetime(panel["as_of"], errors="coerce")
    rows = panel[(dates.dt.date >= start_date) & (dates.dt.date <= end_date)].copy()
    if output_csv:
        click.echo(rows.to_csv(index=False), nl=False)
        return
    click.echo(rows.tail(20).to_string(index=False))
    sources = ", ".join(
        sorted(rows.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())
    )
    click.echo(
        f"{market.upper()} regime coverage: {len(rows)} day(s), "
        f"{start_date} through {end_date}; source={sources or 'unknown'}."
    )


@options.command(name="backtest")
@market_option(
    default="india",
    choices=("india",),
    help="Options market (India NSE UDiff only for now).",
    show_default=True,
)
@click.option(
    "--tickers",
    required=True,
    help="Comma-separated NSE underlyings (e.g. RELIANCE,TCS).",
)
@click.option(
    "--start",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last date (YYYY-MM-DD).",
)
@click.option(
    "--structure",
    type=click.Choice(available_structures()),
    default="long_call",
    show_default=True,
    help="Option structure to simulate.",
)
@click.option(
    "--strike",
    "strike_rule",
    default="atm",
    show_default=True,
    help="Strike rule: atm | moneyness:<±pct> | delta:<abs>.",
)
@click.option(
    "--expiry",
    "expiry_rule",
    default="front",
    show_default=True,
    help="Expiry rule: front | next | dte:<n>.",
)
@click.option(
    "--width-pct",
    type=float,
    default=0.05,
    show_default=True,
    help="Wing/far-leg width as a fraction of spot (spreads/strangles).",
)
@click.option("--lots", type=int, default=1, show_default=True)
@click.option(
    "--entry",
    "entry_expr",
    default="true",
    show_default=True,
    help=(
        "Pine-like entry expression on underlying bars (signal at day-D close; "
        "fill at day-D+1 chain EOD). Use 'true' for unconditional."
    ),
)
@click.option(
    "--exit-expr",
    default=None,
    help="Pine-like exit expression evaluated on underlying bars.",
)
@click.option(
    "--screen-criterion",
    default=None,
    help=(
        "Thin alias for a panel-column entry filter "
        "(unusual_options, high_iv_rank, low_iv_rank, bullish_oi_buildup). "
        "Does not replay the full screening pipeline."
    ),
)
@click.option(
    "--target-pct",
    type=float,
    default=None,
    help="Take profit as premium-relative percent (e.g. 25).",
)
@click.option(
    "--stop-pct",
    type=float,
    default=None,
    help="Stop loss as premium-relative percent (e.g. 50).",
)
@click.option(
    "--exit-dte",
    type=int,
    default=1,
    show_default=True,
    help="Force exit when days-to-expiry <= this value.",
)
@click.option(
    "--max-hold",
    type=int,
    default=None,
    help="Max holding sessions before a time exit.",
)
@click.option(
    "--slippage-pct",
    type=float,
    default=0.0,
    show_default=True,
    help="Per-leg slippage as a fraction of premium (side-aware).",
)
@click.option(
    "--commission-per-order",
    type=float,
    default=0.0,
    show_default=True,
    help="Flat commission per leg per side (rupees).",
)
@click.option(
    "--fill-model",
    type=click.Choice(["legacy", "mid", "cross"]),
    default="legacy",
    show_default=True,
    help="Fill pricing: legacy (mid/last+settle), mid, or bid/ask cross.",
)
@click.option(
    "--slippage-bps",
    type=float,
    default=0.0,
    show_default=True,
    help="Extra fill slippage in basis points (mid/cross models).",
)
@click.option(
    "--slippage-ticks",
    type=float,
    default=0.0,
    show_default=True,
    help="Extra fill slippage in whole ticks (mid/cross models).",
)
@click.option(
    "--tick-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Tick size for --slippage-ticks.",
)
@click.option(
    "--illiquid-spread-pct",
    type=float,
    default=0.0,
    show_default=True,
    help="Proxy spread (fraction of mark) for legs lacking quotes.",
)
@click.option(
    "--margin-model",
    type=click.Choice(["none", "span", "regt"]),
    default="none",
    show_default=True,
    help="Short-option margin: none, SPAN-like (India), or Reg-T (US).",
)
@click.option(
    "--margin-cap-pct",
    type=float,
    default=None,
    help="Skip entries exceeding this fraction of initial capital in margin.",
)
@click.option(
    "--settlement",
    type=click.Choice(["intrinsic", "settle"]),
    default="intrinsic",
    show_default=True,
    help="Expiry mark: intrinsic vs official per-contract settlement price.",
)
@click.option(
    "--physical-settlement",
    is_flag=True,
    help="Record physical (stock) rather than cash (index) expiry settlement.",
)
@click.option(
    "--roll-dte",
    type=int,
    default=None,
    help="Roll (exit + re-enter) when days-to-expiry drops to this value.",
)
@click.option(
    "--roll-delta",
    type=float,
    default=None,
    help="Roll when the position's |net delta| reaches this value.",
)
@click.option(
    "--roll-expiry",
    "roll_expiry_rule",
    default="next",
    show_default=True,
    help="Expiry rule for the re-entered structure when rolling.",
)
@click.option("--min-oi", type=float, default=0.0, show_default=True)
@click.option("--min-volume", type=float, default=0.0, show_default=True)
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
@click.option("--refresh", is_flag=True, help="Re-download cached bhavcopy days.")
@click.pass_context
def backtest(
    ctx: click.Context,
    market: str,
    tickers: str,
    start: datetime,
    end: datetime,
    structure: str,
    strike_rule: str,
    expiry_rule: str,
    width_pct: float,
    lots: int,
    entry_expr: str,
    exit_expr: str | None,
    screen_criterion: str | None,
    target_pct: float | None,
    stop_pct: float | None,
    exit_dte: int,
    max_hold: int | None,
    slippage_pct: float,
    commission_per_order: float,
    fill_model: str,
    slippage_bps: float,
    slippage_ticks: float,
    tick_size: float,
    illiquid_spread_pct: float,
    margin_model: str,
    margin_cap_pct: float | None,
    settlement: str,
    physical_settlement: bool,
    roll_dte: int | None,
    roll_delta: float | None,
    roll_expiry_rule: str,
    min_oi: float,
    min_volume: float,
    output_csv: bool,
    refresh: bool,
) -> None:
    """Backtest multi-leg option structures on NSE UDiff EOD chains.

    Entry signals fire on the underlying day-D close; positions open at the
    next session's chain EOD prices (strictly causal). P&L uses observed
    premiums; greeks are only used for delta strike selection.
    """
    if market != "india":
        raise click.UsageError("options backtest currently supports only -m india.")
    ticker_list = tuple(
        value.strip().upper() for value in tickers.split(",") if value.strip()
    )
    if not ticker_list:
        raise click.UsageError("--tickers must list at least one symbol.")

    # Optional injected seams via Click context obj (tests):
    # obj may be a price fetcher, or a dict with chain_loader / price_fetcher.
    chain_loader = None
    price_fetcher = None
    obj = ctx.obj if ctx is not None else None
    if isinstance(obj, dict):
        chain_loader = obj.get("chain_loader")
        price_fetcher = obj.get("price_fetcher")
    else:
        price_fetcher = get_price_fetcher(obj)

    try:
        cfg = OptionsBacktestConfig(
            market="india",
            tickers=ticker_list,
            start=start.date(),
            end=end.date(),
            structure=structure,
            strike_rule=strike_rule,
            expiry_rule=expiry_rule,
            width_pct=float(width_pct),
            lots=int(lots),
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            screen_criterion=screen_criterion,
            target_pct=target_pct,
            stop_pct=stop_pct,
            exit_dte=int(exit_dte),
            max_hold=max_hold,
            slippage_pct=float(slippage_pct),
            commission_per_order=float(commission_per_order),
            fill_model=cast(FillModel, fill_model),
            slippage_bps=float(slippage_bps),
            slippage_ticks=float(slippage_ticks),
            tick_size=float(tick_size),
            illiquid_spread_pct=float(illiquid_spread_pct),
            margin_model=cast(MarginModel, margin_model),
            margin_cap_pct=(
                float(margin_cap_pct) if margin_cap_pct is not None else None
            ),
            settlement=cast(Settlement, settlement),
            physical_settlement=bool(physical_settlement),
            roll_dte=roll_dte,
            roll_delta=roll_delta,
            roll_expiry_rule=roll_expiry_rule,
            min_oi=float(min_oi),
            min_volume=float(min_volume),
            refresh=bool(refresh),
        )
    except Exception as exc:  # noqa: BLE001 - surface config errors cleanly
        raise click.UsageError(str(exc)) from exc

    try:
        result = run_options_position_backtest(
            cfg,
            chain_loader=chain_loader,
            price_fetcher=price_fetcher,
        )
    except (ValueError, KeyError) as exc:
        raise click.UsageError(str(exc)) from exc

    if output_csv:
        _print_options_csv(result.trades)
        return

    console = Console()
    if result.warnings:
        for warning in result.warnings[:20]:
            console.print(f"[yellow]{warning}[/yellow]")
        if len(result.warnings) > 20:
            console.print(
                f"[yellow]…and {len(result.warnings) - 20} more warnings[/yellow]"
            )

    summary = compute_backtest_summary(result.trades, strategy=structure)
    _print_options_summary(console, summary)
    if margin_model != "none" and result.peak_margin > 0:
        console.print(
            f"[cyan]Peak margin used: {result.peak_margin:,.0f} "
            f"({result.peak_margin_utilization * 100:.1f}% of capital)[/cyan]"
        )
    if result.trades:
        _print_options_ledger(console, result.trades)
    else:
        console.print(
            "[yellow]No option trades taken for the given parameters.[/yellow]"
        )


@options.command(name="intraday-backtest")
@click.option("--tickers", required=True, help="Comma-separated underlyings.")
@click.option("--start", "start_str", required=True, help="Session start (YYYY-MM-DD).")
@click.option("--end", "end_str", required=True, help="Session end (YYYY-MM-DD).")
@market_option(
    default="us",
    choices=("us", "india"),
    help="Options market.",
    show_default=True,
)
@click.option("--structure", default="long_call", show_default=True)
@click.option("--strike", "strike_rule", default="atm", show_default=True)
@click.option("--expiry", "expiry_rule", default="front", show_default=True)
@click.option(
    "--entry-time", default=None, help="Enter at first snapshot ≥ HH:MM (market-local)."
)
@click.option(
    "--exit-time", default=None, help="Flatten at/after HH:MM (market-local)."
)
@click.option("--target-pct", type=float, default=None)
@click.option("--stop-pct", type=float, default=None)
@click.option("--lots", type=int, default=1, show_default=True)
@click.option(
    "--fill-model",
    type=click.Choice(["legacy", "mid", "cross"]),
    default="legacy",
    show_default=True,
)
@click.option("--slippage-bps", type=float, default=0.0, show_default=True)
@click.option(
    "--margin-model",
    type=click.Choice(["none", "span", "regt"]),
    default="none",
    show_default=True,
)
@click.option(
    "--equity-hedge",
    type=float,
    default=0.0,
    show_default=True,
    help="Signed underlying units held alongside each option position (Phase 4.3).",
)
@click.pass_context
def intraday_backtest(
    ctx: click.Context,
    tickers: str,
    start_str: str,
    end_str: str,
    market: str,
    structure: str,
    strike_rule: str,
    expiry_rule: str,
    entry_time: str | None,
    exit_time: str | None,
    target_pct: float | None,
    stop_pct: float | None,
    lots: int,
    fill_model: str,
    slippage_bps: float,
    margin_model: str,
    equity_hedge: float,
) -> None:
    """Backtest an option structure over intraday contract-store snapshots.

    Walks recorded intraday chains point-in-time (entry/mark/exit at snapshot
    timestamps); positions flatten at each session's last snapshot.
    """
    from datetime import time as _time

    from screener.options.intraday_backtest import (
        IntradayOptionsBacktestConfig,
        run_intraday_options_backtest,
    )

    def _hhmm(value: str | None) -> _time | None:
        if not value:
            return None
        hh, mm = value.split(":")
        return _time(int(hh), int(mm))

    cfg = IntradayOptionsBacktestConfig(
        tickers=tuple(t.strip() for t in tickers.split(",") if t.strip()),
        start=date.fromisoformat(start_str),
        end=date.fromisoformat(end_str),
        market=cast(OptionsMarket, market),
        structure=structure,
        strike_rule=strike_rule,
        expiry_rule=expiry_rule,
        entry_time=_hhmm(entry_time),
        exit_time=_hhmm(exit_time),
        target_pct=target_pct,
        stop_pct=stop_pct,
        lots=lots,
        fill_model=cast(FillModel, fill_model),
        slippage_bps=float(slippage_bps),
        margin_model=cast(MarginModel, margin_model),
        equity_hedge_qty=float(equity_hedge),
    )

    obj = ctx.obj if ctx is not None else None
    provider = obj.get("provider") if isinstance(obj, dict) else None
    result = run_intraday_options_backtest(cfg, provider)

    console = Console()
    for warning in result.warnings[:20]:
        console.print(f"[yellow]{warning}[/yellow]")
    summary = compute_backtest_summary(result.trades, strategy=structure)
    _print_options_summary(console, summary)
    if margin_model != "none" and result.peak_margin > 0:
        console.print(f"[cyan]Peak margin used: {result.peak_margin:,.0f}[/cyan]")
    if result.trades:
        _print_options_ledger(console, result.trades)
    else:
        console.print(
            "[yellow]No option trades taken for the given parameters.[/yellow]"
        )


def _print_options_summary(console: Console, summary: dict) -> None:
    table = Table(
        title="Options Position Backtest Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    labels = {
        "total_events": "Trades Scanned",
        "trades_taken": "Trades Taken",
        "strategy": "Structure",
        "win_rate": "Win Rate (%)",
        "avg_return_pct": "Avg Return (%)",
        "median_return_pct": "Median Return (%)",
        "total_return_pct": "Cumulative Return (%)",
        "max_winner_pct": "Best Trade (%)",
        "max_loser_pct": "Worst Trade (%)",
        "profit_factor": "Profit Factor",
        "avg_holding_days": "Avg Hold (days)",
        "sharpe_approx": "Sharpe (approx)",
    }
    for key, label in labels.items():
        val = summary.get(key, "")
        if isinstance(val, float) and key == "profit_factor" and val == float("inf"):
            val = "inf"
        table.add_row(label, str(val))
    console.print(table)


def _print_options_ledger(console: Console, trades: list[OptionPositionTrade]) -> None:
    table = Table(title="Trade Ledger", show_header=True, header_style="bold cyan")
    for col in (
        "symbol",
        "structure",
        "entry",
        "exit",
        "pnl",
        "return%",
        "reason",
    ):
        table.add_column(col)
    for trade in trades:
        table.add_row(
            trade.symbol,
            trade.structure,
            trade.entry_date.isoformat(),
            trade.exit_date.isoformat(),
            f"{trade.pnl:.2f}",
            f"{trade.return_pct:.2f}",
            trade.exit_reason,
        )
    console.print(table)


def _print_options_csv(trades: list[OptionPositionTrade]) -> None:
    if not trades:
        click.echo(
            "symbol,structure,entry_date,exit_date,entry_premium,exit_premium,"
            "pnl,return_pct,exit_reason"
        )
        return
    rows = [
        {
            "symbol": t.symbol,
            "structure": t.structure,
            "entry_date": t.entry_date.isoformat(),
            "exit_date": t.exit_date.isoformat(),
            "entry_premium": t.entry_premium,
            "exit_premium": t.exit_premium,
            "pnl": t.pnl,
            "return_pct": t.return_pct,
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]
    click.echo(pd.DataFrame(rows).to_csv(index=False), nl=False)


__all__ = [
    "backtest",
    "build_panel",
    "options",
    "participants",
    "record",
    "regime",
    "show",
    "signals",
    "snapshot",
]
