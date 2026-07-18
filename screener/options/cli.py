"""Click commands for building and inspecting options panels."""

from __future__ import annotations

from datetime import date, datetime

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.cache import append_panel_snapshot
from screener.earnings_backtest.metrics import compute_backtest_summary
from screener.markets import get_price_fetcher, market_option
from screener.options.bt_models import OptionPositionTrade, OptionsBacktestConfig
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
    "regime",
    "show",
    "snapshot",
]
