"""Click commands for building and inspecting options panels."""

from __future__ import annotations

from datetime import date, datetime

import click
import pandas as pd

from screener.cache import append_panel_snapshot
from screener.markets import market_option
from screener.options.panels import build_india_panel, show_symbol, snapshot_us
from screener.options.participant import build_participant_panel
from screener.options.regime import (
    build_india_vix_panel,
    build_us_regime_panel,
    fetch_india_vix_live,
)


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


__all__ = [
    "build_panel",
    "options",
    "participants",
    "regime",
    "show",
    "snapshot",
]
