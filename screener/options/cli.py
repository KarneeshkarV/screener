"""Click commands for building and inspecting options panels."""

from __future__ import annotations

from datetime import date, datetime

import click
import pandas as pd

from screener.markets import market_option
from screener.options.panels import build_india_panel, show_symbol


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


__all__ = ["build_panel", "options", "show"]
