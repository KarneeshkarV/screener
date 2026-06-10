"""Click command for calendar seasonality statistics."""

from __future__ import annotations

from datetime import date, timedelta

import click
from rich.console import Console

from screener.backtester.data import build_price_fetcher, tv_to_yf
from screener.seasonality import compute_seasonality, render_report, report_to_csv

# Tolerance before warning that the available history is shorter than
# requested — avoids noise from weekends/holidays at the window edge.
_SPAN_TOLERANCE_DAYS = 45


@click.command(name="seasonality")
@click.argument("ticker")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market the ticker trades in (controls symbol mapping).",
)
@click.option(
    "--years",
    type=int,
    default=10,
    show_default=True,
    help="Years of history to analyze.",
)
@click.option(
    "--csv",
    "csv_output",
    is_flag=True,
    default=False,
    help="Emit the stats as CSV on stdout instead of tables.",
)
def seasonality(ticker: str, market: str, years: int, csv_output: bool) -> None:
    """Show monthly, turn-of-month and day-of-week seasonality for TICKER."""
    if years < 1:
        raise click.UsageError("--years must be >= 1")
    end = date.today()
    start = end - timedelta(days=int(years * 365.25) + 7)
    fetcher = click.get_current_context().obj or build_price_fetcher()
    yf_symbol = tv_to_yf(ticker, market)
    bars = fetcher.fetch([yf_symbol], start, end).get(yf_symbol)
    if bars is None or bars.empty:
        raise click.ClickException(f"No price data for {ticker} ({yf_symbol}).")
    try:
        report = compute_seasonality(bars, ticker=ticker)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if report.start > start + timedelta(days=_SPAN_TOLERANCE_DAYS):
        span_years = (report.end - report.start).days / 365.25
        click.echo(
            f"Note: only ~{span_years:.1f} years of data available "
            f"({report.start} → {report.end}); requested {years}.",
            err=True,
        )

    if csv_output:
        click.echo(report_to_csv(report), nl=False)
    else:
        render_report(report, Console())
