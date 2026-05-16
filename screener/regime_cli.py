"""CLI commands for regime detection."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import YFinancePriceFetcher
from screener.regime import RegimeDetector


DEFAULT_BENCHMARKS = {"us": "SPY", "india": "^NSEI"}


@click.command(name="detect-regime")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market to evaluate.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Date to classify (default: today).",
)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark ticker (default: SPY / ^NSEI).",
)
def detect_regime(market, as_of_arg, benchmark):
    """Print the current market regime classification."""
    console = Console()
    as_of_date: date = (
        as_of_arg.date() if isinstance(as_of_arg, datetime) else (as_of_arg or date.today())
    )
    bench = benchmark or DEFAULT_BENCHMARKS[market]

    fetcher = click.get_current_context().obj or YFinancePriceFetcher()
    start = as_of_date - timedelta(days=400)
    end = as_of_date + timedelta(days=1)
    panel = fetcher.fetch([bench], start, end)
    benchmark_bars = panel.get(bench, pd.DataFrame())

    if benchmark_bars is None or benchmark_bars.empty:
        console.print("[red]No benchmark data fetched.[/red]")
        raise click.Abort()

    regime = RegimeDetector.classify(benchmark_bars, benchmark_bars)

    table = Table(title=f"{market.upper()} Regime as of {as_of_date}", show_header=True, header_style="bold")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Hurst", str(regime.hurst))
    table.add_row("Vol Regime", regime.vol_regime)
    table.add_row("Trend Regime", regime.trend_regime)
    table.add_row("Stress", str(regime.stress))
    table.add_row("Tradeable", "Yes" if regime.is_tradeable else "No")
    console.print(table)


@click.command(name="regime-history")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market to evaluate.",
)
@click.option(
    "--start",
    "start_arg",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    "end_arg",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark ticker (default: SPY / ^NSEI).",
)
def regime_history(market, start_arg, end_arg, benchmark):
    """Output a regime timeline between two dates."""
    console = Console()
    start_date: date = start_arg.date() if isinstance(start_arg, datetime) else start_arg
    end_date: date = end_arg.date() if isinstance(end_arg, datetime) else end_arg
    bench = benchmark or DEFAULT_BENCHMARKS[market]

    fetcher = click.get_current_context().obj or YFinancePriceFetcher()
    start_fetch = start_date - timedelta(days=300)
    end_fetch = end_date + timedelta(days=1)
    panel = fetcher.fetch([bench], start_fetch, end_fetch)
    benchmark_bars = panel.get(bench, pd.DataFrame())

    if benchmark_bars is None or benchmark_bars.empty:
        console.print("[red]No benchmark data fetched.[/red]")
        raise click.Abort()

    history = RegimeDetector.classify_series(benchmark_bars, benchmark_bars)
    if history.empty:
        console.print("[yellow]Insufficient data to compute regime history.[/yellow]")
        return

    mask = (history.index >= pd.Timestamp(start_date)) & (history.index <= pd.Timestamp(end_date))
    history = history.loc[mask]

    table = Table(
        title=f"{market.upper()} Regime History ({start_date} to {end_date})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Date", justify="left")
    table.add_column("Hurst", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Trend", justify="right")
    table.add_column("Stress", justify="right")
    table.add_column("Tradeable", justify="right")

    for day, row in history.iterrows():
        table.add_row(
            str(day.date() if hasattr(day, "date") else day),
            str(round(row["hurst"], 4)),
            str(row["vol_regime"]),
            str(row["trend_regime"]),
            str(round(row["stress"], 4)),
            "Yes" if row["is_tradeable"] else "No",
        )
    console.print(table)
