"""Click command for the S&P 500 index-inclusion event study (US only)."""

from __future__ import annotations

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import build_price_fetcher
from screener.display import print_csv
from screener.index_inclusion import LIMITATION_NOTE, run_inclusion_study
from screener.universes import load_sp500_membership


@click.command(name="index-inclusion")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us"]),
    default="us",
    help="Market to study. Only 'us' (S&P 500) is supported.",
)
@click.option(
    "--years",
    type=int,
    default=5,
    show_default=True,
    help="Trailing window of index additions to include.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output per-event rows as CSV.")
def index_inclusion(market: str, years: int, output_csv: bool) -> None:
    """Event study of post-addition excess drift for S&P 500 additions vs SPY."""
    if years <= 0:
        raise click.ClickException("--years must be a positive integer.")

    membership = load_sp500_membership()
    fetcher = build_price_fetcher()
    study = run_inclusion_study(membership, fetcher, years=years)

    if not study.events:
        click.echo(
            f"No S&P 500 additions in the last {years} year(s) had enough "
            "price data for the event study."
        )
        click.echo(f"Skipped {study.skipped} event(s) with insufficient price data.")
        click.echo(LIMITATION_NOTE)
        return

    if output_csv:
        rows = [
            {
                "symbol": event.symbol,
                "date_added": event.date_added.isoformat(),
                **{
                    f"excess_{horizon}d": event.excess[horizon]
                    for horizon in study.horizons
                },
            }
            for event in study.events
        ]
        print_csv(pd.DataFrame(rows))
        click.echo(
            f"Skipped {study.skipped} event(s) with insufficient price data.",
            err=True,
        )
        click.echo(LIMITATION_NOTE, err=True)
        return

    console = Console()
    table = Table(
        title=(
            f"S&P 500 post-addition drift vs SPY — additions in the last "
            f"{years} year(s)"
        )
    )
    table.add_column("Horizon")
    table.add_column("Mean excess", justify="right")
    table.add_column("Median excess", justify="right")
    table.add_column("Hit rate", justify="right")
    for summary in study.summaries:
        table.add_row(
            f"+{summary.horizon}d",
            f"{summary.mean:+.2%}",
            f"{summary.median:+.2%}",
            f"{summary.hit_rate:.0%}",
        )
    console.print(table)
    click.echo(
        f"Events: {len(study.events)} | "
        f"Skipped (insufficient price data): {study.skipped}"
    )
    click.echo(LIMITATION_NOTE)
