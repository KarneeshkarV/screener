"""Click subcommand: ``screener operator-scan``.

Wired into the existing ``main.py:cli`` group at import time.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import click

from .output import write_csv
from .process import build_dataset
from .screen import label


def register(cli_group: click.Group) -> None:
    """Attach the ``operator-scan`` subcommand to ``cli_group``."""
    cli_group.add_command(operator_scan)


@click.command(name="operator-scan")
@click.option(
    "--date", "as_of",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to scan (YYYY-MM-DD). Defaults to today; "
         "weekends/holidays auto-walk back to the most recent trading day.",
)
@click.option(
    "--universe",
    type=click.Choice(["fo", "fo+cash"]),
    default="fo+cash",
    show_default=True,
    help="fo = F&O list only (~210); fo+cash = F&O + top-500 cash by volume.",
)
@click.option(
    "--output",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output CSV path. Defaults to daily_operator_data_YYYYMMDD.csv in CWD.",
)
@click.option(
    "--only-actions",
    is_flag=True,
    help="Only emit rows with a non-null Operator_Action.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Log progress to stderr.",
)
def operator_scan(as_of, universe, out_path, only_actions, verbose):
    """NSE Operator Intent screener — daily Cash + F&O OI signal.

    Combines NSE Cash Bhavcopy delivery + VWAP with the F&O UDiff
    Bhavcopy per-expiry Open Interest, and labels each F&O stock with
    one of: Long Build-up, Short Covering, Short Build-up, Long Unwinding.
    Long Build-ups within 15% of the 52-week high are flagged as
    High_Momentum_Watch. Output is a single CSV.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if isinstance(as_of, datetime):
        as_of = as_of.date()
    if as_of is None:
        as_of = date.today()

    df, actual = build_dataset(as_of, universe_mode=universe)
    df = label(df)
    written = write_csv(df, actual, out_path=out_path, only_actions=only_actions)

    actions = df["Operator_Action"].value_counts(dropna=True)
    n_hmw = int(df["High_Momentum_Watch"].sum())
    click.echo(f"Operator scan: trading day {actual}  ·  {len(df)} symbols  ·  wrote {written}")
    if not actions.empty:
        for label_name, count in actions.items():
            click.echo(f"  {label_name:<16} {count}")
    click.echo(f"  High_Momentum_Watch: {n_hmw}")
