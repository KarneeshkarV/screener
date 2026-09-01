"""Discover built-in and user-defined universe providers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.universe_backfill import backfill_snapshots, backfillable_universes
from screener.universes import (
    available_universes,
    configured_universe_names,
    get_universe_definition,
    sync_universe_snapshot,
)


@click.group(name="universes")
def universes_group() -> None:
    """List available backtest universes."""


@universes_group.command(name="list")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Include custom universes from this TOML/YAML/JSON file.",
)
def list_universes(config_path: Path | None) -> None:
    """Show built-in and configured universe names."""
    table = Table(title="Universes")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Market")
    table.add_column("Benchmark")
    for name in available_universes():
        definition = get_universe_definition(name)
        table.add_row(name, "built-in", definition.market, definition.benchmark)
    table.add_row("dynamic", "rule-based", "us / india", "market default")
    if config_path is not None:
        try:
            names = configured_universe_names(config_path)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        for name in names:
            table.add_row(name, "custom", "from config", "from config")
    Console().print(table)


@universes_group.command(name="sync")
@click.argument("name", type=click.Choice(list(available_universes())))
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Snapshot CSV path (default: ~/.screener/universes/<name>_snapshots.csv).",
)
@click.option(
    "--use-cache",
    is_flag=True,
    default=False,
    help="Permit a cached constituent response instead of forcing a refresh.",
)
def sync_universe(name: str, output: Path | None, use_cache: bool) -> None:
    """Capture today's membership, appending only when it changes."""
    target = output or (
        Path.home() / ".screener" / "universes" / f"{name}_snapshots.csv"
    )
    try:
        path, changed, count = sync_universe_snapshot(
            name, output=target, use_cache=use_cache
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    action = "appended" if changed else "unchanged"
    Console().print(f"{name}: {action}; {count} symbols; {path}")


@universes_group.command(name="backfill")
@click.argument("name", type=click.Choice(list(backfillable_universes())))
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Snapshot CSV path (default: ~/.screener/universes/<name>_snapshots.csv).",
)
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Ignore archived crawls observed before this date.",
)
@click.option(
    "--until",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Ignore archived crawls observed after this date.",
)
@click.option(
    "--min-symbols",
    type=int,
    default=1,
    show_default=True,
    help="Reject a crawl that parsed into fewer symbols than this.",
)
def backfill_universe(
    name: str,
    output: Path | None,
    since: datetime | None,
    until: datetime | None,
    min_symbols: int,
) -> None:
    """Rebuild dated membership history from Internet Archive crawls.

    Each distinct archived copy of the index constituent CSV becomes one dated
    snapshot, which a ``type: snapshots`` custom universe turns into
    point-in-time eligibility windows. Membership changes are dated at the crawl
    that first observed them, so the history is lookahead-free but only as fine
    as the crawl cadence - inspect the printed dates for gaps.
    """
    target = output or (
        Path.home() / ".screener" / "universes" / f"{name}_snapshots.csv"
    )
    try:
        result = backfill_snapshots(
            name,
            output=target,
            since=since.date() if since else None,
            until=until.date() if until else None,
            min_symbols=min_symbols,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    console = Console()
    table = Table(title=f"{name}: archived membership snapshots")
    table.add_column("Observed")
    table.add_column("Symbols", justify="right")
    table.add_column("Source")
    members_by_date = _snapshot_counts(result.path)
    for snapshot in result.snapshots:
        observed = snapshot.observed.isoformat()
        table.add_row(observed, str(members_by_date.get(observed, 0)), snapshot.url)
    console.print(table)
    console.print(
        f"{name}: {len(result.snapshots)} snapshots; "
        f"{len(result.symbols)} distinct symbols; {result.rows} rows; {result.path}"
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _snapshot_counts(path: Path) -> dict[str, int]:
    """Return symbols-per-effective-date for a written snapshot CSV."""
    frame = pd.read_csv(path, dtype=str)
    counts = frame.groupby("effective_date")["symbol"].nunique()
    return {str(key): int(value) for key, value in counts.items()}
