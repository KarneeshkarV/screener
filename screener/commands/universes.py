"""Discover built-in and user-defined universe providers."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

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
