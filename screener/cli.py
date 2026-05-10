"""Package-owned Click entrypoint for the screener CLI."""

from __future__ import annotations

import functools
import time
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from screener.backtester.historical import backtest_historical
from screener.backtester.lab import backtest_lab
from screener.backtester.optimization.cli import optimize
from screener.backtester.rolling import backtest_rolling
from screener.commands.garp import garp
from screener.commands.insiders import promoter_buys
from screener.commands.rs_breakout import rs_breakout
from screener.commands.screen import screen
from screener.config import load_config
from screener.logging_config import configure_logging
from screener.operator.cli import register as _register_operator_cli
from screener import usage
from screener.unusual_volume.cli import unusual_volume


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="YAML or JSON config file with CLI defaults.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging verbosity for diagnostic events on stderr.",
)
@click.option(
    "--log-json",
    is_flag=True,
    default=False,
    help="Emit one JSON event per line on stderr instead of human-readable logs.",
)
@click.pass_context
def cli(
    ctx: click.Context, config_path: str | None, log_level: str, log_json: bool
) -> None:
    """Stock screener for US and Indian markets."""
    if config_path:
        config = load_config(config_path)
        ctx.default_map = config
        if (
            ctx.get_parameter_source("log_level") == click.core.ParameterSource.DEFAULT
            and "log_level" in config
        ):
            log_level = str(config["log_level"])
        if (
            ctx.get_parameter_source("log_json") == click.core.ParameterSource.DEFAULT
            and "log_json" in config
        ):
            log_json = bool(config["log_json"])
    configure_logging(level=log_level, json=log_json)


cli.add_command(screen)
cli.add_command(rs_breakout)
cli.add_command(garp)
cli.add_command(promoter_buys)
cli.add_command(unusual_volume)
cli.add_command(backtest_historical)
cli.add_command(backtest_rolling)
cli.add_command(backtest_lab)
_register_operator_cli(cli)
cli.add_command(optimize)


def _wrap_usage_tracking(
    command: click.Command, feature_path: tuple[str, ...]
) -> None:
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            _wrap_usage_tracking(child, (*feature_path, name))
        return
    if command.callback is None or getattr(command.callback, "_usage_tracked", False):
        return

    feature = " ".join(feature_path)
    original = command.callback

    @functools.wraps(original)
    def tracked_callback(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original(*args, **kwargs)
        usage.record_feature_usage(
            feature,
            command_path=f"screener {feature}",
            duration_ms=usage.elapsed_ms(started_at),
        )
        return result

    tracked_callback._usage_tracked = True
    command.callback = tracked_callback


@click.command(name="usage-report")
def usage_report() -> None:
    """Show successful feature usage counts from Turso."""
    rows = usage.feature_usage_counts()
    if not rows:
        click.echo("No feature usage recorded for this project yet.")
        return

    table = Table(title="Feature Usage")
    table.add_column("Feature")
    table.add_column("Uses", justify="right")
    table.add_column("Last Used")
    for row in rows:
        table.add_row(row.feature, str(row.count), row.last_used_at or "")
    Console().print(table)


cli.add_command(usage_report)


for _name, _command in cli.commands.items():
    if _name != "usage-report":
        _wrap_usage_tracking(_command, (_name,))


__all__ = ["cli"]
