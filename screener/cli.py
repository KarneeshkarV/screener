"""Package-owned Click entrypoint for the screener CLI."""
from __future__ import annotations

import click

from screener.backtester.historical import backtest_historical
from screener.backtester.optimization.cli import optimize
from screener.backtester.rolling import backtest_rolling
from screener.commands.insiders import promoter_buys
from screener.commands.rs_breakout import rs_breakout
from screener.commands.screen import screen
from screener.config import load_config
from screener.logging_config import configure_logging
from screener.operator.cli import register as _register_operator_cli
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
def cli(ctx: click.Context, config_path: str | None, log_level: str, log_json: bool) -> None:
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
cli.add_command(promoter_buys)
cli.add_command(unusual_volume)
cli.add_command(backtest_historical)
cli.add_command(backtest_rolling)
_register_operator_cli(cli)
cli.add_command(optimize)


__all__ = ["cli"]
