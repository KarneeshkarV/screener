"""Package-owned Click entrypoint for the screener CLI."""
from __future__ import annotations

import click

from screener.backtester.historical import backtest_historical
from screener.backtester.optimization.cli import optimize
from screener.backtester.rolling import backtest_rolling
from screener.commands.insiders import promoter_buys
from screener.commands.rs_breakout import rs_breakout
from screener.commands.screen import screen
from screener.operator.cli import register as _register_operator_cli
from screener.unusual_volume.cli import unusual_volume


@click.group()
def cli() -> None:
    """Stock screener for US and Indian markets."""


cli.add_command(screen)
cli.add_command(rs_breakout)
cli.add_command(promoter_buys)
cli.add_command(unusual_volume)
cli.add_command(backtest_historical)
cli.add_command(backtest_rolling)
_register_operator_cli(cli)
cli.add_command(optimize)


__all__ = ["cli"]
