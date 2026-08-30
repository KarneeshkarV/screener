"""Package-owned Click entrypoint for the screener CLI."""

from __future__ import annotations

import functools
import importlib
import time
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from screener.config import load_config
from screener.logging_config import configure_logging

# ``screener.usage`` is imported where it is used, not here. It pulls in
# pydantic, and nothing needs it until a command has already finished running -
# so at module load it is 0.15s spent before Click has even read argv, on every
# invocation including a bare ``--help``.

# Keep in sync with screener.agentio.DETAIL_LEVELS (avoid importing agentio /
# pandas at CLI module load for --help).
_AGENT_DETAIL_LEVELS = ("summary", "head", "full")
_AGENT_DEFAULT_DETAIL = "head"

# Lazy subcommand table: CLI name -> (module path, attribute, short help).
# Import happens on first resolve (invoke or subcommand --help), not at
# `import screener.cli` / top-level `--help`. Short help is stored here so
# Click's group help formatter does not force-load every module.
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "screen": (
        "screener.commands.screen",
        "screen",
        "Screen stocks based on technical criteria.",
    ),
    "history": (
        "screener.commands.history_list",
        "history_list",
        "List persisted screen runs (replay them with `backtest-historical --from-run`).",
    ),
    "history-backup": (
        "screener.commands.history_backup",
        "history_backup",
        "Mirror local screen-run history to Turso (or restore it with --restore).",
    ),
    "rs-breakout": (
        "screener.commands.rs_breakout",
        "rs_breakout",
        "Screen stocks for RS + SuperTrend + breakout/volume setups.",
    ),
    "garp": (
        "screener.commands.garp",
        "garp",
        "Find GARP stocks using market-specific fundamental data.",
    ),
    "mark-minervini": (
        "screener.commands.minervini",
        "mark_minervini",
        "Screen for stocks matching Mark Minervini's Trend Template.",
    ),
    "vol-breakout-live": (
        "screener.commands.live_strategies",
        "vol_breakout_live",
        "Donchian N-day high breakout confirmed by above-average volume.",
    ),
    "obv-trend-live": (
        "screener.commands.live_strategies",
        "obv_trend_live",
        "OBV crosses above/below its EMA — flow-leads-price trend follower.",
    ),
    "conviction": (
        "screener.commands.conviction",
        "conviction",
        "One composite conviction card for TICKER, fusing the screen pillars.",
    ),
    "promoter-buys": (
        "screener.commands.insiders",
        "promoter_buys",
        "Find stocks where promoter/insider holding has increased.",
    ),
    "institutional": (
        "screener.commands.institutional",
        "institutional",
        "Show FMP institutional ownership per ticker, ranked by QoQ change.",
    ),
    "filings": (
        "screener.commands.filings",
        "filings",
        "Read US SEC filings (10-K/10-Q/8-K) via Financial Modeling Prep.",
    ),
    "index-inclusion": (
        "screener.commands.index_inclusion",
        "index_inclusion",
        "Event study of post-addition excess drift for S&P 500 additions vs SPY.",
    ),
    "seasonality": (
        "screener.commands.seasonality",
        "seasonality",
        "Show monthly, turn-of-month and day-of-week seasonality for TICKER.",
    ),
    "universes": (
        "screener.commands.universes",
        "universes_group",
        "List available backtest universes.",
    ),
    "unusual-volume": (
        "screener.unusual_volume.cli",
        "unusual_volume",
        "Detect abnormal trading volume across a market on a given day.",
    ),
    "earnings-backtest": (
        "screener.earnings_backtest.cli",
        "earnings_backtest",
        "Backtest earnings-drift entry (E-1/E-2 → E) with sentiment filters.",
    ),
    "earnings-pead": (
        "screener.earnings_backtest.cli",
        "earnings_pead",
        "Backtest post-earnings-announcement drift (next open → hold N days).",
    ),
    "backtest-historical": (
        "screener.backtester.historical",
        "backtest_historical",
        "Run an accurate historical backtest with Pine-like entry/exit expressions.",
    ),
    "backtest-rolling": (
        "screener.backtester.rolling",
        "backtest_rolling",
        "Run a true daily rolling backtest over a date window.",
    ),
    "factor-tearsheet": (
        "screener.backtester.factor_tearsheet",
        "factor_tearsheet",
        "Compute factor IC and quantile tearsheet for a named strategy.",
    ),
    "operator-scan": (
        "screener.operator.cli",
        "operator_scan",
        "NSE Operator Intent screener — daily Cash + F&O OI signal.",
    ),
    "optimize": (
        "screener.backtester.optimization.cli",
        "optimize",
        "Optimize and validate backtest parameters.",
    ),
    "cache": (
        "screener.commands.cache",
        "cache_group",
        "Inspect and prune the screener's on-disk caches.",
    ),
    "options": (
        "screener.options.cli",
        "options",
        "Build, snapshot, and inspect normalized options data.",
    ),
}


def _instrument_usage_tracking(
    command: click.Command, feature_path: tuple[str, ...]
) -> None:
    """Wrap ``command``'s leaf callbacks with usage tracking.

    Groups recurse (their own callback is left untouched); each leaf records a
    successful feature usage plus an invocation row under its space-joined
    feature path. The ``_usage_tracked`` marker keeps re-wrapping idempotent.
    """
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            _instrument_usage_tracking(child, (*feature_path, name))
        return
    if command.callback is None or getattr(command.callback, "_usage_tracked", False):
        return

    feature = " ".join(feature_path)
    original = command.callback

    @functools.wraps(original)
    def tracked_callback(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        status = "success"
        try:
            return original(*args, **kwargs)
        except BaseException as exc:
            status = type(exc).__name__
            raise
        finally:
            from screener import usage

            duration = usage.elapsed_ms(started_at)
            if status == "success":
                usage.record_feature_usage(
                    feature,
                    command_path=f"screener {feature}",
                    duration_ms=duration,
                )
            usage.record_feature_invocation(
                feature,
                command_path=f"screener {feature}",
                duration_ms=duration,
                status=status,
                params=kwargs,
            )

    # Marker attribute so re-wrapping is idempotent (read via getattr above);
    # mypy can't model dynamic attributes on a function object.
    tracked_callback._usage_tracked = True  # type: ignore[attr-defined]
    command.callback = tracked_callback


class UsageTrackedGroup(click.Group):
    """Click group that instruments each command it gains with usage tracking.

    Commands are wrapped as they are added (subgroups recurse), so every leaf
    records its own feature usage and invocation rows under the full
    space-joined command path. Mark a command's callback with
    ``_usage_tracked = True`` to opt it out (used for ``usage-report``, which
    reads the usage tables and should not record itself).

    Heavy subcommands may be registered via :meth:`register_lazy` so
    ``import screener.cli`` / ``screener --help`` do not pay their import tax.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # name -> (module, attr, short_help)
        self._lazy_commands: dict[str, tuple[str, str, str]] = {}

    def register_lazy(
        self, name: str, module: str, attr: str, short_help: str = ""
    ) -> None:
        """Register a subcommand loaded from ``module.attr`` on first use."""
        self._lazy_commands[name] = (module, attr, short_help)

    def add_command(self, cmd: click.Command, name: str | None = None) -> None:
        super().add_command(cmd, name)
        resolved = name or cmd.name
        if resolved is not None:
            self._lazy_commands.pop(resolved, None)
            _instrument_usage_tracking(cmd, (resolved,))

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = set(super().list_commands(ctx))
        names.update(self._lazy_commands)
        return sorted(names)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None
        module_path, attr, _short = spec
        module = importlib.import_module(module_path)
        loaded = getattr(module, attr)
        if not isinstance(loaded, click.Command):
            raise TypeError(
                f"lazy command {cmd_name!r} resolved to {type(loaded).__name__}, "
                "expected click.Command"
            )
        # add_command instruments usage and removes the lazy entry.
        self.add_command(loaded, cmd_name)
        return loaded

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """List subcommands without importing lazy modules for short help."""
        commands: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name in self._lazy_commands:
                commands.append((name, self._lazy_commands[name][2]))
                continue
            cmd = super().get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            commands.append((name, cmd.get_short_help_str()))

        if not commands:
            return

        limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
        formatted: list[tuple[str, str]] = []
        for name, help_s in commands:
            if help_s and limit > 0 and len(help_s) > limit:
                help_s = help_s[: max(0, limit - 3)].rstrip() + "..."
            formatted.append((name, help_s))

        with formatter.section("Commands"):
            formatter.write_dl(formatted)


@click.group(cls=UsageTrackedGroup)
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
@click.option(
    "--agent/--no-agent",
    "agent_mode",
    default=None,
    help=(
        "Token-lean output for AI agents: a bounded digest on stdout plus a "
        "full-data CSV in ~/tmp. Auto-enabled under a known agent harness; "
        "override with SCREENER_AGENT=0/1."
    ),
)
@click.option(
    "--agent-detail",
    type=click.Choice(_AGENT_DETAIL_LEVELS),
    default=None,
    help=(
        "How much of the result to inline in agent mode. "
        f"[default: {_AGENT_DEFAULT_DETAIL}]"
    ),
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str | None,
    log_level: str,
    log_json: bool,
    agent_mode: bool | None,
    agent_detail: str | None,
) -> None:
    """Stock screener for US and Indian markets."""
    from screener import agentio

    agentio.configure(agent_mode, agent_detail)  # type: ignore[arg-type]
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


assert isinstance(cli, UsageTrackedGroup)
for _name, (_module, _attr, _short) in _LAZY_COMMANDS.items():
    cli.register_lazy(_name, _module, _attr, _short)


@click.command(name="usage-report")
def usage_report() -> None:
    """Show successful feature usage counts from Turso."""
    from screener import usage

    console = Console()
    rows = usage.feature_usage_counts()
    if not rows:
        click.echo("No feature usage recorded for this project yet.")
    else:
        table = Table(title="Feature Usage")
        table.add_column("Feature")
        table.add_column("Uses", justify="right")
        table.add_column("Last Used")
        for row in rows:
            table.add_row(row.feature, str(row.count), row.last_used_at or "")
        console.print(table)

    invocations = usage.invocation_rollup(limit=30)
    if not invocations:
        click.echo("No invocations recorded yet.")
        return

    inv_table = Table(title="Recent invocations (by feature/market/criteria/status)")
    inv_table.add_column("Feature")
    inv_table.add_column("Market")
    inv_table.add_column("Criteria")
    inv_table.add_column("Status")
    inv_table.add_column("Uses", justify="right")
    inv_table.add_column("Last Used")
    inv_table.add_column("Top extras")
    for inv in invocations:
        inv_table.add_row(
            inv.feature,
            inv.market or "",
            inv.criteria or "",
            inv.status,
            str(inv.count),
            inv.last_used_at or "",
            inv.top_extras or "",
        )
    console.print(inv_table)


# usage-report reads the usage tables; exempt it from tracking itself by
# pre-marking its callback so UsageTrackedGroup.add_command skips it.
if usage_report.callback is not None:
    usage_report.callback._usage_tracked = True  # type: ignore[attr-defined]
cli.add_command(usage_report)


__all__ = ["cli"]
