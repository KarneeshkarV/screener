"""RS-breakout ``screen -c`` compatibility alias."""

from __future__ import annotations

from datetime import date

from rich.console import Console

from screener.cache import parse_ttl
from screener.rs_breakout import render_result


def rs_breakout_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    """Run the alias; ignores --csv (cache TTL is honored)."""
    import click

    if output_csv:
        click.echo("rs-breakout ignores --csv", err=True)
    from screener.commands.rs_breakout import (
        run_rs_breakout_screen,
        write_default_outputs,
    )

    console = Console()
    result = run_rs_breakout_screen(
        market,
        as_of=date.today(),
        benchmark=None,
        history_days=220,
        cache_ttl=parse_ttl(cache_ttl, default=900),
        refresh=refresh,
        console=console,
    )
    render_result(result, console, limit=int(limit), market=market)
    json_written, md_written = write_default_outputs(result, market, None, None)
    console.print(f"\n[dim]Wrote {json_written} + {md_written}[/dim]")
