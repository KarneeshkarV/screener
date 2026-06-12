"""RS-breakout pipeline criterion — ``screen -c rs-breakout``.

The standalone ``rs-breakout`` command has no ``--csv`` flag (it writes JSON
and Markdown by default), so ``--csv`` is ignored on this path. Use the
``rs-breakout`` command directly for the full option surface.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from rich.console import Console

from screener.cache import parse_ttl
from screener.criteria import criterion
from screener.rs_breakout import render_result


@criterion("rs-breakout", pipeline=True)
def rs_breakout_pipeline(
    *,
    market: str,
    limit: int,
    refresh: bool,
    cache_ttl: str,
    **_: Any,
) -> None:
    from screener.commands.rs_breakout import (
        run_rs_breakout_screen,
        write_default_outputs,
    )

    console = Console()
    as_of = date.today()
    result = run_rs_breakout_screen(
        market,
        as_of=as_of,
        benchmark=None,
        history_days=220,
        cache_ttl=parse_ttl(cache_ttl, default=900),
        refresh=refresh,
        console=console,
    )
    render_result(result, console, limit=int(limit), market=market)
    json_written, md_written = write_default_outputs(result, market, None, None)
    console.print(f"\n[dim]Wrote {json_written} + {md_written}[/dim]")
