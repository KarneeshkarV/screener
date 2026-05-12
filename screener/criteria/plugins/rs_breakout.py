"""RS-breakout pipeline criterion — ``screen -c rs-breakout``.

The standalone ``rs-breakout`` command has no ``--csv`` flag (it writes JSON
and Markdown by default), so ``--csv`` is ignored on this path. Use the
``rs-breakout`` command directly for the full option surface.
"""

from __future__ import annotations

from typing import Any

import click

from screener.criteria import criterion


@criterion("rs-breakout", pipeline=True)
def rs_breakout_pipeline(
    *,
    market: str,
    limit: int,
    refresh: bool,
    cache_ttl: str,
    **_: Any,
) -> None:
    from screener.commands.rs_breakout import rs_breakout as rs_breakout_cmd

    click.get_current_context().invoke(
        rs_breakout_cmd,
        market=market,
        limit=limit,
        refresh=refresh,
        cache_ttl=cache_ttl,
    )
