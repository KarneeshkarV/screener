"""GARP pipeline criterion — ``screen -c garp`` runs the GARP scan."""

from __future__ import annotations

from typing import Any

import click

from screener.criteria import criterion


@criterion("garp", pipeline=True)
def garp_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
    **_: Any,
) -> None:
    from screener.commands.garp import garp as garp_cmd

    click.get_current_context().invoke(
        garp_cmd,
        market=market,
        limit=limit,
        output_csv=output_csv,
        refresh=refresh,
        cache_ttl=cache_ttl,
    )
