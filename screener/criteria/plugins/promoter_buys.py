"""Promoter-buys pipeline criterion — ``screen -c promoter-buys``."""

from __future__ import annotations

from typing import Any

import click

from screener.criteria import criterion


@criterion("promoter-buys", pipeline=True)
def promoter_buys_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
    **_: Any,
) -> None:
    from screener.commands.insiders import promoter_buys as promoter_buys_cmd

    click.get_current_context().invoke(
        promoter_buys_cmd,
        market=market,
        limit=limit,
        output_csv=output_csv,
        refresh=refresh,
        cache_ttl=cache_ttl,
    )
