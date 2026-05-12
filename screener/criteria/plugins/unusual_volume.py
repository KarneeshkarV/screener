"""Unusual-volume pipeline criterion — ``screen -c unusual-volume``.

The standalone ``unusual-volume`` command has no ``--csv`` or ``--cache-ttl``
flags, so those screen-level options are ignored on this path. Use the
``unusual-volume`` command directly for the full option surface.
"""

from __future__ import annotations

from typing import Any

import click

from screener.criteria import criterion


@criterion("unusual-volume", pipeline=True)
def unusual_volume_pipeline(
    *,
    market: str,
    limit: int,
    refresh: bool,
    **_: Any,
) -> None:
    from screener.unusual_volume.cli import unusual_volume as unusual_volume_cmd

    click.get_current_context().invoke(
        unusual_volume_cmd,
        market=market,
        limit=limit,
        refresh=refresh,
    )
