"""vol_breakout pipeline criterion — ``screen -c vol-breakout``.

Donchian N-day high breakout confirmed by above-average volume. Sweep winner
on US SP500 was ``window=100, hold=15``; defaults in the underlying command
match. Use ``screen -c vol-breakout -m us`` for the configured combo.
"""

from __future__ import annotations

from typing import Any

import click

from screener.criteria import criterion


@criterion("vol-breakout", pipeline=True)
def vol_breakout_pipeline(
    *,
    market: str,
    limit: int,
    **_: Any,
) -> None:
    from screener.commands.live_strategies import vol_breakout_live

    click.get_current_context().invoke(
        vol_breakout_live,
        market=market,
        limit=limit,
    )
