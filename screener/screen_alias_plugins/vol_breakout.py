"""Volume-breakout ``screen -c`` compatibility alias."""

from __future__ import annotations

from datetime import date


def vol_breakout_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    from screener.commands.live_strategies import run_vol_breakout_live

    run_vol_breakout_live(market=market, as_of=date.today(), limit=limit)
