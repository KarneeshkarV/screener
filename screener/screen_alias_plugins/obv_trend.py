"""OBV-trend ``screen -c`` compatibility alias."""

from __future__ import annotations

from datetime import date


def obv_trend_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    from screener.commands.live_strategies import run_obv_trend_live

    run_obv_trend_live(market=market, as_of=date.today(), limit=limit)
