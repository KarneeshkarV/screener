"""Promoter-buys ``screen -c`` compatibility alias."""

from __future__ import annotations

_DEFAULT_UNIVERSE_SIZE = 200
_DEFAULT_MIN_CHANGE_PCT = 0.0
_DEFAULT_WORKERS = 10


def promoter_buys_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    from screener.commands.insiders import run_promoter_buys

    run_promoter_buys(
        market=market,
        universe_size=_DEFAULT_UNIVERSE_SIZE,
        limit=limit,
        min_change_pct=_DEFAULT_MIN_CHANGE_PCT,
        min_yf_net_pct=None,
        require_both=False,
        min_market_cap=None,
        workers=_DEFAULT_WORKERS,
        output_csv=output_csv,
        refresh=refresh,
        cache_ttl=cache_ttl,
    )
