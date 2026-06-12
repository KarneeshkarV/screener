"""Promoter-buys pipeline criterion — ``screen -c promoter-buys``."""

from __future__ import annotations

from typing import Any

from screener.criteria import criterion

# Screen-context defaults for options the generic ``screen`` command does not
# expose (the standalone ``promoter-buys`` command's own defaults).
_DEFAULT_UNIVERSE_SIZE = 200
_DEFAULT_MIN_CHANGE_PCT = 0.0
_DEFAULT_WORKERS = 10


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
