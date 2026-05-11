"""Registry for implemented research strategies."""

from __future__ import annotations

from collections.abc import Iterator

from screener.strategies.base import StrategyFn
from screener.strategies.pine_ports import (
    strat_bb_breakout,
    strat_ma_cross_st_entry,
    strat_supertrend,
)

STRATEGIES: dict[str, StrategyFn] = {
    "supertrend": strat_supertrend,
    "bb_breakout": strat_bb_breakout,
    "ma_cross_st_entry": strat_ma_cross_st_entry,
}


def get_strategy(name: str) -> StrategyFn:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}"
        ) from None


def iter_strategies() -> Iterator[tuple[str, StrategyFn]]:
    return iter(STRATEGIES.items())
