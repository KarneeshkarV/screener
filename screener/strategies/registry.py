"""Registry for implemented research strategies."""
from __future__ import annotations

from collections.abc import Iterator

from screener.strategies.base import StrategyFn
from screener.strategies.pine_ports import (
    strat_bb_breakout,
    strat_macd_rsi,
    strat_ma_cross,
    strat_ma_cross_regime,
    strat_ma_cross_st_entry,
    strat_ma_cross_st_exit,
    strat_rsi_ema,
    strat_supertrend,
    strat_supertrend_rsi,
    strat_vivek_equity_tool,
)

STRATEGIES: dict[str, StrategyFn] = {
    "supertrend": strat_supertrend,
    "supertrend_rsi": strat_supertrend_rsi,
    "macd_rsi": strat_macd_rsi,
    "rsi_ema": strat_rsi_ema,
    "ma_cross": strat_ma_cross,
    "bb_breakout": strat_bb_breakout,
    "vivek_equity_tool": strat_vivek_equity_tool,
    "ma_cross_regime": strat_ma_cross_regime,
    "ma_cross_st_entry": strat_ma_cross_st_entry,
    "ma_cross_st_exit": strat_ma_cross_st_exit,
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
