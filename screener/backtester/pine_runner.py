"""Compatibility wrapper for the package-owned research strategy runner."""
from __future__ import annotations

from screener.indicators.numpy import (
    _atr,
    _ema,
    _rma,
    _rsi,
    _sma,
    _stdev,
    _supertrend_dir,
)
from screener.research.pine_runner import (
    BENCHMARKS,
    fetch_ohlcv,
    load_universe,
    main,
)
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
from screener.strategies.registry import STRATEGIES
from screener.strategies.trades import Trade, _walk

__all__ = [
    "BENCHMARKS",
    "STRATEGIES",
    "Trade",
    "_atr",
    "_ema",
    "_rma",
    "_rsi",
    "_sma",
    "_stdev",
    "_supertrend_dir",
    "_walk",
    "fetch_ohlcv",
    "load_universe",
    "main",
    "strat_bb_breakout",
    "strat_macd_rsi",
    "strat_ma_cross",
    "strat_ma_cross_regime",
    "strat_ma_cross_st_entry",
    "strat_ma_cross_st_exit",
    "strat_rsi_ema",
    "strat_supertrend",
    "strat_supertrend_rsi",
    "strat_vivek_equity_tool",
]


if __name__ == "__main__":
    main()
