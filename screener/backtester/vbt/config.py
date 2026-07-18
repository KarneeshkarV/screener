"""Shared configuration and types for vectorbt sweeps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pandas as pd

MetricName = Literal["sharpe", "total_return", "calmar"]
StrategyName = Literal["sma_cross"]
IndicatorName = Literal[
    "sma",
    "ema",
    "breakout",
    "bbands",
    "supertrend",
    "keltner",
    "rsi",
    "macd",
    "vol_breakout",
    "obv_trend",
    "sma_rsi",
    "breakout_rsi",
]
VALID_INDICATORS: tuple[str, ...] = (
    "sma",
    "ema",
    "breakout",
    "bbands",
    "supertrend",
    "keltner",
    "rsi",
    "macd",
    "vol_breakout",
    "obv_trend",
    "sma_rsi",
    "breakout_rsi",
)

# Fixed-config defaults (kept as module constants so the viewer can document them).
DEFAULT_BREAKOUT_WINDOWS: tuple[int, ...] = (20, 55, 100)
DEFAULT_BBANDS_WINDOWS: tuple[int, ...] = (20, 50)
DEFAULT_BBANDS_STD: float = 2.0
DEFAULT_SUPERTREND_PERIODS: tuple[int, ...] = (7, 10, 14)
DEFAULT_SUPERTREND_MULT: float = 3.0
DEFAULT_KELTNER_WINDOWS: tuple[int, ...] = (20, 50)
DEFAULT_KELTNER_MULT: float = 2.0
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_RSI_THRESHOLDS: tuple[int, ...] = (50, 55, 60)
DEFAULT_VOL_MA_WINDOW: int = 20
DEFAULT_VOL_MULTIPLIER: float = 1.0
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9
DEFAULT_OBV_EMA_WINDOWS: tuple[int, ...] = (20, 50)

# Indicators that require extra OHLCV panels for signal generation.
VOLUME_INDICATORS: frozenset[str] = frozenset({"vol_breakout", "obv_trend"})
HL_INDICATORS: frozenset[str] = frozenset({"supertrend", "keltner"})

SOLO_INDICATORS_NAN_SLOW: frozenset[str] = frozenset(
    {
        "breakout",
        "bbands",
        "supertrend",
        "keltner",
        "rsi",
        "vol_breakout",
        "obv_trend",
        "breakout_rsi",
    }
)

StrategyBuilder = Callable[
    [pd.DataFrame, int, int, int, Any],
    tuple[pd.DataFrame, pd.DataFrame],
]

INITIAL_CAPITAL_DEFAULT = 100_000.0

# Representative notional for converting CostModel side fractions into the
# single per-side ``fees`` scalar that vectorbt accepts.
COST_MODEL_DEFAULT = "flat"
COMMISSION_BPS_DEFAULT = 0.0
FEE_NOTIONAL_DEFAULT = 100_000.0

DISCLAIMER = (
    "[yellow]Exploration only — approximations; validate with backtest-rolling.\n"
    "Not modeled: slot allocation, partial exits, dividends, custom slippage "
    "(slippage=0), trailing stops, stop-loss, take-profit.\n"
    "Fees: approximated per-side from the cost model at a representative "
    "notional (TAF/per-share components excluded); not full cash accounting.\n"
    "Fills: next-bar open (MOO match); residual MOC differences may apply.[/yellow]"
)

# Months in --walk-forward TRAIN:TEST are approximated as 30 calendar days.
WALK_FORWARD_DAYS_PER_MONTH = 30

# OOS metric columns aggregated across walk-forward windows.
WALK_FORWARD_OOS_COLUMNS: tuple[str, ...] = (
    "sharpe",
    "total_return",
    "calmar",
    "max_drawdown",
    "win_rate",
)
