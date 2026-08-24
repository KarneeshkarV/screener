"""Long-trend EMA strategy: ride EMA20 above EMA200, exit on close cross-under."""

from __future__ import annotations

from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)

register_expression_strategy(
    "ema_trend",
    entry="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
    exit="crossunder(close, ema(close, 20))",
    profile=DEFAULT_STRATEGY_PROFILE,
)
