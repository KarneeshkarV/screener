"""EMA150/EMA200 uptrend gated by sequential quarterly revenue growth."""

from __future__ import annotations

from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)

register_expression_strategy(
    "ema150_200_revenue_up_3q",
    entry=(
        "ema(close, 150) > ema(close, 200) "
        "and ema(close, 150) > 0 "
        "and ema(close, 200) > 0 "
        "and revenue_up_3q > 0"
    ),
    exit=None,
    profile=DEFAULT_STRATEGY_PROFILE,
)
