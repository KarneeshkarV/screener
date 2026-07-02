"""EMA150/EMA200 uptrend gated by sequential quarterly revenue growth."""

from __future__ import annotations

from screener.strategies.spec import strategy


@strategy(
    "ema150_200_revenue_up_3q",
    entry=(
        "ema(close, 150) > ema(close, 200) "
        "and ema(close, 150) > 0 "
        "and ema(close, 200) > 0 "
        "and revenue_up_3q > 0"
    ),
    exit=None,
)
def _ema150_200_revenue_up_3q() -> None:
    """Expression-only strategy. Body unused."""
