"""Supertrend(10, 3) flip: long while the band is bullish, flat when it flips.

The callable ``supertrend`` strategy in ``supertrend.py`` is the per-ticker
research form of the same rule; this is its expression form, which is what the
historical and rolling portfolio backtesters consume.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

register_expression_strategy(
    "supertrend_flip",
    entry="supertrend(10, 3.0) < 0",
    exit="supertrend(10, 3.0) > 0",
)
