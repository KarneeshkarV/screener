"""52-week breakout on above-average volume — entry-only Pine expression."""

from __future__ import annotations

from screener.strategies.spec import (
    StrategyProfile,
    register_expression_strategy,
)

# The ``breakout`` criterion's TradingView filters, declared rather than
# duplicated: ``close`` within 10% of ``price_52_week_high`` on above-average
# volume is the vendor-side spelling of the entry expression below. It cuts the
# field before bars are downloaded and must never remove a name the expression
# would have kept.
_PROFILE = StrategyProfile(tv_prefilter="breakout")

register_expression_strategy(
    "breakout",
    entry="close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)",
    exit=None,
    profile=_PROFILE,
)
