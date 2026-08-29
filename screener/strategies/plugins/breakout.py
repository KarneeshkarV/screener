"""52-week breakout on above-average volume — entry-only Pine expression."""

from __future__ import annotations

from screener.strategies.spec import (
    StrategyProfile,
    register_expression_strategy,
)

# The volume leg of the ``breakout`` criterion, declared rather than
# duplicated. It cuts the field before bars are downloaded and must never
# remove a name the expression would have kept.
#
# Only the volume leg, not the whole criterion: the criterion's other leg
# reads ``price_52_week_high``, the 52-week high of *highs*, while the entry
# below reads the 52-week high of *closes*. ``max(high) >= max(close)``, so
# the vendor threshold sits at or above the rule's and drops names inside the
# band. ``tests/correctness/test_screen_backtest_reconciliation.py`` caught it.
_PROFILE = StrategyProfile(tv_prefilter="above_avg_volume")

register_expression_strategy(
    "breakout",
    entry="close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)",
    exit=None,
    profile=_PROFILE,
)
