"""George-Hwang (2004) 52-week high factor.

Paper: George & Hwang, "The 52-Week High and Momentum Investing", Journal of
Finance 59(5), 2004. Stocks trading near their 52-week high continue to do
better than stocks far below theirs, and how far a name sits from its own high
predicts returns at least as well as past-return momentum.

Signal (causal, as-of bar ``t``):

    dist_from_52w_high[t] = (close[t] / max(close[t-251 : t]) - 1) * 100

In percent, in ``(-100, 0]``: 0 at the one-year high, -25 for a name 25% below
it. Larger (closer to zero) is better, so the rolling backtester fills its
``--top`` slots with the names nearest their own high.

The formula is *not* defined here: it lives once in
``screener.factors.recipes`` and this module is the backtest adapter over it.
The ``dist_from_52w_high`` screen scorer is the other adapter over the same
recipe. It is distinct from the older ``near_52_high`` criterion, which is a
vendor-snapshot breakout composite over TradingView columns; this factor reads
bars only.

Which is why this factor needs the tradeability gate. A dormant shell carried
at a flat stub closes at its own 52-week high every day, so an ungated ranking
hands it the first slot permanently. The recipe returns NaN for a window that
does not rest on real, traded prices, and NaN fails the entry gate.

Selection: this is a real cross-sectional factor portfolio, so the prepared
bars carry ``rank_score`` equal to the percent distance. The entry expression
``dist_from_52w_high > -100`` is satisfied by every tradeable name once a full
year of closes exists, so it is purely a "has enough history" gate; ranking
does the work.
"""

from __future__ import annotations

from screener.factors import entry_gate_expression, get_price_score
from screener.strategies.factor_adapter import (
    make_rank_score_lookback,
    make_rank_score_prepare,
)
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    register_expression_strategy,
)

_SCORE = get_price_score("dist_from_52w_high")

ENTRY = entry_gate_expression(_SCORE)

_prepare = make_rank_score_prepare(_SCORE)
_lookback = make_rank_score_lookback(_SCORE)

# No vendor prefilter: no TradingView column carries this exact ratio (the
# vendor's ``price_52_week_high`` is anchored to calendar dates, not to the last
# 252 sessions), so the screen scans unfiltered and the bar rule judges the
# whole field.

register_expression_strategy(
    "dist_from_52w_high",
    entry=ENTRY,
    exit=None,
    prepare_bars=_prepare,
    required_lookback=_lookback,
    profile=DEFAULT_STRATEGY_PROFILE,
)
