"""12-1 momentum, ranked only while Heikin-Ashi confirms an active uptrend.

Combines two independent signals rather than restating either:

* the ranking leg is the same ``mom_12_1`` factor as ``momentum_12_1``
  (Jegadeesh & Titman 12-1 cross-sectional momentum);
* the eligibility leg is Heikin-Ashi trend confirmation - a name only
  carries a score while its HA candles are on a bullish streak (see
  ``screener.factors.recipes.ha_momentum`` for the exact rule).

Both live once in ``screener.factors.recipes`` so this module is a thin
backtest adapter, same as ``momentum_12_1.py``.
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

_HA_MOMENTUM_SCORE = get_price_score("ha_momentum")

ENTRY = entry_gate_expression(_HA_MOMENTUM_SCORE)

register_expression_strategy(
    "ha_momentum",
    entry=ENTRY,
    exit=None,
    prepare_bars=make_rank_score_prepare(_HA_MOMENTUM_SCORE),
    required_lookback=make_rank_score_lookback(_HA_MOMENTUM_SCORE),
    profile=DEFAULT_STRATEGY_PROFILE,
)
