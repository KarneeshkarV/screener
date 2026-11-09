"""Williams %R mean reversion (Larry Williams).

Source: Williams, "How I Made One Million Dollars Last Year Trading
Commodities", 1979, Windsor Books.
https://www.amazon.com/Million-Dollars-Last-Trading-Commodities/dp/0934233128

%R is the inverse of fast stochastic: how close today's close is to the
14-period high:

    %R = -100 * (highest(high, 14) - close) / (highest(high, 14) - lowest(low, 14))

Williams' rule: buy when %R dips below -80 (deep oversold), sell when it rises
above -20 (overbought). Same family as the stochastic but with the opposite
polarity and a slightly different interpretation of the extremes.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_WINDOW = 14
_OS = -80
_OB = -20

_WR = (
    f"-100 * (highest(high, {_WINDOW}) - close) "
    f"/ (highest(high, {_WINDOW}) - lowest(low, {_WINDOW}))"
)

register_expression_strategy(
    "williams_percent_r",
    entry=f"{_WR} < {_OS}",
    exit=f"{_WR} > {_OB}",
    required_lookback=lambda: _WINDOW,
)
