"""Stochastic oscillator %K/%D cross (George Lane).

Source: George C. Lane, stochastic oscillator seminars (1950s-60s); modern
reference: StockCharts chartSchool
https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

Lane's oscillator measures where close sits within the recent high-low range:

    %K = 100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14))
    %D = sma(%K, 3)

The classic trading rule: buy when %K crosses above %D in the oversold zone
(< 30), sell when %K crosses below %D in the overbought zone (> 70).
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_WINDOW = 14
_D_SMOOTH = 3
_OS = 30
_OB = 70

_K = (
    f"100 * (close - lowest(low, {_WINDOW})) "
    f"/ (highest(high, {_WINDOW}) - lowest(low, {_WINDOW}))"
)
_D = f"sma({_K}, {_D_SMOOTH})"

register_expression_strategy(
    "stochastic_cross",
    entry=f"crossover({_K}, {_D}) and {_K} < {_OS}",
    exit=f"crossunder({_K}, {_D}) and {_K} > {_OB}",
    required_lookback=lambda: _WINDOW,
)
