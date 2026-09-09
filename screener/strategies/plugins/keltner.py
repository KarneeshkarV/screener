"""Keltner Channel breakout (Chester Keltner).

Source: Keltner, "How to Make Money in Commodities", 1960, Keltner Statistical
Service. Modern reference (20-day EMA ± 2×ATR): StockCharts chartSchool
https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

Channels are built on volatility rather than standard deviation:

    mid   = ema(close, 20)
    upper = mid + 2 * atr(20)
    lower = mid - 2 * atr(20)

Breakout rule: buy when close crosses above the upper channel (volatility
expansion is starting a move), ride until close crosses back below the lower
channel (the full channel is the exit). Trend-following profile.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_WINDOW = 20
_MULT = 2.0

_MID = f"ema(close, {_WINDOW})"
_UPPER = f"{_MID} + {_MULT} * atr({_WINDOW})"
_LOWER = f"{_MID} - {_MULT} * atr({_WINDOW})"

register_expression_strategy(
    "keltner_breakout",
    entry=f"crossover(close, {_UPPER})",
    exit=f"crossunder(close, {_LOWER})",
    required_lookback=lambda: _WINDOW,
)
