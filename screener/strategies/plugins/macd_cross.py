"""Classic MACD signal-line cross (12, 26, 9).

Source: Appel, "Technical Analysis: Power Tools for Active Investors", 2005
(Prentice Hall). MACD = EMA12 - EMA26 of close; the signal line is a 9-period
EMA of MACD. Appel's original timing rule: buy when MACD crosses above its
signal line, sell when it crosses below.

Signal (as-of bar ``t``):

    macd   = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    entry  = crossover(macd, signal)
    exit   = crossunder(macd, signal)

Momentum-oscillator profile: more responsive than the golden cross, but whipsaws
in ranges. The repo's existing ``macd_oscillator`` (SMA 10/21 cross) and
``macd_rsi`` (MACD + RSI confirmation) are different rules; this is the
canonical Appel signal-line cross.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_MACD_EXPR = "ema(close, 12) - ema(close, 26)"
_SIGNAL_EXPR = f"ema({_MACD_EXPR}, 9)"

register_expression_strategy(
    "macd_signal_cross",
    entry=f"crossover({_MACD_EXPR}, {_SIGNAL_EXPR})",
    exit=f"crossunder({_MACD_EXPR}, {_SIGNAL_EXPR})",
    required_lookback=lambda: 26,
)
