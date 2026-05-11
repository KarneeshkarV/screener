"""52-week breakout on above-average volume — entry-only Pine expression."""

from __future__ import annotations

from screener.strategies.spec import strategy


@strategy(
    "breakout",
    entry="close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)",
    exit=None,
)
def _breakout() -> None:
    """Expression-only strategy. Body unused."""
