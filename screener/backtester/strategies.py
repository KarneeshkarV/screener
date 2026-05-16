"""Compatibility exports for named strategy expression shortcuts."""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedStrategy:
    entry: str
    exit: Optional[str]


STRATEGIES: dict[str, NamedStrategy] = {
    "ema_trend": NamedStrategy(
        entry="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
        exit="crossunder(close, ema(close, 20))",
    ),
    "breakout": NamedStrategy(
        entry="close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)",
        exit=None,
    ),
    "golden_cross": NamedStrategy(
        entry="crossover(sma(close, 50), sma(close, 200))",
        exit="crossunder(sma(close, 50), sma(close, 200))",
    ),
    "ema_trend_volume": NamedStrategy(
        entry="close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)",
        exit="crossunder(close, ema(close, 20))",
    ),
    "rsi2_mean_reversion": NamedStrategy(
        entry="rsi(close, 2) < 20 and close > ema(close, 200)",
        exit="rsi(close, 2) > 60",
    ),
    "golden_cross_volume": NamedStrategy(
        entry="crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)",
        exit="crossunder(sma(close, 50), sma(close, 200))",
    ),
}


def resolve_strategy(name: str) -> NamedStrategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}"
        ) from None


__all__ = ["NamedStrategy", "STRATEGIES", "resolve_strategy"]
