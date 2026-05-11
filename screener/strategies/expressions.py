"""Named Pine-like strategy expression shortcuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NamedStrategy:
    entry: str
    exit: Optional[str]


NAMED_STRATEGIES: dict[str, NamedStrategy] = {
    "ema_trend": NamedStrategy(
        entry="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
        exit="crossunder(close, ema(close, 20))",
    ),
    "breakout": NamedStrategy(
        entry="close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)",
        exit=None,
    ),
    "rs_breakout": NamedStrategy(
        entry="rs_breakout_entry > 0",
        exit=None,
    ),
}


def resolve_strategy(name: str) -> NamedStrategy:
    try:
        return NAMED_STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(NAMED_STRATEGIES)}"
        ) from None
