"""Strategy metadata and callable types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from screener.strategies.trades import Trade


@dataclass(frozen=True)
class StrategyMeta:
    name: str
    family: str
    min_bars: int = 50
    description: str = ""


StrategyFn = Callable[[pd.DataFrame], list[Trade]]
