"""Strategy callable types."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from screener.strategies.trades import Trade


StrategyFn = Callable[[pd.DataFrame], list[Trade]]
