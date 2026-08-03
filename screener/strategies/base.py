"""Strategy callable types."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from screener.strategies.trades import ResearchTrade

StrategyFn = Callable[[pd.DataFrame], list[ResearchTrade]]
