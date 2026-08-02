"""Trade model and long-only walker for research strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from screener.ledger import Trade


class ResearchTrade(Trade):
    """Index-based research extension of the neutral trade lifecycle."""

    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    return_pct: float

    @property
    def ret(self) -> float:
        """Fractional research return kept for strategy-runner compatibility."""
        return self.return_pct


def _walk(
    entries: np.ndarray, exits: np.ndarray, close: np.ndarray, dates
) -> list[ResearchTrade]:
    """Long-only round-trip walker with close-based entries and exits."""
    trades: list[ResearchTrade] = []
    in_pos = False
    entry_i = -1
    entry_px = 0.0
    n = len(close)
    for i in range(n):
        if not in_pos:
            if entries[i]:
                in_pos = True
                entry_i = i
                entry_px = float(close[i])
        elif exits[i]:
            trades.append(
                ResearchTrade(
                    entry_idx=entry_i,
                    exit_idx=i,
                    entry_px=entry_px,
                    exit_px=float(close[i]),
                    entry_date=pd.Timestamp(dates[entry_i]),
                    exit_date=pd.Timestamp(dates[i]),
                    return_pct=float(close[i]) / entry_px - 1.0
                    if entry_px > 0
                    else 0.0,
                )
            )
            in_pos = False
    if in_pos:
        trades.append(
            ResearchTrade(
                entry_idx=entry_i,
                exit_idx=n - 1,
                entry_px=entry_px,
                exit_px=float(close[-1]),
                entry_date=pd.Timestamp(dates[entry_i]),
                exit_date=pd.Timestamp(dates[-1]),
                return_pct=float(close[-1]) / entry_px - 1.0 if entry_px > 0 else 0.0,
            )
        )
    return trades
