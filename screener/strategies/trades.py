"""Trade model and long-only walker for research strategies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp

    @property
    def ret(self) -> float:
        return self.exit_px / self.entry_px - 1.0 if self.entry_px > 0 else 0.0


def _walk(entries: np.ndarray, exits: np.ndarray, close: np.ndarray, dates) -> list[Trade]:
    """Long-only round-trip walker with close-based entries and exits."""
    trades: list[Trade] = []
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
                Trade(
                    entry_i,
                    i,
                    entry_px,
                    float(close[i]),
                    pd.Timestamp(dates[entry_i]),
                    pd.Timestamp(dates[i]),
                )
            )
            in_pos = False
    if in_pos:
        trades.append(
            Trade(
                entry_i,
                n - 1,
                entry_px,
                float(close[-1]),
                pd.Timestamp(dates[entry_i]),
                pd.Timestamp(dates[-1]),
            )
        )
    return trades
