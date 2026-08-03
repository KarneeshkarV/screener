"""Shooting Star candlestick pattern strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import strategy
from screener.strategies.trades import ResearchTrade, _walk


@strategy("shooting_star")
def strat_shooting_star(df: pd.DataFrame) -> list[ResearchTrade]:
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)

    lower_bound = 0.2
    body_size = 0.5
    stop_threshold = 0.05
    holding_period = 7

    n_len = len(cl)
    entries = np.zeros(n_len, dtype=bool)
    exits = np.zeros(n_len, dtype=bool)

    if n_len < 4:
        return _walk(entries, exits, cl, df["date"].values)

    mean_body = np.mean(np.abs(op - cl))

    in_position = False
    entry_pos = 0.0
    counter = 0

    for i in range(3, n_len):
        if not in_position:
            # Check shooting star at i-1 and confirmation at i
            # condition1
            c1 = op[i - 1] >= cl[i - 1]
            body_diff = op[i - 1] - cl[i - 1] if c1 else cl[i - 1] - op[i - 1]

            if c1:
                # condition2
                c2 = (cl[i - 1] - lo[i - 1]) < lower_bound * body_diff
                # condition3
                c3 = body_diff < mean_body * body_size
                # condition4
                c4 = (hi[i - 1] - op[i - 1]) >= 2 * body_diff
                # condition5
                c5 = cl[i - 1] >= cl[i - 2]
                # condition6
                c6 = cl[i - 2] >= cl[i - 3]

                # confirmation at i
                c7 = hi[i] <= hi[i - 1]
                c8 = cl[i] <= cl[i - 1]

                if c2 and c3 and c4 and c5 and c6 and c7 and c8:
                    entries[i] = True
                    in_position = True
                    entry_pos = cl[i]
                    counter = 0

        else:
            counter += 1
            # Stop loss / profit
            if (
                np.abs(cl[i] / entry_pos - 1) > stop_threshold
                or counter >= holding_period
            ):
                exits[i] = True
                in_position = False

    return _walk(entries, exits, cl, df["date"].values)
