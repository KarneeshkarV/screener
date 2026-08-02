"""Bollinger Bands Double Bottom Pattern Recognition strategy."""

from __future__ import annotations
import numpy as np
import pandas as pd
from screener.indicators.plugins.bollinger_bands import bollinger_bands as _bb
from screener.indicators.plugins.stdev import stdev as _stdev
from screener.strategies.spec import strategy
from screener.strategies.trades import ResearchTrade, _walk


@strategy("bb_pattern")
def strat_bb_pattern(df: pd.DataFrame) -> list[ResearchTrade]:
    cl = df["close"].to_numpy(dtype=float)
    lower, middle, upper = _bb(cl, 20, 2.0)
    stds = _stdev(cl, 20)

    period = 75
    alpha = 0.0001
    beta = 0.0001

    entries = np.zeros_like(cl, dtype=bool)
    exits = np.zeros_like(cl, dtype=bool)

    in_position = False

    for i in range(period, len(cl)):
        moveon = False
        threshold = 0.0

        if not in_position:
            # Entry condition 4
            if cl[i] > upper[i]:
                j = -1
                for _j in range(i, i - period, -1):
                    # Entry condition 2
                    if (np.abs(middle[_j] - cl[_j]) < alpha) and (
                        np.abs(middle[_j] - upper[i]) < alpha
                    ):
                        moveon = True
                        j = _j
                        break

                if moveon:
                    moveon = False
                    k = -1
                    for _k in range(j, i - period, -1):
                        # Entry condition 1
                        if np.abs(lower[_k] - cl[_k]) < alpha:
                            threshold = cl[_k]
                            moveon = True
                            k = _k
                            break

                if moveon:
                    moveon = False
                    _node_l = -1
                    for _l in range(k, i - period, -1):
                        # Node L (for pattern)
                        if middle[_l] < cl[_l]:
                            moveon = True
                            _node_l = _l
                            break

                if moveon:
                    moveon = False
                    for m in range(i, j, -1):
                        # Entry condition 3
                        if (
                            (cl[m] - lower[m] < alpha)
                            and (cl[m] > lower[m])
                            and (cl[m] < threshold)
                        ):
                            entries[i] = True
                            in_position = True
                            moveon = True
                            break

        # Exit condition: contraction
        if in_position and stds[i] < beta and not moveon:
            exits[i] = True
            in_position = False

    return _walk(entries, exits, cl, df["date"].values)
