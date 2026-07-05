"""Parabolic SAR (Stop and Reverse) indicator."""

from __future__ import annotations

import numpy as np
from screener.indicators.registry import indicator


@indicator("sar")
def sar(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n_len = len(close)
    real_sar = np.zeros(n_len, dtype=np.float64)
    if n_len < 2:
        return real_sar

    initial_af = 0.02
    step_af = 0.02
    end_af = 0.2

    trend = np.zeros(n_len, dtype=np.int32)
    calc_sar = np.zeros(n_len, dtype=np.float64)
    ep = np.zeros(n_len, dtype=np.float64)
    af = np.zeros(n_len, dtype=np.float64)

    trend[1] = 1 if close[1] > close[0] else -1
    calc_sar[1] = high[0] if trend[1] > 0 else low[0]
    real_sar[1] = calc_sar[1]
    ep[1] = high[1] if trend[1] > 0 else low[1]
    af[1] = initial_af

    for i in range(2, n_len):
        temp = calc_sar[i - 1] + af[i - 1] * (ep[i - 1] - calc_sar[i - 1])
        if trend[i - 1] < 0:
            calc_sar[i] = max(temp, high[i - 1], high[i - 2])
            trend[i] = 1 if calc_sar[i] < high[i] else trend[i - 1] - 1
        else:
            calc_sar[i] = min(temp, low[i - 1], low[i - 2])
            trend[i] = -1 if calc_sar[i] > low[i] else trend[i - 1] + 1

        if trend[i] < 0:
            ep[i] = min(low[i], ep[i - 1]) if trend[i] != -1 else low[i]
        else:
            ep[i] = max(high[i], ep[i - 1]) if trend[i] != 1 else high[i]

        if abs(trend[i]) == 1:
            real_sar[i] = ep[i - 1]
            af[i] = initial_af
        else:
            real_sar[i] = calc_sar[i]
            if ep[i] == ep[i - 1]:
                af[i] = af[i - 1]
            else:
                af[i] = min(end_af, af[i - 1] + step_af)

    return real_sar
