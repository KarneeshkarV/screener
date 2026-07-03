"""RSI Pattern Recognition strategy."""
from __future__ import annotations
import numpy as np
import pandas as pd
from screener.indicators.numpy import _rsi
from screener.strategies.spec import strategy
from screener.strategies.trades import Trade, _walk

@strategy("rsi_pattern")
def strat_rsi_pattern(df: pd.DataFrame) -> list[Trade]:
    cl = df["close"].to_numpy(dtype=float)
    lag = 14
    rsi_vals = _rsi(cl, lag)
    
    period = 25
    delta = 0.2
    head = 1.1
    shoulder = 1.1
    
    n_len = len(cl)
    entries = np.zeros(n_len, dtype=bool)
    exits = np.zeros(n_len, dtype=bool)
    
    in_position = False
    counter = 0
    entry_rsi = 0.0
    exit_rsi = 4.0
    exit_days = 5
    
    for i in range(period + lag, n_len):
        moveon = False
        top = 0.0
        bottom = 0.0
        
        if not in_position and cl[i] != np.max(cl[i-period:i]):
            j = (i - period) + np.argmax(cl[i-period:i])
            if np.abs(cl[j] - cl[i]) > head * delta:
                bottom = cl[i]
                moveon = True
                
            if moveon:
                moveon = False
                k = -1
                for _k in range(j, i):
                    if np.abs(cl[_k] - bottom) < delta:
                        moveon = True
                        k = _k
                        break
                        
            if moveon:
                moveon = False
                l = -1
                for _l in range(j, i - period, -1):
                    if np.abs(cl[_l] - bottom) < delta:
                        moveon = True
                        l = _l
                        break
                        
            if moveon:
                moveon = False
                m = -1
                for _m in range(i - period, l):
                    if np.abs(cl[_m] - bottom) < delta:
                        moveon = True
                        m = _m
                        break
                        
            if moveon:
                moveon = False
                n = m + np.argmax(cl[m:l]) if l > m else m
                if (cl[n] - bottom > shoulder * delta) and (cl[j] - cl[n] > shoulder * delta):
                    top = cl[n]
                    moveon = True
                    
            if moveon:
                for o in range(k, i):
                    if np.abs(cl[o] - top) < delta:
                        entries[i] = True
                        in_position = True
                        entry_rsi = rsi_vals[i]
                        counter = 0
                        moveon = True
                        break
                        
        if in_position and not moveon:
            counter += 1
            if (rsi_vals[i] - entry_rsi > exit_rsi) or (counter > exit_days):
                exits[i] = True
                in_position = False
                counter = 0
                entry_rsi = 0.0
                
    return _walk(entries, exits, cl, df["date"].values)
