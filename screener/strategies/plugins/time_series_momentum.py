"""Moskowitz-Ooi-Pedersen (2012) time-series momentum.

Paper: Moskowitz, Ooi & Pedersen, "Time Series Momentum", Journal of
Financial Economics 104(2), 2012. Unlike cross-sectional momentum, this signal
compares each asset only with its own history: a positive trailing 12-month
return identifies an uptrend and permits a long entry.

Signal (causal, as-of bar ``t``):

    ts_ret[t] = close[t] / close[t-252] - 1
    vol_ann[t] = std(log(close).diff()[t-251:t], 252) * sqrt(252)
    rank_score[t] = ts_ret[t] / vol_ann[t]

The return deliberately includes the most recent month, matching the pure
trailing-return sign signal and distinguishing it from 12-1 cross-sectional
momentum. The inverse-volatility scaling follows the paper's risk-scaling idea;
for the long-only top-N backtester, it ranks eligible positive trends by their
risk-adjusted strength. Zero or unavailable volatility produces a NaN score and
therefore cannot enter.

All windows end at ``t`` and use only prices available through ``t``. The
required lookback is 253 bars: 252 daily log returns require 253 closes, and the
trailing return also needs the close 252 bars before the signal bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252
_REQUIRED_LOOKBACK = _WINDOW + 1


def trailing_12m_return(close: pd.Series) -> pd.Series:
    """Return each asset's causal trailing-252-trading-day return."""
    close = close.astype(float)
    return close / close.shift(_WINDOW) - 1.0


def annualized_volatility(close: pd.Series) -> pd.Series:
    """Return causal annualized volatility from 252 trailing log returns."""
    close_f = close.astype(float)
    log_close = pd.Series(np.log(close_f.to_numpy()), index=close_f.index)
    log_returns = log_close.diff()
    std = log_returns.rolling(_WINDOW, min_periods=_WINDOW).std()
    vol: pd.Series = std * np.sqrt(_WINDOW)
    return vol


def _prepare_time_series_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach own-asset trend, volatility, and risk-scaled ranking columns."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        ts_ret = trailing_12m_return(frame["close"])
        vol_ann = annualized_volatility(frame["close"])
        frame["ts_ret"] = ts_ret
        frame["vol_ann"] = vol_ann
        frame["rank_score"] = (ts_ret / vol_ann).where(vol_ann > 0)
        out[tv] = frame
    return out


def _time_series_momentum_lookback() -> int:
    """Return the 253 closes needed for a 252-return trailing window."""
    return _REQUIRED_LOOKBACK


register_expression_strategy(
    "time_series_momentum",
    entry="ts_ret > 0",
    exit=None,
    prepare_bars=_prepare_time_series_momentum,
    required_lookback=_time_series_momentum_lookback,
)
