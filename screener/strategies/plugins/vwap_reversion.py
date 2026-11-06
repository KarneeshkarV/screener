"""RSI(2) mean-reversion inside an uptrend, anchored at/near VWAP.

Combines two documented edges:

* Connors' RSI(2) short-term mean reversion — buy a 2-day-RSI oversold print
  (< 10) only when the longer trend is up (close > SMA200), sell the bounce
  when RSI(2) crosses back above 70. The 200-SMA trend filter is what turns
  the raw RSI(2) rule from a martingale into a positive-expectancy setup.
* The VWAP anchor (institutional benchmark; see ``vwap_trend``) as a pullback
  zone: entries are only taken while price is at or just below the anchored
  VWAP, i.e. a healthy pullback within an uptrend rather than a late chase.

``vwap_reversion`` — trend + VWAP-filtered RSI(2) bounce. --hold caps the
maximum holding period (default recommendation ~10-20 sessions).
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.frames import wilder_rsi
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_RSI_PERIOD = 2
_RSI_ENTRY = 10.0
_RSI_EXIT = 70.0
_TREND_SMA = 200


def _prepare_vwap_rsi(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        typical = (frame["high"] + frame["low"] + close) / 3.0
        volume = frame["volume"].astype(float)
        cum_pv = (typical * volume).cumsum()
        cum_v = volume.cumsum()
        frame["vwap"] = cum_pv / cum_v.where(cum_v > 0)
        frame["rsi_2"] = wilder_rsi(close, _RSI_PERIOD)
        out[tv] = frame
    return out


def _lookback() -> int:
    return _TREND_SMA


register_expression_strategy(
    "vwap_reversion",
    entry=(
        f"rsi_2 < {_RSI_ENTRY} and close > sma(close, {_TREND_SMA}) "
        "and close < vwap * 1.03"
    ),
    exit=f"rsi_2 > {_RSI_EXIT}",
    prepare_bars=_prepare_vwap_rsi,
    required_lookback=_lookback,
)
