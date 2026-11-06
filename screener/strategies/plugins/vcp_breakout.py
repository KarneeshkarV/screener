"""VCP-lite: breakout after a volatility contraction (Minervini).

Evidence base:

* Mark Minervini, "Trade Like a Stock Market Wizard" (McGraw-Hill, 2013) —
  the Volatility Contraction Pattern: a stock basing after a prior advance
  shows *successively tighter* daily ranges / smaller candles; the contraction
  (quiet, low-ATR base) is the pre-requisite for a low-risk breakout. Buying
  the breakout out of contraction, not the continuation, is the whole point.
* Under the hood this is volatility mean reversion / clustering: after a
  period of compressed volatility, expansions carry the largest moves
  (Bollerslev 1986 GARCH; Mandelbrot 1963). A breakout from a *contracted*
  base also has a tighter risk reference (the base low) than a breakout in an
  already-expanded tape.
* Distinct from the repo's existing breakouts (``turtle_breakout``,
  ``keltner_breakout``, ``chandelier_breakout``, ``bll_trading_range_break``,
  ``vol_expansion_breakout``): the entry requires ATR-to-price BELOW its own
  trailing median (contraction) rather than above it, and the exit is the
  20-day prior low (the base support).

Signal (causal, as-of bar ``t``):

    prior_high_60[t] = max(high[t-60 : t])
    atr_pct[t]       = atr(14)[t] / close[t]
    vcp_contract[t]  = atr_pct[t] < median(atr_pct[t-120 : t])   # contraction
    entry = close > prior_high_60 and vcp_contract
    exit  = close < min(low[t-20 : t])
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.frames import wilder_atr
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_BREAKOUT = 60
_ATR = 14
_ATR_MEDIAN = 120
_EXIT_WINDOW = 20


def _prepare_vcp(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        frame["prior_high_60"] = high.rolling(_BREAKOUT).max().shift(1)
        frame["prior_low_20"] = low.rolling(_EXIT_WINDOW).min().shift(1)
        atr = wilder_atr(high, low, close, _ATR, min_periods=_ATR)
        atr_pct = atr / close
        frame["atr_med_120"] = atr_pct.rolling(
            _ATR_MEDIAN, min_periods=_ATR_MEDIAN
        ).median()
        frame["vcp_contract"] = atr_pct < frame["atr_med_120"]
        out[tv] = frame
    return out


def _lookback() -> int:
    return max(_BREAKOUT, _ATR_MEDIAN)


register_expression_strategy(
    "vcp_breakout",
    entry="close > prior_high_60 and vcp_contract",
    exit="close < prior_low_20",
    prepare_bars=_prepare_vcp,
    required_lookback=_lookback,
)
