"""Volatility-expansion breakout (Donchian channel + ATR expansion confirmation).

Evidence base:

* Time-series momentum / trend following — Moskowitz, Ooi & Pedersen,
  "Time Series Momentum", Journal of Financial Economics 104(2), 2012:
  past 12-month excess returns predict future returns across asset classes;
  the classic implementation is a Donchian-channel breakout (Turtle system).
* Volatility clustering — Mandelbrot (1963); Bollerslev GARCH (1986):
  volatility is serially correlated, so a regime of expanding volatility
  tends to persist. A breakout accompanied by ATR expansion is more likely to
  be a real, news-driven directional move ("volatility begets volatility")
  rather than a random wiggle above the channel in a quiet tape.

This strategy is deliberately distinct from the repo's existing breakout
family (``turtle_breakout`` 20-day no-filter, ``keltner_breakout`` EMA±ATR
channel, ``chandelier_breakout`` 55-day with ATR trailing stop,
``bll_trading_range_break`` 150-day close channel, ``breakout`` 52-week +
volume): the entry demands BOTH a fresh N-day-high breakout AND that the
stock's ATR-to-price ratio is above its own trailing median (volatility
expansion confirmation). In sideways, low-vol regimes the breakout fires but
the expansion gate blocks it, which is the mechanism intended to protect the
flat 1y/2y India windows.

Signal (causal, as-of bar ``t``):

    prior_high_60[t] = max(high[t-60 : t])          # fresh-breakout reference
    atr_14[t]        = Wilder ATR(14)               # volatility level
    atr_pct[t]       = atr_14[t] / close[t]         # volatility-to-price ratio
    vol_expand[t]    = atr_pct[t] > median(atr_pct[t-120 : t])  # expansion
    entry = close > prior_high_60 + 0.5 * atr_14 and vol_expand
    exit  = close < min(low[t-20 : t])              # give back the channel
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.frames import wilder_atr
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_BREAKOUT = 60  # N-day high reference for the breakout
_ATR = 14  # Wilder ATR period
_ATR_MEDIAN = 120  # trailing window for the ATR-pct median
_EXIT_WINDOW = 20  # prior low reference for the exit
_ATR_ADD = 0.5  # breakout must clear the channel by 0.5 * ATR


def _prepare_vol_expansion(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        # Prior N-day high/low, today excluded (fresh-breakout semantics).
        frame["prior_high_60"] = high.rolling(_BREAKOUT).max().shift(1)
        frame["prior_low_20"] = low.rolling(_EXIT_WINDOW).min().shift(1)
        atr = wilder_atr(high, low, close, _ATR, min_periods=_ATR)
        frame["atr_14"] = atr
        atr_pct = atr / close
        frame["atr_med_120"] = atr_pct.rolling(
            _ATR_MEDIAN, min_periods=_ATR_MEDIAN
        ).median()
        frame["vol_expand"] = atr_pct > frame["atr_med_120"]
        out[tv] = frame
    return out


def _lookback() -> int:
    return max(_BREAKOUT, _ATR_MEDIAN)


register_expression_strategy(
    "vol_expansion_breakout",
    entry=f"close > prior_high_60 + {_ATR_ADD} * atr_14 and vol_expand",
    exit="close < prior_low_20",
    prepare_bars=_prepare_vol_expansion,
    required_lookback=_lookback,
)
