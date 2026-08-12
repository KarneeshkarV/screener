"""Keltner-squeeze breakout (TTM-squeeze style, John Carter).

Evidence base:

* John Carter, "Mastering the Trade" (McGraw-Hill, 2006): the TTM Squeeze
  fires when Bollinger Bands (SMA20 ± 2 sd) trade *inside* the Keltner
  Channels (EMA20 ± 1.5 × ATR), i.e. volatility has contracted so far that
  the two bands cross. Carter's historical claim: a large majority of the
  market's biggest moves start from such a squeeze, because contraction
  concentrates energy that then releases in a directional expansion.
* The underlying regularity is volatility clustering / mean reversion of
  volatility (Mandelbrot 1963; GARCH, Bollerslev 1986): quiet periods are
  followed by expansion. The squeeze is an operational "volatility is at a
  local low" detector.
* Distinct from the repo's ``keltner_breakout`` (buys ANY cross above the
  upper channel) — here the buy fires only when the cross happens while a
  squeeze is (or was, within the last 20 bars) in force, i.e. a breakout out
  of contraction, not a re-cross in an already-widening channel.

Signal (causal, as-of bar ``t``):

    kc_mid = ema(close, 20);  kc_upper = kc_mid + 1.5 * atr(14)
    bb_mid = sma(close, 20);  bb_upper = bb_mid + 2 * stdev_pop(close, 20)
    bb_width = (bb_upper - bb_lower) / bb_mid;  kc_width = (kc_upper - kc_lower) / kc_mid
    squeeze[t] = bb_width[t] < kc_width[t]          # bands inside the channel
    squeeze_any_20[t] = any squeeze in the trailing 20 bars
    entry = crossover(close, kc_upper) and squeeze_any_20
    exit  = crossunder(close, kc_mid)               # give back the squeeze mid
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.frames import wilder_atr
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_BB_WINDOW = 20
_BB_MULT = 2.0
_KC_WINDOW = 20
_KC_MULT = 1.5
_ATR = 14
_SQUEEZE_LOOKBACK = 20  # how recent the squeeze must have been


def _prepare_squeeze(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)

        # Keltner channel (Pine-style EMA and Wilder ATR).
        kc_mid = close.ewm(span=_KC_WINDOW, adjust=False, min_periods=_KC_WINDOW).mean()
        atr = wilder_atr(high, low, close, _ATR, min_periods=_ATR)
        kc_upper = kc_mid + _KC_MULT * atr
        kc_lower = kc_mid - _KC_MULT * atr

        # Bollinger bands (population std, ddof=0, matching the repo indicator).
        bb_mid = close.rolling(_BB_WINDOW, min_periods=_BB_WINDOW).mean()
        bb_std = close.rolling(_BB_WINDOW, min_periods=_BB_WINDOW).std(ddof=0)
        bb_upper = bb_mid + _BB_MULT * bb_std
        bb_lower = bb_mid - _BB_MULT * bb_std

        bb_width = (bb_upper - bb_lower) / bb_mid
        kc_width = (kc_upper - kc_lower) / kc_mid
        squeeze = (bb_width < kc_width).astype(float)
        # Squeeze in force now or within the trailing 20 bars (contraction then
        # release). min_periods=1 keeps the very first bars defined.
        frame["kc_mid"] = kc_mid
        frame["kc_upper"] = kc_upper
        frame["squeeze_any_20"] = squeeze.rolling(
            _SQUEEZE_LOOKBACK, min_periods=1
        ).max()
        out[tv] = frame
    return out


def _lookback() -> int:
    # KC window + ATR seed, then the 20-bar "recent squeeze" window.
    return _KC_WINDOW + _SQUEEZE_LOOKBACK


register_expression_strategy(
    "keltner_squeeze_breakout",
    entry="crossover(close, kc_upper) and squeeze_any_20 > 0",
    exit="crossunder(close, kc_mid)",
    prepare_bars=_prepare_squeeze,
    required_lookback=_lookback,
)
