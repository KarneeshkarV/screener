"""52-Week High momentum: buy a fresh 252-day-high breakout, ride the trend.

Paper: George & Hwang, "The 52-Week High and Momentum Investing", Journal of
Finance 59(5), 2004. https://doi.org/10.1111/j.1540-6261.2004.00695.x

George & Hwang show that a stock's price nearness to its 52-week high is a
stronger predictor of future returns than raw 12-month momentum, and that the
effect survives after the Jegadeesh-Titman momentum premium is controlled for.
Their economic story is anchoring: investors anchor to the 52-week high, so
prices cluster below it; stocks that push through it carry positive drift.

Signal (as-of bar ``t``):

    high_252_prev[t] = max(close[t-252 : t])            # prior-year high, today excluded
    entry = close > high_252_prev                       # fresh 52-week high
    exit  = crossunder(close, sma(close, 50))           # trend break / momentum loss

The prior-year high is computed on the *shifted* rolling window so the signal
only fires when today's close actually clears the previous year's peak — the
naive `close > highest(close, 252)` would include today's bar in the maximum
and never trigger. The SMA50 exit replaces the paper's monthly rebalancing with
a daily trend-break rule.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252  # one trading year
_TREND = 50  # exit trend line


def _prepare_high(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        # Rolling max over the trailing 252 closes, shifted so today's close is
        # excluded from the reference peak.
        frame["high_252_prev"] = (
            close.rolling(_WINDOW, min_periods=_WINDOW).max().shift(1)
        )
        out[tv] = frame
    return out


def _lookback() -> int:
    return _WINDOW


register_expression_strategy(
    "fifty_two_week_high",
    entry="close > high_252_prev",
    exit=f"crossunder(close, sma(close, {_TREND}))",
    prepare_bars=_prepare_high,
    required_lookback=_lookback,
)
