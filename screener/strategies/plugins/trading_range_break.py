"""Trading-range break (TRB): 150-day channel breakout.

Paper: Brock, Lakonishok & LeBaron, "Simple Technical Trading Rules and the
Stochastic Properties of Stock Returns", Journal of Finance 47(5), 1992.
https://doi.org/10.1111/j.1540-6261.1992.tb04681.x

The paper is the canonical academic study of technical rules on the Dow (1897-
1986). Its TRB rule: buy when the price penetrates the previous 150-day high
(the "resistance"), sell when it penetrates the previous 150-day low (the
"support"). BLL found the rule earned consistently positive returns that a
random-walk null could not explain.

Signal (as-of bar ``t``, prior-window extremes via ``prepare_bars``):

    high_150_prev[t] = max(close[t-150 : t])     # resistance, today excluded
    low_150_prev[t]  = min(close[t-150 : t])     # support, today excluded
    entry = close > high_150_prev
    exit  = close < low_150_prev

The shifted rolling windows keep the channel strictly prior: today's close is
never part of its own resistance/support. Trend-following profile: long holding
periods, "let winners run", and chop losses in sideways markets. Note the
150-day channel differs from the 20/10 Turtle (Zarattini) rule in both window
and research lineage.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_CHANNEL = 150


def _prepare_channel(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["high_150_prev"] = (
            close.rolling(_CHANNEL, min_periods=_CHANNEL).max().shift(1)
        )
        frame["low_150_prev"] = (
            close.rolling(_CHANNEL, min_periods=_CHANNEL).min().shift(1)
        )
        out[tv] = frame
    return out


def _lookback() -> int:
    return _CHANNEL


register_expression_strategy(
    "bll_trading_range_break",
    entry="close > high_150_prev",
    exit="close < low_150_prev",
    prepare_bars=_prepare_channel,
    required_lookback=_lookback,
)
