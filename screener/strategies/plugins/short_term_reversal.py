"""Jegadeesh (1990) / Lehmann (1990) short-term reversal.

Papers: Jegadeesh, "Evidence of Predictable Behavior of Security Returns",
Journal of Finance 45(3), 1990; Lehmann, "Fads, Martingales, and Market
Efficiency", Quarterly Journal of Economics, 1990. Stocks that underperform at
short horizons tend to rebound over the following month.

Signal (causal, as of bar ``t``):

    past_month_return[t] = close[t-1] / close[t-22] - 1
    st_rev[t] = -past_month_return[t]

The latest close in the formation return is the prior bar, deliberately skipping
bar ``t`` to reduce bid-ask-bounce contamination. ``rank_score = st_rev`` makes
the largest recent loser rank highest, while the positive entry gate admits only
names whose past-month return was negative.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_MONTH_START = 22
_SKIP = 1


def short_term_reversal_score(close: pd.Series) -> pd.Series:
    """Return the causal, one-day-skipped past-month reversal score."""
    close = close.astype(float)
    past_month_return = close.shift(_SKIP) / close.shift(_MONTH_START) - 1.0
    return -past_month_return


def _prepare_short_term_reversal(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        score = short_term_reversal_score(frame["close"])
        frame["st_rev"] = score
        frame["rank_score"] = score
        out[tv] = frame
    return out


def _short_term_reversal_lookback() -> int:
    # The ratio needs close[t-22]; retain one additional bar as a small warmup
    # margin around the 22-bar signal requirement.
    return _MONTH_START + 1


register_expression_strategy(
    "short_term_reversal",
    entry="st_rev > 0",
    exit=None,
    prepare_bars=_prepare_short_term_reversal,
    required_lookback=_short_term_reversal_lookback,
)
