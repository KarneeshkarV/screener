"""Ang-Hodrick-Xing-Zhang (2006) low-volatility anomaly.

Paper: Ang, Hodrick, Xing & Zhang, "The Cross-Section of Volatility and Expected
Returns", Journal of Finance 61(1), 2006. Low-volatility stocks historically earn
higher risk-adjusted returns than high-volatility stocks — the opposite of what
CAPM predicts.

Signal (causal, as-of bar ``t``):

    vol_252[t] = stdev( daily_returns , 252 )   # trailing realized volatility

Selection: this is a real cross-sectional factor portfolio. Lower volatility is
better, so the prepared bars carry ``rank_score = -vol_252`` and the rolling
backtester fills its ``--top`` slots with the *lowest*-volatility names. The
entry expression ``vol_252 > 0`` is satisfied for every symbol once 252 days of
returns exist (realized vol is strictly positive on non-constant prices), so it
acts purely as a "has enough history" eligibility gate; ranking does the work.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, strategy

_WINDOW = 252  # ~12 months of trading days


def realized_volatility(close: pd.Series) -> pd.Series:
    """Return the causal trailing-``_WINDOW`` daily-return volatility."""
    returns = close.astype(float).pct_change()
    return returns.rolling(_WINDOW, min_periods=_WINDOW).std()


def _prepare_low_vol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        vol = realized_volatility(frame["close"])
        frame["vol_252"] = vol
        # Lower volatility ranks higher -> negate so the descending ranker picks
        # the calmest names first.
        frame["rank_score"] = -vol
        out[tv] = frame
    return out


def _low_vol_lookback() -> int:
    # pct_change consumes one bar, then the rolling std needs ``_WINDOW`` returns.
    return _WINDOW + 1


@strategy(
    "low_volatility",
    entry="vol_252 > 0",
    exit=None,
    prepare_bars=_prepare_low_vol,
    required_lookback=_low_vol_lookback,
)
def _low_volatility() -> None:
    """Expression-only strategy; ranking is driven by the rank_score column."""
