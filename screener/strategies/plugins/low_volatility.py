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

Which is why this factor needs the tradeability gate as badly as momentum does,
in the mirror direction. A dormant shell carried at a flat $0.0001 stub has a
realized volatility of *exactly zero* - it is the calmest possible stock - so
an ungated ranking hands it the first slot every time. Measured on cached US
bars, the five lowest-volatility names in the whole cache were all stub quotes
with 93-100% of their sessions at zero volume. ``vol_252`` is therefore NaN
wherever :func:`screener.tradeable.tradeable_span` says the window is not a
market, and a NaN fails ``vol_252 > 0``, so those names are not ranked at all.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)
from screener.tradeable import tradeable_span

_WINDOW = 252  # ~12 months of trading days


def realized_volatility(close: pd.Series, volume: pd.Series | None = None) -> pd.Series:
    """Return the causal trailing-``_WINDOW`` daily-return volatility.

    NaN where the window does not rest on real, traded prices. Unlike a ratio
    this reads *every* close in the span, so one stub anywhere inside it is
    enough to corrupt the estimate - hence ``tradeable_span`` rather than
    ``tradeable_window``.
    """
    values = close.astype(float)
    returns = values.pct_change()
    vol = returns.rolling(_WINDOW, min_periods=_WINDOW).std()
    return vol.where(tradeable_span(values, volume, window=_WINDOW))


def _prepare_low_vol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        vol = realized_volatility(frame["close"], frame.get("volume"))
        frame["vol_252"] = vol
        # Lower volatility ranks higher -> negate so the descending ranker picks
        # the calmest names first.
        frame["rank_score"] = -vol
        out[tv] = frame
    return out


def _low_vol_lookback() -> int:
    # pct_change consumes one bar, then the rolling std needs ``_WINDOW`` returns.
    return _WINDOW + 1


register_expression_strategy(
    "low_volatility",
    entry="vol_252 > 0",
    exit=None,
    prepare_bars=_prepare_low_vol,
    required_lookback=_low_vol_lookback,
    profile=DEFAULT_STRATEGY_PROFILE,
)
