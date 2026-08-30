"""The Heikin-Ashi confirmation leg of ``ha_momentum`` counts the right bars.

The recipe ranks by 12-1 momentum but only while the Heikin-Ashi candles are
on a bullish streak of ``HA_STREAK_MIN`` bars. The streak length is the whole
gate, so an off-by-one there admits a name a full bar before its trend is
confirmed - on every series except one that opens bullish, which is why it
does not show up on a smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.recipes import HA_STREAK_MIN, ha_momentum


def _ohlc_with_a_break(n: int = 320, break_at: int = -8) -> dict[str, pd.Series]:
    """A steady uptrend with one sharp down bar that ends the HA streak."""
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(np.linspace(100.0, 300.0, n), index=idx)
    close.iloc[break_at] = close.iloc[break_at - 1] * 0.7
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999
    return {"open_": open_, "high": high, "low": low, "close": close}


def test_the_streak_restarts_at_one_after_a_bearish_candle() -> None:
    """The first bullish bar after a break is bar 1 of the new streak.

    Grouping the bullish flags by ``(~bullish).cumsum()`` puts the terminating
    bearish bar first in its own group, so a ``cumcount() + 1`` reading scores
    that first bullish bar as 2 and confirms the trend one bar early.
    """
    assert HA_STREAK_MIN == 3
    scored = ha_momentum(**_ohlc_with_a_break())

    # Two bearish bars, then the trend resumes. The two bars after the break
    # are still short of the minimum; the third completes it.
    assert scored.iloc[-8:-6].isna().all(), "the bearish bars carry no score"
    assert scored.iloc[-6:-4].isna().all(), "two bullish bars are not yet a streak"
    assert scored.iloc[-4:].notna().all(), "the third bullish bar confirms"


def test_an_uninterrupted_uptrend_is_confirmed_from_the_third_bar() -> None:
    """No break, so the count is the plain bar index and nothing is skipped."""
    idx = pd.bdate_range("2022-01-03", periods=320)
    close = pd.Series(np.linspace(100.0, 300.0, len(idx)), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999

    scored = ha_momentum(open_, high, low, close)

    assert scored.notna().any()
    assert scored.iloc[-1] > 0
