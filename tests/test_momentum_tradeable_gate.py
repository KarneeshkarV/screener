"""A 12-1 ratio is only a return when both its legs are prices a market set.

Two vendor artefacts break that and neither looks like missing data: a dormant
line carried at a flat sub-cent stub, and a series the vendor repeats forward
through months of no trading. Both leave a full-length close column, so the
ratio computes and lands at the top of a cross-sectional rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.recipes import (
    MIN_TRADEABLE_PRICE,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    ha_momentum,
    momentum_12_1,
    tradeable_momentum_window,
)

BARS = MOMENTUM_LOOKBACK + MOMENTUM_SKIP + 5


def _index(periods: int = BARS) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=periods)


def _traded(periods: int = BARS) -> pd.Series:
    return pd.Series(1_000.0, index=_index(periods))


def test_a_stub_denominator_is_rejected() -> None:
    """The observed failure: $4.26 over a $0.0001 stub scored +425,900%."""
    close = pd.Series(4.26, index=_index())
    close.iloc[:150] = 0.0001  # dormant grey-market stub, then real quotes
    volume = _traded()
    volume.iloc[:150] = 0.0

    raw = close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1.0
    assert raw.iloc[-1] > 40_000  # what the ungated formula produced
    assert np.isnan(momentum_12_1(close, volume).iloc[-1])


def test_a_stub_numerator_is_rejected() -> None:
    close = pd.Series(4.26, index=_index())
    close.iloc[-200:] = 0.0001
    assert np.isnan(momentum_12_1(close, _traded()).iloc[-1])


def test_a_real_price_is_scored_unchanged() -> None:
    close = pd.Series(np.linspace(10.0, 20.0, BARS), index=_index())
    value = momentum_12_1(close, _traded()).iloc[-1]
    expected = close.iloc[-1 - MOMENTUM_SKIP] / close.iloc[-1 - MOMENTUM_LOOKBACK] - 1.0
    assert value == expected


def test_the_floor_sits_at_one_tick() -> None:
    """A cent is the minimum tick, so no real quote is excluded."""
    close = pd.Series(MIN_TRADEABLE_PRICE, index=_index())
    assert bool(tradeable_momentum_window(close, _traded()).iloc[-1])
    below = pd.Series(MIN_TRADEABLE_PRICE - 1e-6, index=_index())
    assert not bool(tradeable_momentum_window(below, _traded()).iloc[-1])


def test_a_mostly_untraded_window_is_rejected() -> None:
    """Both legs are above a cent, but the window in between is not a market."""
    close = pd.Series(np.linspace(5.0, 9.0, BARS), index=_index())
    volume = _traded()
    volume.iloc[:-30] = 0.0  # carried forward for all but the last six weeks
    assert not np.isnan(momentum_12_1(close).iloc[-1])  # price floor alone passes
    assert np.isnan(momentum_12_1(close, volume).iloc[-1])


def test_a_thinly_traded_but_real_name_survives() -> None:
    """The liquidity gate is permissive by design: a real microcap survives.

    Two sessions in three, not one in two: an exactly-even split lands on the
    threshold, where an odd-length window rounds to just under it. The point of
    the test is the wide margin below a liquid name, not the boundary.
    """
    close = pd.Series(np.linspace(5.0, 9.0, BARS), index=_index())
    volume = _traded()
    volume.iloc[::3] = 0.0  # quiet one session in three
    assert not np.isnan(momentum_12_1(close, volume).iloc[-1])


def test_volume_is_optional_and_the_price_floor_still_applies() -> None:
    """A close-only frame answers one question, so the gate applies just that one."""
    close = pd.Series(4.26, index=_index())
    close.iloc[:150] = 0.0001
    assert np.isnan(momentum_12_1(close).iloc[-1])


def test_ha_momentum_carries_the_same_gate() -> None:
    close = pd.Series(4.26, index=_index())
    close.iloc[:150] = 0.0001
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}
    )
    value = ha_momentum(
        frame["open"], frame["high"], frame["low"], frame["close"], _traded()
    )
    assert np.isnan(value.iloc[-1])
