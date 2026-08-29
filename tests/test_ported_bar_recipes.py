"""The snapshot scorers ported to bars: RSI, relative volume and Perf.Y.

These three used to read a TradingView snapshot column, which carries only
today's value and so cannot be replayed through history. Computing them from
bars is what lets one number serve both the screen and the backtest.

The tests below pin the two things the port has to get right: the *unit* must
match what the vendor column reported, because the snapshot scorers compare
against thresholds calibrated in those units, and the value at bar ``t`` must
not move when later bars change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.factors import get_price_score, score_bars
from screener.factors.recipes import (
    PERF_Y_LOOKBACK,
    RSI_PERIOD,
    RVOL_WINDOW,
    perf_y,
    relative_volume_10d,
    rsi_14,
)
from screener.indicators.plugins.rsi import rsi as wilder_rsi


def _bars(n: int = 400) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(
                rng.integers(1_000, 5_000, n).astype(float), index=index
            ),
        }
    )


class TestUnits:
    def test_rsi_is_the_one_wilder_implementation_on_the_0_100_scale(self):
        bars = _bars()
        expected = wilder_rsi(bars["close"].to_numpy(dtype=float), RSI_PERIOD)

        got = rsi_14(bars["close"])

        np.testing.assert_allclose(got.to_numpy(), expected, equal_nan=True)
        valid = got.dropna()
        assert valid.between(0.0, 100.0).all()

    def test_flat_volume_is_exactly_one(self):
        index = pd.bdate_range("2020-01-01", periods=30)

        got = relative_volume_10d(pd.Series(1_000.0, index=index))

        assert got.iloc[RVOL_WINDOW - 1 :].eq(1.0).all()
        assert got.iloc[: RVOL_WINDOW - 1].isna().all()

    def test_relative_volume_of_a_doubled_bar_against_a_flat_history(self):
        index = pd.bdate_range("2020-01-01", periods=30)
        volume = pd.Series(1_000.0, index=index)
        volume.iloc[-1] = 2_000.0

        got = relative_volume_10d(volume)

        # Window mean = (9 * 1000 + 2000) / 10 = 1100, so 2000 / 1100.
        assert got.iloc[-1] == pytest.approx(2_000.0 / 1_100.0)

    def test_perf_y_is_a_percent_not_a_fraction(self):
        index = pd.bdate_range("2020-01-01", periods=PERF_Y_LOOKBACK + 1)
        close = pd.Series(100.0, index=index)
        close.iloc[-1] = 110.0

        got = perf_y(close)

        assert got.iloc[-1] == pytest.approx(10.0)


class TestCausality:
    @pytest.mark.parametrize("name", ["rsi_14", "relative_volume_10d", "perf_y"])
    def test_a_score_at_t_does_not_move_when_later_bars_change(self, name):
        spec = get_price_score(name)
        bars = _bars()
        cutoff = bars.index[300]
        perturbed = bars.copy()
        mask = perturbed.index > cutoff
        for column in ("open", "high", "low", "close"):
            perturbed.loc[mask, column] = perturbed.loc[mask, column] * 1000.0
        perturbed.loc[mask, "volume"] = 1.0

        pd.testing.assert_series_equal(
            score_bars(spec, bars).loc[:cutoff],
            score_bars(spec, perturbed).loc[:cutoff],
        )


class TestRegistration:
    @pytest.mark.parametrize(
        ("name", "aux_column", "lookback"),
        [
            ("rsi_14", "rsi_14", RSI_PERIOD),
            ("relative_volume_10d", "rvol_10d", RVOL_WINDOW),
            ("perf_y", "perf_y", PERF_Y_LOOKBACK),
        ],
    )
    def test_each_port_is_registered_with_its_aux_column_and_warmup(
        self, name, aux_column, lookback
    ):
        spec = get_price_score(name)

        assert spec.aux_column == aux_column
        assert spec.required_lookback == lookback
        # These three are *inputs* to a shaped score, not levels that make a
        # name tradeable, so none of them declares an eligibility floor.
        assert spec.eligible_above is None

    @pytest.mark.parametrize("name", ["rsi_14", "relative_volume_10d", "perf_y"])
    def test_the_declared_warmup_covers_the_real_nan_run(self, name):
        spec = get_price_score(name)

        scores = score_bars(spec, _bars())

        leading_nans = int(scores.isna().to_numpy().argmin())
        assert leading_nans <= spec.required_lookback

    def test_a_volumeless_frame_scores_relative_volume_as_ineligible(self):
        bars = _bars().drop(columns=["volume"])

        scores = score_bars(get_price_score("relative_volume_10d"), bars)

        assert scores.isna().all()
