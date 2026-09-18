"""Every price-ratio and price-dispersion factor rejects an untradeable window.

The 12-1 momentum recipe was gated first, after a dormant OTC line carried at a
flat $0.0001 stub divided a $4.26 close by it and topped a 13,326-name field.
The gate belonged on every factor of that shape, not only on the one where the
defect was noticed, so these tests pin the behaviour factor by factor: a stub
denominator, and a series the vendor carried forward through months of no
trading, must both produce "no reading" rather than a number.

Each test builds the smallest series that exhibits the defect and asserts the
factor is NaN (or ``None``) there while an otherwise identical real series
still gets a value. Asserting the real series keeps a gate that rejects
everything from passing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.factors.recipes import PERF_Y_LOOKBACK, perf_y
from screener.relative_strength import (
    RS_RANK_WINDOW,
    RS_RATIO_WINDOW,
    RS_SPREAD_WINDOW,
    relative_strength_rank,
    relative_strength_ratio,
    relative_strength_spread,
)
from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.minervini_filtered import _add_quality_features
from screener.tradeable import (
    MIN_TRADEABLE_PRICE,
    tradeable_span,
    tradeable_window,
)

_BARS = 400


def _index(n: int = _BARS) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _real(n: int = _BARS) -> pd.Series:
    """A plain rising series well above the price floor."""
    return pd.Series(np.linspace(10.0, 40.0, n), index=_index(n))


def _traded(n: int = _BARS) -> pd.Series:
    return pd.Series(1_000_000.0, index=_index(n))


def _stub_then_real(n: int = _BARS) -> pd.Series:
    """A sub-cent stub for the first half, a real quote for the second.

    This is the shape that manufactures the defect: any window whose far leg
    lands in the stub half divides a real price by $0.0001.
    """
    values = np.concatenate([np.full(n // 2, 0.0001), np.full(n - n // 2, 4.26)])
    return pd.Series(values, index=_index(n))


def _mostly_untraded(n: int = _BARS) -> pd.Series:
    """Volume that is zero on four sessions in every five."""
    volume = np.zeros(n)
    volume[::5] = 1_000_000.0
    return pd.Series(volume, index=_index(n))


class TestGatePrimitives:
    def test_window_rejects_a_stub_denominator(self):
        close = _stub_then_real()
        gate = tradeable_window(close, _traded(), lookback=PERF_Y_LOOKBACK)
        assert not bool(gate.iloc[-1])

    def test_window_accepts_a_real_traded_series(self):
        gate = tradeable_window(_real(), _traded(), lookback=PERF_Y_LOOKBACK)
        assert bool(gate.iloc[-1])

    def test_window_rejects_a_series_that_mostly_did_not_trade(self):
        gate = tradeable_window(_real(), _mostly_untraded(), lookback=PERF_Y_LOOKBACK)
        assert not bool(gate.iloc[-1])

    def test_span_rejects_a_stub_anywhere_inside_the_window(self):
        """A span factor reads every close, so one stub bar is enough.

        The endpoints here are both real prices - only a bar in the middle is a
        stub - which is exactly the case ``tradeable_window`` would pass and a
        dispersion measure must not.
        """
        close = _real().copy()
        close.iloc[-10] = 0.0001
        assert bool(tradeable_window(close, _traded(), lookback=50).iloc[-1])
        assert not bool(tradeable_span(close, _traded(), window=50).iloc[-1])

    def test_price_floor_is_inclusive_at_one_tick(self):
        at_floor = pd.Series(MIN_TRADEABLE_PRICE, index=_index())
        below = pd.Series(MIN_TRADEABLE_PRICE - 1e-6, index=_index())
        assert bool(tradeable_window(at_floor, _traded(), lookback=50).iloc[-1])
        assert not bool(tradeable_window(below, _traded(), lookback=50).iloc[-1])

    def test_without_volume_only_the_price_rule_applies(self):
        """A close-only frame cannot answer the liquidity question.

        It must still answer the price one rather than failing open.
        """
        assert bool(tradeable_window(_real(), None, lookback=50).iloc[-1])
        assert not bool(
            tradeable_window(_stub_then_real(), None, lookback=PERF_Y_LOOKBACK).iloc[-1]
        )


class TestPerfY:
    def test_stub_denominator_yields_no_reading(self):
        got = perf_y(_stub_then_real(), _traded())
        assert pd.isna(got.iloc[-1])

    def test_real_series_still_scores(self):
        close = _real()
        expected = (close.iloc[-1] / close.iloc[-1 - PERF_Y_LOOKBACK] - 1.0) * 100.0
        got = perf_y(close, _traded())
        assert got.iloc[-1] == pytest.approx(expected)

    def test_untraded_series_yields_no_reading(self):
        got = perf_y(_real(), _mostly_untraded())
        assert pd.isna(got.iloc[-1])


class TestRelativeStrength:
    def test_ratio_rejects_a_stub_leg(self):
        """Checked at a bar whose 55-session window straddles the stub seam.

        The stub half is far longer than the ratio window, so by the last bar
        both legs are real prices again and the reading is legitimate. The
        defect lives on the bars whose far leg is still in the stub half.
        """
        got = relative_strength_ratio(
            _stub_then_real(), _real(), stock_volume=_traded()
        )
        assert pd.isna(got.iloc[_BARS // 2 + RS_RATIO_WINDOW - 1])

    def test_ratio_keeps_a_real_reading(self):
        got = relative_strength_ratio(_real(), _real(), stock_volume=_traded())
        assert got.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_rank_drops_a_stub_name_instead_of_ranking_it_first(self):
        """The percentile is competitive: a bad name must not take a slot.

        Ungated, the stub name's five-figure return is the largest in the
        field and ranks 100. Gated, it is absent and the real names are ranked
        against each other alone.
        """
        # A constant multiple would leave REAL_B's *return* identical to
        # REAL_A's and the two would tie mid-field, so give it a genuinely
        # weaker trajectory.
        closes = {
            "REAL_A": _real(),
            "REAL_B": pd.Series(np.linspace(10.0, 12.0, _BARS), index=_index()),
            "STUB": _stub_then_real(),
        }
        volumes = {name: _traded() for name in closes}
        ranks = relative_strength_rank(
            closes, volumes_by_symbol=volumes, window=RS_RANK_WINDOW
        )
        last = ranks.iloc[-1]
        assert pd.isna(last["STUB"])
        assert last[["REAL_A", "REAL_B"]].notna().all()
        assert last.max() == pytest.approx(100.0)

    def test_rank_without_volume_still_drops_a_stub_name(self):
        closes = {"REAL": _real(), "STUB": _stub_then_real()}
        ranks = relative_strength_rank(closes, window=RS_RANK_WINDOW)
        assert pd.isna(ranks.iloc[-1]["STUB"])

    def test_spread_is_unknown_rather_than_a_number_on_a_stub(self):
        stub = _stub_then_real()
        # Put the spread window across the stub-to-real seam.
        seam = stub.iloc[: _BARS // 2 + RS_SPREAD_WINDOW]
        got = relative_strength_spread(
            seam,
            _real().reindex(seam.index),
            stock_volume=_traded().reindex(seam.index),
        )
        assert got is None

    def test_spread_returns_a_number_on_real_bars(self):
        got = relative_strength_spread(_real(), _real(), stock_volume=_traded())
        assert got == pytest.approx(0.0, abs=1e-9)


class TestLowVolatility:
    def test_a_flat_stub_is_not_the_calmest_stock(self):
        """Zero variance is the failure, not the goal.

        A line pinned at $0.0001 has a realized volatility of exactly zero, so
        an ungated ``rank_score = -vol`` hands it the first slot in the
        portfolio every time.
        """
        flat_stub = pd.Series(0.0001, index=_index())
        got = realized_volatility(flat_stub, pd.Series(0.0, index=_index()))
        assert pd.isna(got.iloc[-1])

    def test_real_series_still_reports_volatility(self):
        rng = np.random.default_rng(0)
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, _BARS))), index=_index()
        )
        got = realized_volatility(close, _traded())
        assert got.iloc[-1] > 0.0


class TestMinerviniQualityFeatures:
    def _frame(self, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"close": close, "volume": volume})

    @pytest.mark.parametrize(
        "column",
        [
            "ext_above_sma50",
            "sma50_150_spread",
            "sma150_200_spread",
            "mom_63d",
            "mom_126d",
            "vol_20d_ann",
        ],
    )
    def test_every_quality_feature_is_gated(self, column: str):
        stub = pd.Series(0.0001, index=_index())
        frame = _add_quality_features(self._frame(stub, pd.Series(0.0, index=_index())))
        assert pd.isna(frame[column].iloc[-1])

    @pytest.mark.parametrize(
        "column",
        [
            "ext_above_sma50",
            "sma50_150_spread",
            "sma150_200_spread",
            "mom_63d",
            "mom_126d",
            "vol_20d_ann",
        ],
    )
    def test_real_bars_keep_every_quality_feature(self, column: str):
        frame = _add_quality_features(self._frame(_real(), _traded()))
        assert not pd.isna(frame[column].iloc[-1])
