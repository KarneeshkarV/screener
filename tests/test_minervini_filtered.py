"""Tests for the filtered Mark Minervini entry-quality variants."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.pine import parse
from screener.strategies.plugins import minervini_filtered as mf
from screener.strategies.spec import (
    PrepareCtx,
    discover_plugins,
    resolve_strategy_spec,
)
from tests.conftest import StubPriceFetcher, make_bars

MQ_STRATEGIES = ["mq_us1", "mq_us2", "mq_us3", "mq_in1", "mq_in2", "mq_in3"]
LEGACY_STRATEGIES = [
    "minervini_leaders",
    "minervini_growth_us",
    "minervini_growth_in",
    "minervini_pro_us",
    "minervini_pro_in",
]


def test_all_variants_registered():
    discover_plugins()
    for name in MQ_STRATEGIES + LEGACY_STRATEGIES:
        assert resolve_strategy_spec(name) is not None, name


def test_variant_entry_and_exit_expressions_parse():
    discover_plugins()
    for name in MQ_STRATEGIES:
        spec = resolve_strategy_spec(name)
        # Both expressions must be valid Pine so the backtester can evaluate them.
        parse(spec.entry)
        parse(spec.exit)


def test_template_appends_extra_clause():
    base = mf._template(70, 0.75)
    tightened = mf._template(70, 0.75, "pe_ttm >= 25")
    assert "rs_rank >= 70" in base
    assert "highest(high, 252) * 0.75" in base
    assert base + " and pe_ttm >= 25" == tightened


def test_add_quality_features_matches_definitions():
    frame = make_bars(n=320, seed=3, drift=0.05)
    out = mf._add_quality_features(frame.copy())

    for col in (
        "ext_above_sma50",
        "sma50_150_spread",
        "sma150_200_spread",
        "mom_63d",
        "mom_126d",
        "vol_20d_ann",
    ):
        assert col in out.columns

    close = frame["close"].astype(float)
    s50 = close.rolling(50).mean()
    s150 = close.rolling(150).mean()
    s200 = close.rolling(200).mean()
    pd.testing.assert_series_equal(
        out["ext_above_sma50"], (close / s50 - 1.0), check_names=False
    )
    pd.testing.assert_series_equal(
        out["sma50_150_spread"], (s50 / s150 - 1.0), check_names=False
    )
    pd.testing.assert_series_equal(
        out["sma150_200_spread"], (s150 / s200 - 1.0), check_names=False
    )
    pd.testing.assert_series_equal(
        out["mom_63d"], (close / close.shift(63) - 1.0), check_names=False
    )
    pd.testing.assert_series_equal(
        out["mom_126d"], (close / close.shift(126) - 1.0), check_names=False
    )
    expected_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    pd.testing.assert_series_equal(out["vol_20d_ann"], expected_vol, check_names=False)


def test_add_quality_features_leading_window_is_nan():
    frame = make_bars(n=320, seed=7)
    out = mf._add_quality_features(frame.copy())
    # 200-day SMA is undefined before 200 bars; 63-day momentum before 63 bars.
    assert np.isnan(out["sma150_200_spread"].iloc[100])
    assert np.isnan(out["mom_63d"].iloc[10])
    assert not np.isnan(out["ext_above_sma50"].iloc[-1])


def test_prep_attaches_rs_rank_and_quality_features():
    bars = {"AAA": make_bars(n=320, seed=1, drift=0.05)}
    ctx = PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv=bars,
        price_panel={},
        tv_symbols=["AAA"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        fetcher=StubPriceFetcher({}),
        warnings=[],
    )
    out = mf._prep(ctx)
    frame = out["AAA"]
    # rs_rank / sma200_up_1m come from the shared Minervini prep hook ...
    assert "rs_rank" in frame.columns
    assert "sma200_up_1m" in frame.columns
    # ... and the entry-quality features are layered on top.
    assert "ext_above_sma50" in frame.columns
    assert "vol_20d_ann" in frame.columns
