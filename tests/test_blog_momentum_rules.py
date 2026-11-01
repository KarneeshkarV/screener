"""Check source formulas and causality of the daily blog research adapters."""

import numpy as np
import pandas as pd

from scripts.blog_momentum_rules import (
    blog_momentum_columns,
    prepare_chan_log_ma,
    prepare_blog_trend_filters,
)
from screener.strategies.spec import PrepareCtx


def test_chan_full_year_return_does_not_skip_recent_month():
    close = pd.Series(np.arange(1, 301, dtype=float))
    result = blog_momentum_columns(pd.DataFrame({"close": close}))
    assert np.isnan(result.loc[251, "chan_return_12m"])
    assert result.loc[252, "chan_return_12m"] == 252
    assert result.loc[299, "chan_return_12m"] == close[299] / close[47] - 1


def test_chan_weighted_momentum_formula():
    close = pd.Series(np.exp(np.arange(300) / 1000))
    result = blog_momentum_columns(pd.DataFrame({"close": close}))
    expected = (
        sum(
            weight * (close[299] / close[299 - window] - 1)
            for weight, window in ((12, 21), (4, 63), (2, 126), (1, 252))
        )
        / 19
    )
    assert np.isclose(result.loc[299, "chan_weighted_13612"], expected)


def test_chan_crossover_uses_log_prices():
    close = pd.Series(np.arange(1, 301, dtype=float))
    result = blog_momentum_columns(pd.DataFrame({"close": close}))
    expected = np.log(close).rolling(7).mean() - np.log(close).rolling(10).mean()
    pd.testing.assert_series_equal(
        result["chan_log_ma_7_10"], expected, check_names=False
    )


def test_blog_momentum_has_no_future_price_dependency():
    bars = pd.DataFrame({"close": np.exp(np.arange(400) / 1000)})
    original = blog_momentum_columns(bars)
    bars.loc[301:, "close"] *= 10
    changed = blog_momentum_columns(bars)
    pd.testing.assert_frame_equal(original.loc[:300], changed.loc[:300])


def test_chan_us_expression_exits_use_previous_completed_signal():
    from datetime import date
    from screener.backtester.data import build_price_fetcher

    bars = pd.DataFrame({"close": np.exp(np.arange(400) / 1000)})
    ctx = PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv={"SPY": bars},
        price_panel={"SPY": bars},
        tv_symbols=["SPY"],
        start=date(2020, 1, 1),
        end=date(2025, 12, 31),
        fetcher=build_price_fetcher(),
        warnings=[],
    )
    prepared = prepare_chan_log_ma(ctx)["SPY"]
    pd.testing.assert_series_equal(
        prepared["chan_log_ma_7_10_exit"],
        prepared["chan_log_ma_7_10"].shift(1),
        check_names=False,
    )
    india = prepare_chan_log_ma(ctx.model_copy(update={"market": "india"}))["SPY"]
    pd.testing.assert_series_equal(
        india["chan_log_ma_7_10_exit"],
        india["chan_log_ma_7_10"],
        check_names=False,
    )


def test_frozen_blog_bars_replay_without_provider(tmp_path):
    from datetime import date
    from scripts.run_blog_momentum import FrozenStudyFetcher

    class OfflineProvider:
        def fetch(self, tickers, start, end):
            raise AssertionError("Frozen replay must not call a provider")

    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [10, 10],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    frame.to_parquet(tmp_path / "SPY.parquet")
    frozen = FrozenStudyFetcher(OfflineProvider(), tmp_path)
    result = frozen.fetch(["SPY"], date(2020, 1, 3), date(2020, 1, 3))
    pd.testing.assert_frame_equal(result["SPY"], frame.iloc[1:])


def test_frozen_blog_bars_remove_zero_volume_placeholders(tmp_path):
    from datetime import date
    from scripts.run_blog_momentum import FrozenStudyFetcher

    class OfflineProvider:
        def fetch(self, tickers, start, end):
            raise AssertionError("Unexpected acquisition")

    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [0, 10],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    frame.to_parquet(tmp_path / "SPY.parquet")
    result = FrozenStudyFetcher(OfflineProvider(), tmp_path).fetch(
        ["SPY"], date(2020, 1, 1), date(2020, 1, 3)
    )
    pd.testing.assert_frame_equal(result["SPY"], frame.iloc[1:])


def test_alvarez_confirmation_and_unger_slope_formulas():
    from datetime import date
    from screener.backtester.data import build_price_fetcher

    close = pd.Series(np.arange(1, 301, dtype=float))
    bars = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1})
    ctx = PrepareCtx(
        market="india",
        benchmark="^NSEI",
        bars_by_tv={"TEST": bars},
        price_panel={"TEST": bars},
        tv_symbols=["TEST"],
        start=date(2020, 1, 1),
        end=date(2025, 12, 31),
        fetcher=build_price_fetcher(),
        warnings=[],
    )
    result = prepare_blog_trend_filters(ctx)["TEST"]
    assert result.loc[202, "alvarez_ma200_confirm3"] == 1
    assert result.loc[200, "alvarez_ma200_confirm3"] == 0
    expected_slope = close.rolling(200).mean().diff(50)
    pd.testing.assert_series_equal(
        result["unger_sma200_slope50"], expected_slope, check_names=False
    )
    expected_breakout = close - bars.high.rolling(5).max().shift(1)
    pd.testing.assert_series_equal(
        result["unger_donchian5"], expected_breakout, check_names=False
    )


def test_blog_momentum_invalid_prices_do_not_produce_rank():
    bars = pd.DataFrame({"close": np.ones(300)})
    bars.loc[299, "close"] = 0
    result = blog_momentum_columns(bars)
    assert np.isnan(result.loc[299, "chan_return_12m"])
    assert np.isnan(result.loc[299, "chan_weighted_13612"])
