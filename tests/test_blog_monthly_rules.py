"""Verify monthly research signals before portfolio execution."""

from datetime import date

import numpy as np
import pandas as pd

from scripts.blog_monthly_rules import prepare_blog_monthly
from screener.strategies.spec import PrepareCtx


def test_blog_monthly_signals_only_at_month_end():
    index = pd.bdate_range("2018-01-01", "2021-12-31")
    close = pd.Series(np.arange(len(index)) + 100.0, index=index)
    bars = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "volume": 100,
        }
    )
    ctx = PrepareCtx(
        market="india",
        benchmark="TEST",
        bars_by_tv={"TEST": bars},
        price_panel={"TEST": bars},
        tv_symbols=["TEST"],
        start=date(2020, 1, 1),
        end=date(2021, 12, 31),
        fetcher=object(),
        warnings=[],
    )
    frame = prepare_blog_monthly(ctx)["TEST"]
    ends = pd.Series(index, index=index).groupby(index.to_period("M")).last()
    entries = frame.index[frame.alvarez_monthly_momentum10_entry.gt(0)]
    assert set(entries).issubset(set(ends))
    assert not frame.alvarez_monthly_momentum10_entry.loc[:"2018-10-31"].any()
    assert frame.loc["2018-11-30", "alvarez_monthly_momentum10_entry"] == 1
    assert frame.alvarez_monthly_momentum10_exit.sum() == 0
    # Rotation liquidates one session before next-month entry, not on entry day.
    assert frame.loc["2020-01-30", "monthly_rotate_exit"] == 1
    assert frame.loc["2020-01-31", "monthly_rotate_exit"] == 0


def test_blog_monthly_future_prices_do_not_change_past_signals():
    index = pd.bdate_range("2018-01-01", "2021-12-31")
    close = pd.Series(np.arange(len(index)) + 100.0, index=index)

    def prepare(prices):
        bars = pd.DataFrame({"close": prices})
        ctx = PrepareCtx(
            market="us",
            benchmark="TEST",
            bars_by_tv={"TEST": bars},
            price_panel={"TEST": bars},
            tv_symbols=["TEST"],
            start=date(2020, 1, 1),
            end=date(2021, 12, 31),
            fetcher=object(),
            warnings=[],
        )
        return prepare_blog_monthly(ctx)["TEST"]

    original = prepare(close)
    close.loc["2021-01-01":] *= 0.1
    altered = prepare(close)
    pd.testing.assert_frame_equal(
        original.loc[:"2020-12-31"], altered.loc[:"2020-12-31"]
    )
