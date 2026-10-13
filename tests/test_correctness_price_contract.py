"""Explicit adjustment contracts and incomplete overlap regressions."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from screener.backtester import data, price_cache, price_frames


def yahoo_frame(index, close):
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1000.0,
            "Adj Close": close,
            "Stock Splits": 0.0,
        },
        index=index,
    )


def test_pending_split_adjustment_is_idempotent():
    raw = yahoo_frame(pd.bdate_range("2024-01-01", periods=3), [100, 50, 50])
    raw.iloc[1, raw.columns.get_loc("Stock Splits")] = 2.0
    normalized = price_frames.normalize_price_frame(raw)
    first = price_frames.apply_splits_only_adjustment({"AAA": normalized})
    second = price_frames.apply_splits_only_adjustment(first)
    pd.testing.assert_frame_equal(first["AAA"], second["AAA"])
    assert first["AAA"].close.tolist() == [50.0, 50.0, 50.0]


@pytest.mark.parametrize("auto_adjust", [False, True])
def test_yahoo_rejects_legacy_price_cache(tmp_path, monkeypatch, auto_adjust):
    index = pd.bdate_range("2024-01-01", "2024-01-05")
    legacy_key = "AAA__full_turnover_v2" if auto_adjust else "AAA__raw"
    stale = price_frames.normalize_price_frame(yahoo_frame(index, 999.0))
    price_cache.save_cached_frame(legacy_key, stale, tmp_path)
    monkeypatch.setattr(data, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        data._optional,
        "load",
        lambda _: SimpleNamespace(download=lambda *a, **k: yahoo_frame(index, 50.0)),
    )
    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, auto_adjust=auto_adjust)
    result = fetcher.fetch(["AAA"], index[0].date(), index[-1].date())["AAA"]
    assert result.close.eq(50.0).all()


def test_full_replacement_does_not_need_two_bar_overlap(tmp_path, monkeypatch):
    calls = []

    def download(ticker, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return yahoo_frame(pd.DatetimeIndex(["2024-01-05"]), 100.0)
        return yahoo_frame(
            pd.bdate_range(kwargs["start"], kwargs["end"], inclusive="left"), 50.0
        )

    monkeypatch.setattr(data, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        data._optional, "load", lambda _: SimpleNamespace(download=download)
    )
    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path)
    fetcher.fetch(["AAA"], date(2024, 1, 5), date(2024, 1, 5))
    result = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 12))["AAA"]
    assert result.close.eq(50.0).all()
    assert len(calls) == 2


def test_intraday_extension_without_usable_overlap_reloads_full_window(
    tmp_path, monkeypatch
):
    calls = []

    def download(ticker, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            index = pd.DatetimeIndex(["2024-01-05 14:00", "2024-01-05 15:00"])
            return yahoo_frame(index, 100.0)
        if len(calls) == 2:
            return yahoo_frame(
                pd.DatetimeIndex(["2024-01-15 14:00", "2024-01-15 15:00"]), 50.0
            )
        return yahoo_frame(
            pd.DatetimeIndex(
                [
                    "2024-01-05 14:00",
                    "2024-01-05 15:00",
                    "2024-01-15 14:00",
                    "2024-01-15 15:00",
                ]
            ),
            50.0,
        )

    monkeypatch.setattr(data, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        data._optional, "load", lambda _: SimpleNamespace(download=download)
    )
    monkeypatch.setattr(
        data.YFinancePriceFetcher,
        "_intraday_chunks",
        lambda self, start, end: [(start, end)],
    )
    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, interval="1h")
    fetcher.fetch(["AAA"], date(2024, 1, 5), date(2024, 1, 5))
    result = fetcher.fetch(["AAA"], date(2024, 1, 5), date(2024, 1, 15))["AAA"]
    assert result.close.eq(50.0).all()
    assert len(calls) == 3
