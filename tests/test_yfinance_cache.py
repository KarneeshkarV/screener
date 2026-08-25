from __future__ import annotations

import os
import threading
import time
from datetime import date

import pandas as pd
import pytest

from screener.backtester import data as data_module
from screener.backtester.data import YFinancePriceFetcher, _load_cached, _save_cache


def _plain_bars(start, end, base: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1))
    return pd.DataFrame(
        {
            "Open": [base + i for i in range(len(idx))],
            "High": [base + i + 1 for i in range(len(idx))],
            "Low": [base + i - 1 for i in range(len(idx))],
            "Close": [base + i + 0.5 for i in range(len(idx))],
            "Volume": [1000 + i for i in range(len(idx))],
        },
        index=idx,
    )


def _download_frame(tickers, start, end) -> pd.DataFrame:
    if isinstance(tickers, str):
        return _plain_bars(start, end)
    pieces = []
    for offset, ticker in enumerate(tickers):
        frame = _plain_bars(start, end, base=100.0 + offset * 10)
        frame.columns = pd.MultiIndex.from_product([[ticker], frame.columns])
        pieces.append(frame)
    return pd.concat(pieces, axis=1)


def test_yfinance_fetcher_batches_uncached_tickers(tmp_path, monkeypatch):
    import yfinance as yf

    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        batch = tickers.split() if isinstance(tickers, str) else list(tickers)
        return _download_frame(batch, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, batch_size=50)
    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 10))

    assert len(calls) == 1
    assert calls[0][0] == "AAA BBB"
    assert calls[0][1]["timeout"] == data_module.YFINANCE_TIMEOUT_SECONDS
    assert set(out) == {"AAA", "BBB"}
    assert not out["AAA"].empty
    assert not out["BBB"].empty


def test_yfinance_fetcher_uses_full_cache_hit(tmp_path, monkeypatch):
    import yfinance as yf

    calls = {"count": 0}

    def fake_download(tickers, **kwargs):
        calls["count"] += 1
        batch = tickers.split() if isinstance(tickers, str) else list(tickers)
        return _download_frame(batch, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    first = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 10))
    second = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 10))

    assert calls["count"] == 1
    assert first["AAA"].equals(second["AAA"])


def test_atomic_cache_failure_preserves_previous_frame(tmp_path, monkeypatch):
    original = _plain_bars(date(2024, 1, 1), date(2024, 1, 5)).rename(columns=str.lower)
    _save_cache("AAA", original, tmp_path)

    def fail_write(self, path, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    replacement = original.assign(close=999.0)

    _save_cache("AAA", replacement, tmp_path)

    pd.testing.assert_frame_equal(
        _load_cached("AAA", tmp_path), original, check_freq=False
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_yfinance_fetcher_fetches_only_missing_tail(tmp_path, monkeypatch):
    import yfinance as yf

    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((pd.Timestamp(kwargs["start"]), pd.Timestamp(kwargs["end"])))
        batch = tickers.split() if isinstance(tickers, str) else list(tickers)
        return _download_frame(batch, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 12))

    assert calls[0] == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-06"))
    assert calls[1][0] == pd.Timestamp("2024-01-06")
    assert calls[1][1] == pd.Timestamp("2024-01-13")
    assert out["AAA"].index.min() == pd.Timestamp("2024-01-01")
    assert out["AAA"].index.max() == pd.Timestamp("2024-01-12")


def test_yfinance_fetcher_downloads_batches_in_parallel(tmp_path, monkeypatch):
    import yfinance as yf

    lock = threading.Lock()
    active = {"now": 0, "peak": 0}

    def fake_download(tickers, **kwargs):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        batch = tickers.split() if isinstance(tickers, str) else list(tickers)
        return _download_frame(batch, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(
        data_module,
        "call_with_resilience",
        lambda provider, operation, func, *, fallback: func(),
    )

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, batch_size=1, max_workers=4)
    out = fetcher.fetch(["AAA", "BBB", "CCC"], date(2024, 1, 1), date(2024, 1, 10))

    assert set(out) == {"AAA", "BBB", "CCC"}
    assert all(not out[ticker].empty for ticker in out)
    assert active["peak"] >= 2, "batches should overlap when more than one job exists"


def test_yfinance_stale_recent_cache_refreshes_and_merges_tail(tmp_path, monkeypatch):
    import yfinance as yf

    today = date.today()
    start = today - pd.Timedelta(days=5)
    cached = _plain_bars(start, today + pd.Timedelta(days=1))
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    _save_cache("AAA", cached.rename(columns=str.lower), tmp_path)
    cache_path = tmp_path / "AAA.parquet"
    old_mtime = time.time() - 7200
    os.utime(cache_path, (old_mtime, old_mtime))
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "Open": [999.0],
                "High": [1000.0],
                "Low": [998.0],
                "Close": [999.5],
                "Volume": [5000],
            },
            index=pd.DatetimeIndex([today]),
        )

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setenv("SCREENER_PRICE_TAIL_TTL_SECONDS", "3600")

    out = fetcher.fetch(["AAA"], start, today)["AAA"]

    assert len(calls) == 1
    assert calls[0]["start"] == cached.index.max() - pd.Timedelta(days=7)
    assert out.loc[pd.Timestamp(today), "close"] == 999.5


def test_yfinance_tail_refresh_skips_fresh_and_historical_caches(tmp_path, monkeypatch):
    import yfinance as yf

    calls = []
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("SCREENER_PRICE_TAIL_TTL_SECONDS", "3600")

    today = date.today()
    recent_start = today - pd.Timedelta(days=5)
    recent = _plain_bars(recent_start, today + pd.Timedelta(days=1))
    _save_cache("RECENT", recent.rename(columns=str.lower), tmp_path)
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    fetcher.fetch(["RECENT"], recent_start, today)

    historical_start = date(2024, 1, 1)
    historical_end = date(2024, 1, 5)
    historical = _plain_bars(historical_start, date(2024, 1, 6))
    _save_cache("OLD", historical.rename(columns=str.lower), tmp_path)
    old_mtime = time.time() - 7200
    os.utime(tmp_path / "OLD.parquet", (old_mtime, old_mtime))
    fetcher.fetch(["OLD"], historical_start, historical_end)

    assert calls == []


def test_yfinance_refresh_merges_into_stored_history_instead_of_truncating_it(
    tmp_path, monkeypatch
):
    """--refresh re-downloads its window but must keep bars outside it on disk."""
    import yfinance as yf

    wide = _plain_bars(date(2018, 1, 1), date(2024, 6, 1)).rename(columns=str.lower)
    _save_cache("AAA", wide, tmp_path)

    narrow_start, narrow_end = date(2024, 4, 1), date(2024, 4, 30)
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((pd.Timestamp(kwargs["start"]), pd.Timestamp(kwargs["end"])))
        return _download_frame(tickers, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, refresh=True)
    out = fetcher.fetch(["AAA"], narrow_start, narrow_end)["AAA"]

    # The refresh forced a full re-download of exactly the requested window.
    assert len(calls) == 1
    assert calls[0] == (
        pd.Timestamp(narrow_start),
        pd.Timestamp(narrow_end) + pd.Timedelta(days=1),
    )

    stored = _load_cached("AAA", tmp_path)
    assert stored.index.min() == wide.index.min()
    assert stored.index.max() == wide.index.max()

    # Overlapping dates carry the freshly downloaded values, not the old ones.
    first_fresh = out.index[0]
    fresh_close = float(out.loc[first_fresh, "close"])
    assert fresh_close != float(wide.loc[first_fresh, "close"])
    assert float(stored.loc[first_fresh, "close"]) == pytest.approx(fresh_close)

    # Bars outside the refreshed window stay as they were.
    outside = wide.index[~wide.index.isin(out.index)]
    sample = outside[len(outside) // 2]
    assert float(stored.loc[sample, "close"]) == pytest.approx(
        float(wide.loc[sample, "close"])
    )


def test_yfinance_fetcher_frame_equal_fixture(tmp_path, monkeypatch):
    """Regression: batched fetch matches per-ticker normalization for a small fixture."""
    import yfinance as yf

    tickers = ["AAPL", "MSFT", "NVDA"]

    def fake_download(tickers_arg, **kwargs):
        batch = (
            tickers_arg.split() if isinstance(tickers_arg, str) else list(tickers_arg)
        )
        return _download_frame(batch, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    start, end = date(2024, 1, 1), date(2024, 1, 10)
    batched = fetcher.fetch(tickers, start, end)

    for ticker in tickers:
        single = fetcher.fetch([ticker], start, end)
        pd.testing.assert_frame_equal(batched[ticker], single[ticker])


def test_ticker_fetch_timeout_bounds_caller(monkeypatch) -> None:
    blocker = __import__("threading").Event()
    monkeypatch.setattr(data_module, "YFINANCE_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError, match="exceeded"):
        data_module.call_yfinance_with_timeout(lambda: blocker.wait())

    blocker.set()
