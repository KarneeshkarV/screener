from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.data import YFinancePriceFetcher


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
