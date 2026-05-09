from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.data import (
    FallbackPriceFetcher,
    FMPPriceFetcher,
    build_price_fetcher,
)


class DummyResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class DummySession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict, timeout: int) -> DummyResponse:
        self.calls.append((url, {"params": params, "timeout": timeout}))
        return DummyResponse(self.payload)


def _payload() -> dict:
    return {
        "symbol": "AAA",
        "historical": [
            {
                "date": "2024-01-03",
                "open": 105,
                "high": 110,
                "low": 104,
                "close": 108,
                "adjClose": 54,
                "volume": 1200,
            },
            {
                "date": "2024-01-02",
                "open": 100,
                "high": 106,
                "low": 99,
                "close": 104,
                "adjClose": 52,
                "volume": 1000,
            },
        ],
    }


def test_fmp_fetcher_uses_api_key_and_normalizes_adjusted_prices(tmp_path):
    session = DummySession(_payload())
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))

    assert session.calls[0][0].endswith("/AAA")
    assert session.calls[0][1]["params"]["apikey"] == "test-key"
    frame = out["AAA"]
    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]
    assert frame.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert frame.loc[pd.Timestamp("2024-01-02"), "close"] == 52
    assert frame.loc[pd.Timestamp("2024-01-03"), "open"] == 52.5


def test_fmp_fetcher_uses_cache_on_second_call(tmp_path):
    session = DummySession(_payload())
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    first = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))
    second = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))

    assert len(session.calls) == 1
    assert first["AAA"].equals(second["AAA"])


def test_build_price_fetcher_selects_fmp_from_env(monkeypatch):
    monkeypatch.setenv("SCREENER_PRICE_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "env-key")

    fetcher = build_price_fetcher()

    assert isinstance(fetcher, FMPPriceFetcher)


def test_build_price_fetcher_defaults_to_yfinance_with_fmp_fallback(monkeypatch):
    monkeypatch.delenv("SCREENER_PRICE_PROVIDER", raising=False)
    monkeypatch.setenv("FMP_API_KEY", "env-key")

    fetcher = build_price_fetcher()

    assert isinstance(fetcher, FallbackPriceFetcher)


def test_fallback_fetcher_fills_empty_primary_results():
    class StubFetcher:
        def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
            self.frames = frames
            self.calls: list[list[str]] = []

        def fetch(self, tickers, start, end):
            ticker_list = list(tickers)
            self.calls.append(ticker_list)
            return {
                ticker: self.frames.get(ticker, pd.DataFrame())
                for ticker in ticker_list
            }

    fallback_frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    primary = StubFetcher({"AAA": pd.DataFrame(), "BBB": fallback_frame})
    fallback = StubFetcher({"AAA": fallback_frame})
    fetcher = FallbackPriceFetcher(primary, fallback)

    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 5))

    assert fallback.calls == [["AAA"]]
    assert out["AAA"].equals(fallback_frame)
    assert out["BBB"].equals(fallback_frame)
