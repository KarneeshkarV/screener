from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.data import (
    FallbackPriceFetcher,
    FMPPriceFetcher,
    build_price_fetcher,
)
from screener.backtester import data as data_module


class DummyResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class DummySession:
    def __init__(self, payload: object) -> None:
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


def _intraday_payload() -> list[dict]:
    # historical-chart returns a bare list of bars, newest first.
    return [
        {
            "date": "2024-06-20 15:45:00",
            "open": 105,
            "low": 104,
            "high": 110,
            "close": 108,
            "volume": 1200,
        },
        {
            "date": "2024-06-20 15:30:00",
            "open": 100,
            "low": 99,
            "high": 106,
            "close": 104,
            "volume": 1000,
        },
    ]


def test_fmp_intraday_uses_historical_chart_and_keeps_timestamps(tmp_path):
    session = DummySession(_intraday_payload())
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        interval="15m",
    )

    out = fetcher.fetch(["AAA"], date(2024, 6, 18), date(2024, 6, 21))

    assert session.calls[0][0].endswith("/historical-chart/15min/AAA")
    frame = out["AAA"]
    # FMP's Eastern wall-clock is converted to naive UTC (EDT = UTC-4) so the
    # bars align with yfinance intraday frames.
    assert frame.index.tolist() == [
        pd.Timestamp("2024-06-20 19:30:00"),
        pd.Timestamp("2024-06-20 19:45:00"),
    ]
    assert frame.loc[pd.Timestamp("2024-06-20 19:30:00"), "close"] == 104
    assert "adj_close" not in frame.columns


def test_fmp_intraday_cache_round_trip_preserves_timestamps(tmp_path):
    session = DummySession(_intraday_payload())
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        interval="15m",
    )

    first = fetcher.fetch(["AAA"], date(2024, 6, 18), date(2024, 6, 21))
    second = fetcher.fetch(["AAA"], date(2024, 6, 18), date(2024, 6, 21))

    assert len(session.calls) == 1
    assert first["AAA"].equals(second["AAA"])
    assert second["AAA"].index.tolist() == [
        pd.Timestamp("2024-06-20 19:30:00"),
        pd.Timestamp("2024-06-20 19:45:00"),
    ]


def test_fmp_intraday_fetch_includes_whole_end_date(tmp_path):
    session = DummySession(_intraday_payload())
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        interval="15m",
    )

    out = fetcher.fetch(["AAA"], date(2024, 6, 20), date(2024, 6, 20))

    assert out["AAA"].index.tolist() == [
        pd.Timestamp("2024-06-20 19:30:00"),
        pd.Timestamp("2024-06-20 19:45:00"),
    ]


def test_fmp_intraday_cache_key_is_namespaced_per_interval():
    assert data_module._fmp_cache_key("AAA", True) == "fmp_AAA"
    assert data_module._fmp_cache_key("AAA", False) == "fmp_AAA__raw"
    assert data_module._fmp_cache_key("AAA", True, "15m") == "fmp_AAA__15m"
    assert data_module._fmp_cache_key("AAA", False, "15m") == "fmp_AAA__15m__raw"


def test_fmp_fetcher_rejects_unsupported_interval():
    with pytest.raises(ValueError, match="interval"):
        FMPPriceFetcher(api_key="test-key", interval="45m")


def test_build_price_fetcher_intraday_includes_fmp_fallback(monkeypatch):
    monkeypatch.delenv("SCREENER_PRICE_PROVIDER", raising=False)
    monkeypatch.setenv("FMP_API_KEY", "env-key")

    fetcher = build_price_fetcher(interval="15m")

    assert isinstance(fetcher, FallbackPriceFetcher)
    assert isinstance(fetcher.fallback, FMPPriceFetcher)
    assert fetcher.fallback.interval == "15m"


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


def test_build_price_fetcher_loads_fmp_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCREENER_PRICE_PROVIDER", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(data_module, "_DOTENV_LOADED", False)
    (tmp_path / ".env").write_text('FMP_API_KEY="dotenv-key"\n')

    fetcher = build_price_fetcher()

    assert isinstance(fetcher, FallbackPriceFetcher)
    assert isinstance(fetcher.fallback, FMPPriceFetcher)
    assert fetcher.fallback.api_key == "dotenv-key"


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
