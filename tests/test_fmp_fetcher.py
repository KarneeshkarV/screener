from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.data import FMPPriceFetcher, build_price_fetcher


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
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "adj_close"]
    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
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
