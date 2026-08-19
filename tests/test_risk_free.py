"""Which upstream supplies the absolute-momentum hurdle, and in what order."""

from __future__ import annotations

from datetime import date

import pytest


def test_fmp_treasury_is_preferred_over_the_yfinance_quote(monkeypatch):
    """^IRX is a discount rate; FMP month3 is the yield the hurdle wants."""
    import pandas as pd

    from screener import fmp, risk_free

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: "key")

    class FakeClient:
        def __init__(self, *a, **k) -> None:
            pass

        def get(self, path: str, params=None):
            assert path == "treasury"
            return [
                {"date": "2026-01-02", "month3": 4.0},
                {"date": "2026-01-05", "month3": 4.5},
            ]

    monkeypatch.setattr(fmp, "FmpClient", FakeClient)

    class Boom:
        def fetch(self, *a, **k):
            raise AssertionError("yfinance must not be consulted when FMP answers")

    index = pd.DatetimeIndex(pd.date_range("2026-01-02", periods=4, freq="D"))
    rate = risk_free.annualized_rate(
        "us", index, Boom(), date(2026, 1, 1), date(2026, 1, 6)
    )
    # Percent -> decimal, forward-filled across the weekend gap.
    assert rate.tolist() == pytest.approx([0.04, 0.04, 0.04, 0.045])


def test_a_dead_fmp_falls_back_to_the_price_fetcher(monkeypatch):
    import pandas as pd

    from screener import fmp, risk_free

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: "key")

    class Boom:
        def __init__(self, *a, **k) -> None:
            pass

        def get(self, *a, **k):
            raise RuntimeError("503")

    monkeypatch.setattr(fmp, "FmpClient", Boom)

    class Fetcher:
        def fetch(self, tickers, start, end):
            idx = pd.DatetimeIndex(pd.date_range("2026-01-02", periods=2, freq="D"))
            return {"^IRX": pd.DataFrame({"close": [3.0, 3.0]}, index=idx)}

    index = pd.DatetimeIndex(pd.date_range("2026-01-02", periods=2, freq="D"))
    rate = risk_free.annualized_rate(
        "us", index, Fetcher(), date(2026, 1, 1), date(2026, 1, 3)
    )
    assert rate.tolist() == pytest.approx([0.03, 0.03])


def test_india_never_calls_a_provider(monkeypatch):
    import pandas as pd

    from screener import fmp, risk_free

    def explode():
        raise AssertionError("India has no upstream bill series to call")

    monkeypatch.setattr(fmp, "resolve_api_key", explode)
    index = pd.DatetimeIndex(pd.date_range("2026-01-02", periods=2, freq="D"))

    class Boom:
        def fetch(self, *a, **k):
            raise AssertionError("no fetch for India")

    rate = risk_free.annualized_rate(
        "india", index, Boom(), date(2026, 1, 1), date(2026, 1, 3)
    )
    assert rate.tolist() == pytest.approx([risk_free.INDIA_TBILL_RATE] * 2)
