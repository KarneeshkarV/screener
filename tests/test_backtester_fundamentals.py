from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from main import cli

from screener import fmp
from screener.backtester import fundamentals, rolling
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=3,
        top=1,
        entry_expr="roe_ttm > 15 and close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _sample_payload() -> dict[str, object]:
    return {
        "income": [
            {
                "date": "2024-01-31",
                "acceptedDate": "2024-02-05 16:30:00",
                "revenue": 120.0,
                "eps": 1.2,
            },
            {
                "date": "2023-01-31",
                "acceptedDate": "2023-02-04 16:30:00",
                "revenue": 100.0,
                "eps": 1.0,
            },
            {
                "date": "2022-10-31",
                "acceptedDate": "2022-11-04 16:30:00",
                "revenue": 90.0,
                "eps": 0.9,
            },
        ],
        "ratios": [
            {
                "date": "2024-01-31",
                "priceEarningsRatio": 20.0,
                "priceToBookRatio": 3.0,
                "returnOnEquity": 0.18,
                "debtEquityRatio": 0.4,
            }
        ],
        "key_metrics": [{"date": "2024-01-31", "peRatioTTM": 19.0}],
        "balance": [{"date": "2024-01-31", "totalDebt": 40.0}],
        "enterprise_values": [
            {"date": "2024-01-31", "marketCapitalization": 5_000_000_000.0}
        ],
    }


def test_fmp_payload_normalizes_effective_dates_and_fields():
    frame = fundamentals._normalize_fmp_payload(
        _sample_payload(),
        fields=fundamentals.DEFAULT_FUNDAMENTAL_FIELDS,
        lag_days=1,
    )

    assert pd.Timestamp("2024-02-06") in frame.index
    row = frame.loc[pd.Timestamp("2024-02-06")]
    assert row["pe_ttm"] == 19.0
    assert row["pb_ttm"] == 3.0
    assert row["roe_ttm"] == 18.0
    assert row["revenue_growth_yoy"] == pytest.approx(20.0)
    assert row["eps_growth_yoy"] == pytest.approx(20.0)
    assert row["revenue_up_3q"] == 1.0
    assert row["market_cap"] == 5_000_000_000.0


class _FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self) -> object:
        return self._payload


class _FakeSession:
    """Records every GET; stands in for a pooled ``requests.Session``."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append((url, dict(headers), timeout))
        return self.response


def test_fmp_get_routes_through_shared_client_apikey_last_timeout_30():
    session = _FakeSession(_FakeResponse({"ok": True}))

    out = fundamentals._fmp_get(
        session,  # type: ignore[arg-type]
        "income-statement/AAPL",
        {"period": "quarter", "limit": 4},
        "SECRET",
    )

    assert out == {"ok": True}
    url, headers, timeout = session.calls[0]
    # Shared v3 base URL, param order preserved with apikey appended last.
    assert url == (
        f"{fmp.FMP_V3_BASE_URL}/income-statement/AAPL"
        "?period=quarter&limit=4&apikey=SECRET"
    )
    # timeout=30 (not the fmp default 20) preserved, legacy empty-header UA kept.
    assert timeout == 30.0
    assert headers == {}


def test_fmp_get_preserves_requests_http_error_on_non_2xx():
    session = _FakeSession(_FakeResponse(None, status=404))

    with pytest.raises(requests.HTTPError):
        fundamentals._fmp_get(
            session,  # type: ignore[arg-type]
            "income-statement/AAPL",
            {"limit": 1},
            "SECRET",
        )


def test_fetch_fmp_sections_uses_injected_session_for_all_endpoints():
    session = _FakeSession(_FakeResponse([{"date": "2024-01-31"}]))

    payload = fundamentals._fetch_fmp_sections(
        "AAPL",
        api_key="SECRET",
        session=session,  # type: ignore[arg-type]
        limit=4,
        fields=fundamentals.DEFAULT_FUNDAMENTAL_FIELDS,
    )

    assert set(payload) == {
        "income",
        "balance",
        "ratios",
        "key_metrics",
        "enterprise_values",
    }
    requested = [url for url, _headers, _timeout in session.calls]
    assert any("income-statement/AAPL" in u for u in requested)
    assert any("enterprise-values/AAPL" in u for u in requested)
    assert all(u.startswith(fmp.FMP_V3_BASE_URL) for u in requested)
    assert all(u.endswith("apikey=SECRET") for u in requested)


def test_openscreener_payload_normalizes_revenue_up_3q_with_india_lag():
    frame = fundamentals._normalize_openscreener_payload(
        {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
                {"date": "Mar 2024", "sales": 110.0},
            ]
        },
        fields=("revenue_up_3q",),
        lag_days=60,
    )

    assert pd.Timestamp("2025-03-01") in frame.index
    assert frame.loc[pd.Timestamp("2025-03-01"), "revenue_up_3q"] == 1.0
    assert frame.loc[pd.Timestamp("2024-11-29"), "revenue_up_3q"] == 0.0


def test_openscreener_fetcher_falls_back_to_yfinance(monkeypatch, fake_provider):
    empty_provider = fake_provider()
    monkeypatch.setattr(fundamentals, "_OPENSCREENER_PROVIDER", empty_provider)
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", fake_provider()
    )
    monkeypatch.setattr(
        fundamentals,
        "_fetch_openscreener_quarterly",
        lambda symbol: {},
    )
    monkeypatch.setattr(
        fundamentals,
        "_fetch_yfinance_quarterly_revenue",
        lambda ticker: {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
            ]
        },
    )

    fetcher = fundamentals.OpenScreenerFundamentalFetcher(
        fields=("revenue_up_3q",), lag_days=60
    )
    out = fetcher.fetch(
        ["RELIANCE.NS"],
        date(2024, 1, 1),
        date(2025, 12, 31),
        "india",
    )

    assert out["RELIANCE.NS"].loc[pd.Timestamp("2025-03-01"), "revenue_up_3q"] == 1.0


def test_merge_fundamentals_forward_fills_only_after_effective_date():
    bars = make_bars(start="2024-02-01", n=8)
    fundamentals_frame = pd.DataFrame(
        {"roe_ttm": [18.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-02-06")]),
    )

    merged = fundamentals.merge_fundamentals_into_bars(
        {"AAA": bars},
        {"AAA": fundamentals_frame},
        {"AAA": "AAA"},
    )["AAA"]

    assert pd.isna(merged.loc[pd.Timestamp("2024-02-05"), "roe_ttm"])
    assert merged.loc[pd.Timestamp("2024-02-06"), "roe_ttm"] == 18.0
    assert merged.loc[pd.Timestamp("2024-02-08"), "roe_ttm"] == 18.0


class _StubFundamentalFetcher:
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.frame = frame

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
        market: str,
    ) -> dict[str, pd.DataFrame]:
        return {
            ticker: self.frame.copy() if self.frame is not None else pd.DataFrame()
            for ticker in tickers
        }


def test_rolling_backtest_uses_fundamental_columns_in_entry(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=50)
    aaa = make_bars(n=50, start="2024-01-01", open_base=100.0)
    aaa.index = idx
    aaa["volume"] = 100_000.0
    spy = make_bars(n=50, start="2024-01-01", open_base=400.0)
    spy.index = idx
    fundamental_frame = pd.DataFrame(
        {"roe_ttm": [20.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-22")]),
    )

    monkeypatch.setattr(
        rolling,
        "build_fundamental_fetcher",
        lambda *args, **kwargs: _StubFundamentalFetcher(fundamental_frame),
    )

    result = run_rolling_backtest(
        _cfg(fundamentals_provider="fmp", fundamental_fields=("roe_ttm",)),
        StubPriceFetcher({"AAA": aaa, "SPY": spy}),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 29),
    )

    assert not result.selection.empty
    assert result.selection["signal_date"].min() >= date(2024, 1, 22)


def test_rolling_backtest_missing_fundamentals_does_not_break_price_only_entry(
    monkeypatch,
):
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=40, start="2024-01-01", open_base=100.0),
            "SPY": make_bars(n=40, start="2024-01-01", open_base=400.0),
        }
    )
    monkeypatch.setattr(
        rolling,
        "build_fundamental_fetcher",
        lambda *args, **kwargs: _StubFundamentalFetcher(),
    )

    result = run_rolling_backtest(
        _cfg(entry_expr="close > 0", fundamentals_provider="fmp"),
        fetcher,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 20),
    )

    assert isinstance(result.trades, list)


def test_fundamentals_provider_rejects_non_us_market():
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "india",
            "--tickers",
            "RELIANCE",
            "--entry",
            "close > 0",
            "--fundamentals-provider",
            "fmp",
        ],
        obj=StubPriceFetcher({}),
    )

    assert res.exit_code != 0
    assert "supports only -m us" in res.output


def test_openscreener_provider_rejects_non_india_market():
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--entry",
            "close > 0",
            "--fundamentals-provider",
            "openscreener",
        ],
        obj=StubPriceFetcher({}),
    )

    assert res.exit_code != 0
    assert "supports only -m india" in res.output
