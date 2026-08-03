from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli

from screener.backtester import fundamentals
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest

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


def test_yfinance_fetcher_fetches_quarterly_revenue(monkeypatch, fake_provider):
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", fake_provider()
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

    fetcher = fundamentals.YFinanceFundamentalFetcher(
        fields=("revenue_up_3q",), lag_days=60
    )
    out = fetcher.fetch(
        ["RELIANCE.NS"],
        date(2024, 1, 1),
        date(2025, 12, 31),
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
    ) -> dict[str, pd.DataFrame]:
        return {
            ticker: self.frame.copy() if self.frame is not None else pd.DataFrame()
            for ticker in tickers
        }


def test_rolling_backtest_uses_fundamental_columns_in_entry():
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

    result = run_rolling_backtest(
        _cfg(fundamentals_provider="fmp", fundamental_fields=("roe_ttm",)),
        StubPriceFetcher({"AAA": aaa, "SPY": spy}),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 29),
        fundamental_fetcher=_StubFundamentalFetcher(fundamental_frame),
    )

    assert not result.selection.empty
    assert result.selection["signal_date"].min() >= date(2024, 1, 22)


def test_rolling_backtest_missing_fundamentals_does_not_break_price_only_entry():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=40, start="2024-01-01", open_base=100.0),
            "SPY": make_bars(n=40, start="2024-01-01", open_base=400.0),
        }
    )
    result = run_rolling_backtest(
        _cfg(entry_expr="close > 0", fundamentals_provider="fmp"),
        fetcher,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 20),
        fundamental_fetcher=_StubFundamentalFetcher(),
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


def test_referenced_fundamental_fields_detects_known_fields():
    from screener.backtester.cli_common import referenced_fundamental_fields

    assert referenced_fundamental_fields("revenue_up_3q > 0 and close > 0", None) == {
        "revenue_up_3q"
    }
    # Pure-price expressions reference no fundamentals.
    assert referenced_fundamental_fields("ema(close, 150) > ema(close, 200)", None) == (
        set()
    )
    # Exit expressions are inspected too.
    assert referenced_fundamental_fields("close > 0", "pe_ttm > 30") == {"pe_ttm"}


def _rolling_request(**overrides):
    from screener.backtester.workflow import BacktestRequest

    values = dict(
        mode="rolling",
        context_obj=StubPriceFetcher({}),
        market="us",
        hold=20,
        top=10,
        entry_expr="close > 0",
        exit_expr=None,
        strategy_name=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        cost_model="flat",
        initial_capital=100_000.0,
        benchmark=None,
        tickers="AAPL",
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        adv_window=20,
        slippage_model="fixed",
        half_spread_bps=0.0,
        vol_impact_k=0.1,
        no_gap_fills=False,
        entry_order="moo",
        entry_limit_bps=None,
        partial_exit_args=(),
        price_adjustment="full",
        interval="1d",
        output_csv=False,
        report_path=None,
        open_report=False,
        sizing_rule="equal_slot",
        sizing_risk_pct=0.01,
        sizing_position_pct=0.1,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
        sizing_vol_window=20,
        intraday_only=False,
    )
    values.update(overrides)
    return BacktestRequest(**values)


def test_rolling_auto_enables_fundamentals_for_fundamental_expr(monkeypatch):
    from screener.backtester.workflow import resolve_backtest_run

    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")
    run = resolve_backtest_run(
        _rolling_request(strategy_name="ema150_200_revenue_up_3q", entry_expr=None)
    )

    assert run.config.fundamentals_provider == "fmp"
    assert isinstance(run.fundamental_fetcher, fundamentals.FMPFundamentalFetcher)


def test_rolling_does_not_enable_fundamentals_for_price_only_expr():
    from screener.backtester.workflow import resolve_backtest_run

    run = resolve_backtest_run(
        _rolling_request(entry_expr="ema(close, 150) > ema(close, 200)")
    )

    assert run.config.fundamentals_provider is None
    assert run.fundamental_fetcher is None


def test_rolling_unions_referenced_field_into_explicit_field_list(monkeypatch):
    from screener.backtester.workflow import resolve_backtest_run

    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")
    run = resolve_backtest_run(
        _rolling_request(
            entry_expr="revenue_up_3q > 0",
            fundamental_field_args=("roe_ttm",),
        )
    )

    assert run.config.fundamentals_provider == "fmp"
    assert set(run.config.fundamental_fields) == {"roe_ttm", "revenue_up_3q"}


# --------------------------------------------------------------------------- #
# Unit coverage for the fundamentals adapter helpers and fetcher orchestration
# --------------------------------------------------------------------------- #


class _RaisingProvider:
    """Provider seam whose ``fetch`` always raises (drives thread except-branch)."""

    def fetch(self, *args, **kwargs):  # noqa: D401 - test double
        raise RuntimeError("provider boom")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, *, headers, timeout):
        return _FakeResponse(self._payload)


def test_num_handles_unparseable_and_nan():
    from screener.financials import to_number as _num

    assert _num("not-a-number") is None
    assert _num(float("nan")) is None
    assert _num("N/A") is None
    assert _num(None) is None
    assert _num("1,234.5%") == 1234.5


def test_increased_last_n_quarters_none_when_value_missing():
    rows = [{"revenue": 100.0}, {"revenue": None}, {"revenue": 90.0}]
    assert fundamentals._increased_last_n_quarters(rows, 0, "revenue", 3) is None


def test_increased_last_n_revenues_none_when_value_missing():
    rows = [{"sales": 100.0}, {"sales": None}, {"sales": 90.0}]
    assert fundamentals._increased_last_n_revenues(rows, 0, 3) is None


def test_effective_date_none_when_unparseable():
    assert fundamentals._effective_date({"date": "not-a-date"}, 1) is None
    assert fundamentals._effective_date({}, 1) is None


def test_parse_india_period_end_handles_empty_iso_and_garbage():
    assert fundamentals._parse_india_period_end("") is None
    assert fundamentals._parse_india_period_end(None) is None

    iso = fundamentals._parse_india_period_end("2024-03-31")
    assert iso == pd.Timestamp("2024-03-31")

    assert fundamentals._parse_india_period_end("nonsense-label") is None


def test_fmp_get_returns_parsed_json():
    out = fundamentals._fmp_get(
        _FakeSession({"symbol": "AAPL", "revenue": 1}),
        "income-statement/AAPL",
        {"period": "quarter", "limit": 120},
        "test-key",
    )
    assert out == {"symbol": "AAPL", "revenue": 1}


def test_fmp_fetcher_requires_api_key(monkeypatch):
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        fundamentals.FMPFundamentalFetcher()


def test_fmp_fetcher_init_normalizes_config():
    fetcher = fundamentals.FMPFundamentalFetcher(
        api_key="x",
        fields=["roe_ttm", "roe_ttm", "pe_ttm"],
        lag_days=-1,
        limit=0,
        max_workers=0,
    )
    assert fetcher.api_key == "x"
    assert fetcher.fields == ("roe_ttm", "pe_ttm")
    assert fetcher.lag_days == 0
    assert fetcher.limit == 1
    assert fetcher.max_workers == 1
    assert fetcher.refresh is False


def test_fundamental_fetchers_declare_supported_markets():
    assert fundamentals.FMPFundamentalFetcher.markets == frozenset({"us"})
    assert fundamentals.OpenScreenerFundamentalFetcher.markets == frozenset({"india"})
    assert fundamentals.YFinanceFundamentalFetcher.markets == frozenset({"india"})


def test_fmp_fetcher_fetch_single_ticker(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_fetch_fmp_sections", lambda symbol, **k: _sample_payload()
    )
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=1)

    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 12, 31))

    assert "AAA" in out
    assert pd.Timestamp("2024-02-06") in out["AAA"].index


def test_fmp_fetcher_fetch_threaded(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_fetch_fmp_sections", lambda symbol, **k: _sample_payload()
    )
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31))

    assert set(out) == {"AAA", "BBB"}
    assert not out["AAA"].empty


def test_fmp_fetcher_fetch_threaded_handles_provider_failures(monkeypatch):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", _RaisingProvider())
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31))

    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].empty
    assert out["BBB"].empty


def test_fundamental_fetcher_protocol_has_no_market_argument():
    assert "market" not in fundamentals.FundamentalFetcher.fetch.__annotations__


def test_openscreener_fetcher_fetch_threaded(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_OPENSCREENER_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", fake_provider()
    )
    monkeypatch.setattr(
        fundamentals,
        "_fetch_openscreener_quarterly",
        lambda symbol: {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
            ]
        },
    )
    fetcher = fundamentals.OpenScreenerFundamentalFetcher(max_workers=4)

    out = fetcher.fetch(["RELIANCE.NS", "TCS.NS"], date(2024, 1, 1), date(2025, 12, 31))

    assert set(out) == {"RELIANCE.NS", "TCS.NS"}
    assert pd.Timestamp("2025-03-01") in out["RELIANCE.NS"].index


def test_openscreener_fetcher_fetch_threaded_handles_failures(monkeypatch):
    monkeypatch.setattr(fundamentals, "_OPENSCREENER_PROVIDER", _RaisingProvider())
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", _RaisingProvider()
    )
    fetcher = fundamentals.OpenScreenerFundamentalFetcher(max_workers=4)

    out = fetcher.fetch(["RELIANCE.NS", "TCS.NS"], date(2024, 1, 1), date(2025, 12, 31))

    assert set(out) == {"RELIANCE.NS", "TCS.NS"}
    assert out["RELIANCE.NS"].empty
    assert out["TCS.NS"].empty


def test_build_fundamental_fetcher_resolves_providers(monkeypatch):
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")

    assert fundamentals.build_fundamental_fetcher(None, market="us") is None
    assert fundamentals.build_fundamental_fetcher("   ", market="us") is None

    assert isinstance(
        fundamentals.build_fundamental_fetcher("fmp", market="us"),
        fundamentals.FMPFundamentalFetcher,
    )
    assert isinstance(
        fundamentals.build_fundamental_fetcher("FMP", market="us"),
        fundamentals.FMPFundamentalFetcher,
    )
    assert isinstance(
        fundamentals.build_fundamental_fetcher("openscreener", market="india"),
        fundamentals.OpenScreenerFundamentalFetcher,
    )
    assert isinstance(
        fundamentals.build_fundamental_fetcher("open-screener", market="india"),
        fundamentals.OpenScreenerFundamentalFetcher,
    )

    assert isinstance(
        fundamentals.build_fundamental_fetcher("yfinance", market="india"),
        fundamentals.YFinanceFundamentalFetcher,
    )
    with pytest.raises(ValueError, match="supports only -m us"):
        fundamentals.build_fundamental_fetcher("fmp", market="india")
    with pytest.raises(ValueError):
        fundamentals.build_fundamental_fetcher("garbage", market="us")


def test_merge_fundamentals_skips_empty_or_none_bars():
    out = fundamentals.merge_fundamentals_into_bars(
        {"AAA": pd.DataFrame(), "BBB": None},
        {},
        {},
    )
    assert out["AAA"].empty
    assert out["BBB"] is None
