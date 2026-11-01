from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import cache
from screener import garp as garp_module
from screener.cli import cli
from screener.garp import (
    INDIA_THRESHOLDS,
    _fmp_us_row,
    _passes_garp,
    add_garp_score,
    screen_us_garp,
)


def _passing_row(**overrides):
    row = {
        "name": "AAA",
        "market_cap": 1500.0,
        "sales": 1600.0,
        "peg": 1.2,
        "sales_growth_5y": 18.0,
        "operating_profit_growth": 12.0,
        "eps_growth_5y": 16.0,
        "roe_5y": 17.0,
        "roce_or_roic": 18.0,
        "quarterly_profit_growth": 20.0,
    }
    row.update(overrides)
    return row


def test_garp_filter_accepts_complete_india_match() -> None:
    assert _passes_garp(_passing_row(), INDIA_THRESHOLDS) is True


def test_garp_filter_rejects_missing_required_input() -> None:
    assert _passes_garp(_passing_row(peg=None), INDIA_THRESHOLDS) is False


def test_garp_score_prefers_lower_peg_and_stronger_growth() -> None:
    df = pd.DataFrame(
        [
            _passing_row(name="LOWPEG", peg=0.8, eps_growth_5y=20.0),
            _passing_row(name="HIGHPEG", peg=1.8, eps_growth_5y=13.0),
        ]
    )
    scored = add_garp_score(df)
    assert scored.iloc[0]["name"] == "LOWPEG"
    assert "garp_score" in scored.columns


def test_openscreener_adapter_preserves_cache_contract(monkeypatch) -> None:
    captured: dict = {}

    def fake_cached_json_call(namespace, key_parts, *, ttl_seconds, refresh, fetch):
        captured["namespace"] = namespace
        captured["key_parts"] = key_parts
        captured["ttl_seconds"] = ttl_seconds
        captured["refresh"] = refresh
        captured["fetch_result"] = fetch()
        return _india_sections()

    monkeypatch.setattr(garp_module, "cached_json_call", fake_cached_json_call)
    monkeypatch.setattr(
        garp_module, "_fetch_india_sections", lambda symbol: {"raw": symbol}
    )

    row = garp_module.OpenScreenerGarpAdapter().load_row(
        "AAA", "Alpha", cache_ttl=123.0, refresh=True
    )

    assert captured == {
        "namespace": "garp_india",
        "key_parts": ("india", "AAA", garp_module._INDIA_PAYLOAD_VERSION),
        "ttl_seconds": 123.0,
        "refresh": True,
        "fetch_result": {"raw": "AAA"},
    }
    assert row is not None
    assert row["name"] == "AAA"
    assert row["description"] == "Alpha"
    assert row["peg"] == pytest.approx(_INDIA_PEG)


def test_garp_cli_emits_csv(monkeypatch) -> None:
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    results = add_garp_score(pd.DataFrame([_passing_row(description="Alpha")]))

    monkeypatch.setattr(garp_module, "load_garp_universe", lambda *a, **k: universe)
    monkeypatch.setattr(garp_module, "screen_india_garp", lambda *a, **k: results)

    res = CliRunner().invoke(cli, ["garp", "-m", "india", "--csv"])

    assert res.exit_code == 0, res.output
    assert "garp_score" in res.output
    assert "AAA" in res.output


def test_run_garp_screen_returns_scored_results(monkeypatch) -> None:
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    results = add_garp_score(pd.DataFrame([_passing_row(description="Alpha")]))
    announced: list[int] = []

    monkeypatch.setattr(garp_module, "load_garp_universe", lambda *a, **k: universe)
    monkeypatch.setattr(garp_module, "screen_india_garp", lambda *a, **k: results)

    out = garp_module.run_garp_screen(
        "india",
        200,
        limit=30,
        workers=8,
        cache_ttl=None,
        refresh=False,
        on_universe=lambda df: announced.append(len(df)),
    )

    assert out is not None
    assert list(out["name"]) == ["AAA"]
    assert announced == [1]


def test_invalid_provider_row_logs_symbol_and_first_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr(
        garp_module.logger,
        "warning",
        lambda event, **kwargs: warnings.append((event, kwargs)),
    )

    assert (
        garp_module._coerce_garp_fundamentals({"unexpected": True}, symbol="BROKEN")
        is None
    )
    assert warnings[0][0] == "garp_fundamentals_validation_failed"
    assert warnings[0][1]["symbol"] == "BROKEN"
    assert warnings[0][1]["error"]["loc"] == ("unexpected",)


def test_run_garp_screen_returns_none_on_empty_universe(monkeypatch) -> None:
    monkeypatch.setattr(
        garp_module, "load_garp_universe", lambda *a, **k: pd.DataFrame()
    )

    out = garp_module.run_garp_screen(
        "india", 200, limit=30, workers=8, cache_ttl=None, refresh=False
    )

    assert out is None


# ── FMP-backed US fundamentals ──────────────────────────────────────────────

_ANNUAL_DATES = ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
_REVENUE = [5.0e9, 4.5e9, 4.0e9, 3.5e9, 2.5e9]
_OPERATING = [1.2e9, 1.0e9, 0.9e9, 0.8e9, 0.7e9]
_NET_INCOME = [8.0e8, 7.0e8, 6.0e8, 5.0e8, 4.0e8]
_EQUITY = [4.0e9, 3.5e9, 3.0e9, 2.5e9, 2.0e9]
_DEBT = [1.0e9] * 5
_QUARTER_DATES = ["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31"]
_QUARTER_EPS = [1.2, 1.1, 1.0, 0.9, 0.8]


def _fmp_payload() -> dict:
    income = [
        {
            "date": date,
            "revenue": _REVENUE[i],
            "operatingIncome": _OPERATING[i],
            "netIncome": _NET_INCOME[i],
            "incomeTaxExpense": 2.0e8,
            "incomeBeforeTax": 1.0e9,
        }
        for i, date in enumerate(_ANNUAL_DATES)
    ]
    balance = [
        {"date": d, "totalStockholdersEquity": _EQUITY[i], "totalDebt": _DEBT[i]}
        for i, d in enumerate(_ANNUAL_DATES)
    ]
    quarterly = [
        {"date": d, "eps": _QUARTER_EPS[i]} for i, d in enumerate(_QUARTER_DATES)
    ]
    return {
        "profile": [{"mktCap": 2.0e9, "companyName": "Alpha"}],
        "ratios_ttm": [{"priceEarningsToGrowthRatioTTM": 1.2}],
        "income_annual": income,
        "balance_annual": balance,
        "income_quarterly": quarterly,
        "estimates_quarterly": [
            {"date": "2026-06-30", "estimatedEpsAvg": 1.4},
            {"date": "2026-03-31", "estimatedEpsAvg": 1.3},
            {"date": "2025-12-31", "estimatedEpsAvg": 1.15},
        ],
    }


def test_fmp_us_row_maps_fmp_fields_to_scorer_inputs() -> None:
    row = _fmp_us_row("AAA", "Alpha", _fmp_payload())

    assert row is not None
    assert row["name"] == "AAA"
    assert row["description"] == "Alpha"
    assert row["market_cap"] == pytest.approx(2.0e9)
    assert row["sales"] == pytest.approx(5.0e9)
    assert row["peg"] == pytest.approx(1.2)
    # CAGR over 4 years: (5e9 / 2.5e9) ** (1/4) - 1
    assert row["sales_growth_5y"] == pytest.approx((2.0**0.25 - 1.0) * 100.0, rel=1e-9)
    assert row["operating_profit_growth"] == pytest.approx(20.0)
    assert row["eps_growth_5y"] == pytest.approx((2.0**0.25 - 1.0) * 100.0, rel=1e-9)
    assert row["roe_5y"] == pytest.approx(20.0)
    # NOPAT = operating income * (1 - 0.2); invested capital = debt + equity.
    assert row["roce_or_roic"] == pytest.approx((19.2 + 1600.0 / 90.0 + 18.0) / 3.0)
    assert row["expected_quarterly_profit"] == pytest.approx(1.3)
    assert row["profit_3q_back"] == pytest.approx(0.9)
    assert row["quarterly_profit_growth"] == pytest.approx(400.0 / 9.0)


def test_fmp_us_row_returns_none_without_statements() -> None:
    payload = {"profile": [{"mktCap": 2.0e9}], "income_annual": []}

    assert _fmp_us_row("AAA", "Alpha", payload) is None


def test_us_fundamentals_adapter_prefers_fmp_and_preserves_cache_args(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_fetch(symbol, api_key, *, cache_ttl, refresh):
        captured["symbol"] = symbol
        captured["api_key"] = api_key
        captured["cache_ttl"] = cache_ttl
        captured["refresh"] = refresh
        return _fmp_payload()

    def _no_yfinance(symbol, description):
        raise AssertionError("yfinance path must not run when FMP has data")

    monkeypatch.setattr(garp_module, "_fetch_fmp_us_cached", fake_fetch)
    monkeypatch.setattr(garp_module, "_us_row", _no_yfinance)

    adapter = garp_module.UsGarpFundamentalsAdapter(
        garp_module.FmpGarpAdapter("test-key")
    )
    row = adapter.load_row("AAA", "Alpha", cache_ttl=456.0, refresh=True)

    assert captured == {
        "symbol": "AAA",
        "api_key": "test-key",
        "cache_ttl": 456.0,
        "refresh": True,
    }
    assert row is not None
    assert row["name"] == "AAA"
    assert row["peg"] == pytest.approx(1.2)


def test_us_fundamentals_adapter_falls_back_when_fmp_has_no_row(
    monkeypatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        garp_module,
        "_fetch_fmp_us_cached",
        lambda symbol, api_key, *, cache_ttl, refresh: {
            "profile": [],
            "income_annual": [],
        },
    )

    def _yf_row(symbol, description):
        calls.append(symbol)
        return _passing_row(name=symbol, market_cap=2.0e9, sales=5.0e9)

    monkeypatch.setattr(garp_module, "_us_row", _yf_row)

    adapter = garp_module.UsGarpFundamentalsAdapter(
        garp_module.FmpGarpAdapter("test-key")
    )
    row = adapter.load_row("AAA", "Alpha", cache_ttl=None, refresh=False)

    assert calls == ["AAA"]
    assert row is not None
    assert row["name"] == "AAA"


def test_fmp_row_matches_yfinance_row_on_equivalent_data(monkeypatch) -> None:
    dates = pd.to_datetime(_ANNUAL_DATES)
    income = pd.DataFrame(
        [_REVENUE, _OPERATING, _NET_INCOME, _OPERATING, [0.2] * 5],
        index=[
            "Total Revenue",
            "Operating Income",
            "Net Income",
            "EBIT",
            "Tax Rate For Calcs",
        ],
        columns=dates,
    )
    balance = pd.DataFrame(
        [_EQUITY, _DEBT],
        index=["Stockholders Equity", "Total Debt"],
        columns=dates,
    )
    estimates = pd.DataFrame({"avg": [1.3], "yearAgoEps": [0.9]}, index=["0q"])

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {
                "marketCap": 2.0e9,
                "trailingPegRatio": 1.2,
                "shortName": "Alpha",
            }
            self.income_stmt = income
            self.balance_sheet = balance
            self.earnings_estimate = estimates

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    yf_row = garp_module._us_row("AAA", "Alpha")
    fmp_row = _fmp_us_row("AAA", "Alpha", _fmp_payload())

    assert fmp_row is not None
    assert set(type(fmp_row).model_fields) == set(type(yf_row).model_fields)
    for key, expected in yf_row.model_dump().items():
        if isinstance(expected, float):
            assert fmp_row[key] == pytest.approx(expected), key
        else:
            assert fmp_row[key] == expected, key


def test_screen_us_garp_uses_fmp_when_key_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        garp_module, "_fetch_fmp_us_sections", lambda symbol, api_key: _fmp_payload()
    )

    def _no_yfinance(symbol, description):
        raise AssertionError("yfinance path must not run when FMP has data")

    monkeypatch.setattr(garp_module, "_us_row", _no_yfinance)

    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    out = screen_us_garp(universe, limit=10, workers=1, cache_ttl=None, refresh=True)

    assert list(out["name"]) == ["AAA"]
    assert out.iloc[0]["peg"] == pytest.approx(1.2)
    assert "garp_score" in out.columns


def test_screen_us_garp_falls_back_to_yfinance_without_key(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: None)

    def _no_fmp(symbol, api_key):
        raise AssertionError("FMP must not be queried without an API key")

    monkeypatch.setattr(garp_module, "_fetch_fmp_us_sections", _no_fmp)
    monkeypatch.setattr(
        garp_module,
        "_us_row",
        lambda symbol, description: _passing_row(
            name=symbol, market_cap=2.0e9, sales=5.0e9
        ),
    )

    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    out = screen_us_garp(universe, limit=10, workers=1, cache_ttl=None, refresh=True)

    assert list(out["name"]) == ["AAA"]


def test_screen_us_garp_falls_back_when_fmp_has_no_statements(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        garp_module,
        "_fetch_fmp_us_sections",
        lambda symbol, api_key: {"profile": [], "income_annual": []},
    )
    called: list[str] = []

    def _yf_row(symbol, description):
        called.append(symbol)
        return _passing_row(name=symbol, market_cap=2.0e9, sales=5.0e9)

    monkeypatch.setattr(garp_module, "_us_row", _yf_row)

    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    out = screen_us_garp(universe, limit=10, workers=1, cache_ttl=None, refresh=True)

    assert called == ["AAA"]
    assert list(out["name"]) == ["AAA"]


# ── public per-symbol GARP loader (used by the conviction card) ─────────────

_INDIA_SALES_CAGR = ((2200.0 / 1000.0) ** (1 / 5) - 1) * 100.0
_INDIA_EPS_CAGR = ((40.0 / 16.0) ** (1 / 5) - 1) * 100.0
_INDIA_PEG = 30.0 / _INDIA_EPS_CAGR


def _india_sections() -> dict:
    """openscreener section payloads: tables are lists, oldest period first."""
    years = [f"Mar {y}" for y in range(2019, 2026)]
    sales = [900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2200.0]
    eps = [15.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0]
    op_profit = [150.0, 180.0, 220.0, 260.0, 300.0, 320.0, 400.0]
    profit_loss: list[dict] = [
        {
            "year": year,
            "sales": s,
            "operating_profit": op,
            "interest": 50.0,
            "profit_before_tax": 200.0,
            "net_profit": 200.0,
            "eps": e,
        }
        for year, s, op, e in zip(years, sales, op_profit, eps, strict=True)
    ]
    profit_loss.append(
        {"year": "TTM", "sales": 2100.0, "net_profit": 220.0, "eps": 42.0}
    )
    balance_sheet = [
        {"year": year, "equity_capital": 100.0, "reserves": 900.0, "borrowings": 0.0}
        for year in years
    ]
    # Trailing half-year column has no P&L partner and must be ignored.
    balance_sheet.append(
        {"year": "Sep 2025", "equity_capital": 1.0, "reserves": 1.0, "borrowings": 0.0}
    )
    quarters = ["Sep 2024", "Dec 2024", "Mar 2025", "Jun 2025", "Sep 2025"]
    quarterly = [
        {"date": q, "net_profit": np}
        for q, np in zip(quarters, [40.0, 45.0, 50.0, 55.0, 60.0], strict=True)
    ]
    return {
        "summary": {"ratios": {"market_cap": 5000.0, "stock_p_e": 30.0}},
        "profit_loss": profit_loss,
        "balance_sheet": balance_sheet,
        "quarterly_results": quarterly,
    }


_GARP_ROW_KEYS = (
    "peg",
    "sales_growth_5y",
    "operating_profit_growth",
    "eps_growth_5y",
    "roe_5y",
    "roce_or_roic",
    "quarterly_profit_growth",
)


def test_load_garp_row_india_returns_scorer_shape(monkeypatch) -> None:
    sections = _india_sections()
    monkeypatch.setattr(
        garp_module,
        "cached_json_call",
        lambda ns, kp, *, ttl_seconds, refresh, fetch: fetch(),
    )
    monkeypatch.setattr(garp_module, "_fetch_india_sections", lambda sym: sections)

    row = garp_module.load_garp_row("RELIANCE", "india", cache_ttl=None, refresh=False)

    assert row is not None
    assert row["name"] == "RELIANCE"
    assert set(_GARP_ROW_KEYS) <= set(row)
    assert row["market_cap"] == pytest.approx(5000.0)
    assert row["sales"] == pytest.approx(2100.0)  # TTM, not the last fiscal year
    assert row["sales_growth_5y"] == pytest.approx(_INDIA_SALES_CAGR)
    assert row["eps_growth_5y"] == pytest.approx(_INDIA_EPS_CAGR)
    assert row["peg"] == pytest.approx(_INDIA_PEG)
    assert row["operating_profit_growth"] == pytest.approx(25.0)
    # Net profit / (equity capital + reserves), last five fiscal years.
    assert row["roe_5y"] == pytest.approx(20.0)
    # (PBT + interest) / (equity + borrowings), last three fiscal years.
    assert row["roce_or_roic"] == pytest.approx(25.0)
    # Latest quarter against the same quarter a year earlier.
    assert row["quarterly_profit_growth"] == pytest.approx(50.0)


def test_india_garp_passes_on_openscreener_tables() -> None:
    row = garp_module._india_row("AAA", "Alpha", _india_sections())

    assert garp_module._passes_garp(row, INDIA_THRESHOLDS) is True


def test_india_row_averages_opening_and_closing_balances() -> None:
    # screener.in divides by the mean of the prior and current year-end
    # balances; closing balances alone understate a growing company.
    sections = _india_sections()
    for i, bs in enumerate(sections["balance_sheet"][:-1]):
        bs["reserves"] = 900.0 + 100.0 * i
        bs["borrowings"] = 100.0 * i

    row = garp_module._india_row("AAA", "", sections)

    # Equity at year-ends 2..6 is 1200..1600, so the five averages are
    # 1150..1550; net profit is 200 each year.
    equity_avgs = [1150.0, 1250.0, 1350.0, 1450.0, 1550.0]
    assert row.roe_5y == pytest.approx(sum(200.0 / e * 100.0 for e in equity_avgs) / 5)
    # Capital employed adds borrowings 100*i, so the last three averages are
    # 1350+350, 1450+450, 1550+550 against PBT + interest of 250.
    employed_avgs = [1700.0, 1900.0, 2100.0]
    assert row.roce_or_roic == pytest.approx(
        sum(250.0 / e * 100.0 for e in employed_avgs) / 3
    )


def test_india_row_reads_lender_table_columns() -> None:
    # Banks and NBFCs label the same lines revenue / financing_profit and keep
    # borrowing (singular) plus deposits on the balance sheet.
    sections = _india_sections()
    for row in sections["profit_loss"]:
        row["revenue"] = row.pop("sales")
        if "operating_profit" in row:
            row["financing_profit"] = row.pop("operating_profit")
    for bs in sections["balance_sheet"]:
        bs["borrowing"] = bs.pop("borrowings")
        bs["deposits"] = 1000.0

    row = garp_module._india_row("BANK", "", sections)

    assert row.sales == pytest.approx(2100.0)
    assert row.sales_growth_5y == pytest.approx(_INDIA_SALES_CAGR)
    assert row.operating_profit_growth == pytest.approx(25.0)
    # Deposits double capital employed: (200 + 50) / (1000 + 1000).
    assert row.roce_or_roic == pytest.approx(12.5)


def test_load_garp_row_india_non_dict_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        garp_module,
        "cached_json_call",
        lambda ns, kp, *, ttl_seconds, refresh, fetch: None,
    )
    out = garp_module.load_garp_row("RELIANCE", "india", cache_ttl=None, refresh=False)
    assert out is None


def test_load_garp_row_us_uses_fmp(monkeypatch) -> None:
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(
        garp_module, "_fetch_fmp_us_cached", lambda *a, **k: _fmp_payload()
    )

    def _no_yfinance(*a, **k):
        raise AssertionError("yfinance path must not run when FMP has data")

    monkeypatch.setattr(garp_module, "_us_row", _no_yfinance)

    row = garp_module.load_garp_row("AAPL", "us", cache_ttl=None, refresh=False)

    assert row is not None
    assert row["name"] == "AAPL"
    assert set(_GARP_ROW_KEYS) <= set(row)
    assert row["peg"] == pytest.approx(1.2)


def test_load_garp_row_us_falls_back_to_yfinance(monkeypatch) -> None:
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: None)
    monkeypatch.setattr(
        garp_module, "_us_row", lambda symbol, description: {"name": symbol, "peg": 2.0}
    )
    row = garp_module.load_garp_row("AAPL", "us", cache_ttl=None, refresh=False)
    assert row is not None
    assert row["name"] == "AAPL"
    assert row["peg"] == 2.0
    assert row["market_cap"] is None


def test_to_number_is_financials_to_number() -> None:
    from screener import financials

    assert garp_module.to_number is financials.to_number
    assert garp_module.to_number("1,234.5") == pytest.approx(1234.5)
    assert garp_module.to_number("12%") == pytest.approx(12.0)
    assert garp_module.to_number(None) is None
    assert garp_module.to_number("x") is None
