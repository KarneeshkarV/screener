from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import cache
from screener import garp as garp_module
from screener.cli import cli
from screener.garp import (
    INDIA_THRESHOLDS,
    add_garp_score,
    _fmp_us_row,
    _passes_garp,
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
    assert set(fmp_row) == set(yf_row)
    for key, expected in yf_row.items():
        if isinstance(expected, float):
            assert fmp_row[key] == pytest.approx(expected), key
        else:
            assert fmp_row[key] == expected, key


def test_screen_us_garp_uses_fmp_when_key_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(garp_module, "_fmp_api_key", lambda: "test-key")
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
    monkeypatch.setattr(garp_module, "_fmp_api_key", lambda: None)

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
    monkeypatch.setattr(garp_module, "_fmp_api_key", lambda: "test-key")
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
