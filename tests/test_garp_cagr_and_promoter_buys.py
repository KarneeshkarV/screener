"""Regressions for US GARP CAGR spans and India promoter-buy fallbacks."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import garp as garp_module
from screener import insiders as insiders_module
from screener import scanner as scanner_module
from screener.cli import cli
from screener.commands import insiders as insiders_command
from screener.garp import _us_row
from screener.insiders import filter_promoter_increased

# ── US GARP: CAGR exponent follows the reported span ────────────────────────

_REVENUE_BASE = 100.0
_GROWTH = 0.20


def _annual_frames(
    periods: int, *, tax: float | None = 0.21
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance-style statements whose columns run newest to oldest in value."""
    dates = pd.date_range("2025-12-31", periods=periods, freq="YE")
    revenue = [
        _REVENUE_BASE * (1.0 + _GROWTH) ** (periods - 1 - i) for i in range(periods)
    ]
    operating = [value * 0.20 for value in revenue]
    net_income = [value * 0.10 for value in revenue]

    income_rows = [revenue, operating, net_income, operating]
    index = ["Total Revenue", "Operating Income", "Net Income", "EBIT"]
    if tax is not None:
        index.append("Tax Rate For Calcs")
        income_rows.append([tax] * periods)
    income = pd.DataFrame(income_rows, index=index, columns=dates)
    balance = pd.DataFrame(
        [[1000.0] * periods, [0.0] * periods],
        index=["Stockholders Equity", "Total Debt"],
        columns=dates,
    )
    return income, balance


def _patch_ticker(monkeypatch, periods: int, *, tax: float | None = 0.21) -> None:
    income, balance = _annual_frames(periods, tax=tax)

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {
                "marketCap": 5.0e9,
                "trailingPegRatio": 1.1,
                "shortName": "Alpha",
            }
            self.income_stmt = income
            self.balance_sheet = balance
            self.earnings_estimate = None

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)


def test_us_row_three_intervals_uses_three_year_exponent(monkeypatch) -> None:
    # yfinance's default four annual columns are three growth intervals.
    _patch_ticker(monkeypatch, periods=4)

    row = _us_row("AAA", "Alpha")

    assert row.sales_growth_5y == pytest.approx(20.0)
    assert row.eps_growth_5y == pytest.approx(20.0)
    assert row.operating_profit_growth == pytest.approx(20.0)


def test_us_row_four_intervals_uses_four_year_exponent(monkeypatch) -> None:
    _patch_ticker(monkeypatch, periods=5)

    row = _us_row("AAA", "Alpha")

    assert row.sales_growth_5y == pytest.approx(20.0)
    assert row.eps_growth_5y == pytest.approx(20.0)
    assert row.operating_profit_growth == pytest.approx(20.0)


def test_us_row_single_period_reports_no_growth(monkeypatch) -> None:
    _patch_ticker(monkeypatch, periods=1)

    row = _us_row("AAA", "Alpha")

    assert row.sales_growth_5y is None
    assert row.eps_growth_5y is None
    assert row.operating_profit_growth is None


def test_us_row_preserves_zero_tax_rate(monkeypatch) -> None:
    # NOPAT at 0% tax = EBIT = 20, over 1000 invested capital -> 2.0% ROIC.
    # The old ``tax or 0.21`` fallback overwrote the valid 0.0 with 21%.
    _patch_ticker(monkeypatch, periods=1, tax=0.0)

    row = _us_row("AAA", "Alpha")

    assert row.roce_or_roic == pytest.approx(2.0)


def test_india_garp_universe_prefilter_uses_inr_not_crore(monkeypatch) -> None:
    captured: dict = {}

    class FakeField:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other: object) -> tuple[str, str, object]:
            return ("eq", self.name, other)

        def __ge__(self, other: object) -> tuple[str, str, object]:
            return ("ge", self.name, other)

        def isin(self, values: object) -> tuple[str, str, object]:
            return ("isin", self.name, values)

    def fake_scan(*, filters, **kwargs):
        captured["filters"] = filters
        return 0, pd.DataFrame(), None

    import tradingview_screener

    monkeypatch.setattr(tradingview_screener, "col", FakeField)
    monkeypatch.setattr(garp_module, "scan", fake_scan)

    garp_module.load_garp_universe("india", 10, cache_ttl=None, refresh=False)

    # 1,000 crore = 1e10 rupees; TradingView's India market cap is raw INR.
    assert ("ge", "market_cap_basic", 1e10) in captured["filters"]
    assert ("ge", "market_cap_basic", 1000.0) not in captured["filters"]


# ── promoter-buys: India fallback paths and CSV stdout ─────────────────────

_UNIVERSE = pd.DataFrame(
    [
        {
            "name": "AAA",
            "description": "Alpha",
            "close": 100.0,
            "change": 1.0,
            "volume": 1000.0,
            "market_cap_basic": 5.0e9,
        },
        {
            "name": "BBB",
            "description": "Beta",
            "close": 200.0,
            "change": -1.0,
            "volume": 900.0,
            "market_cap_basic": 6.0e9,
        },
    ]
)

_YF_FRAME = pd.DataFrame(
    [
        {
            "name": "AAA",
            "yf_symbol": "AAA.NS",
            "yf_net_shares_6m": 100.0,
            "yf_net_pct_6m": 0.5,
            "yf_total_held": 20000.0,
            "yf_buy_trans_6m": 2.0,
            "yf_sell_trans_6m": 0.0,
        },
        {
            "name": "BBB",
            "yf_symbol": "BBB.NS",
            "yf_net_shares_6m": -50.0,
            "yf_net_pct_6m": -0.2,
            "yf_total_held": 10000.0,
            "yf_buy_trans_6m": 0.0,
            "yf_sell_trans_6m": 1.0,
        },
    ]
)

_OS_FRAME = pd.DataFrame(
    [
        {
            "name": "AAA",
            "promoter_pct_prev": 50.0,
            "promoter_change": 0.5,
            "latest_quarter": "Sep 2025",
            "fii_pct_latest": 5.0,
            "dii_pct_latest": 6.0,
        }
    ]
)


def _run_kwargs(*, output_csv: bool = False) -> dict:
    return {
        "market": "india",
        "universe_size": 2,
        "limit": 10,
        "min_change_pct": 0.0,
        "min_yf_net_pct": None,
        "require_both": False,
        "min_market_cap": None,
        "workers": 1,
        "output_csv": output_csv,
        "refresh": False,
        "cache_ttl": "15m",
    }


def _patch_sources(monkeypatch, *, os_df: pd.DataFrame, yf_df: pd.DataFrame) -> None:
    monkeypatch.setattr(
        scanner_module,
        "get_scanner_data_cached",
        lambda *a, **k: (len(_UNIVERSE), _UNIVERSE.copy()),
    )
    monkeypatch.setattr(
        insiders_module, "fetch_openscreener_promoters", lambda *a, **k: os_df
    )
    monkeypatch.setattr(
        insiders_module, "fetch_yfinance_insiders", lambda *a, **k: yf_df
    )


def test_india_filter_tolerates_missing_promoter_change() -> None:
    # yfinance-only fallback: no promoter_change column at all.
    out = filter_promoter_increased(_YF_FRAME, market="india")

    assert out.empty


def test_india_filter_tolerates_missing_yfinance_columns() -> None:
    out = filter_promoter_increased(_OS_FRAME, market="india", require_both=True)

    assert out.empty


def test_promoter_buys_india_yfinance_fallback_does_not_raise(monkeypatch) -> None:
    _patch_sources(monkeypatch, os_df=pd.DataFrame(), yf_df=_YF_FRAME)

    # Must not raise KeyError: 'promoter_change'.
    insiders_command.run_promoter_buys(**_run_kwargs())


def test_promoter_buys_india_openscreener_only_does_not_raise(monkeypatch) -> None:
    _patch_sources(monkeypatch, os_df=_OS_FRAME, yf_df=pd.DataFrame())
    shown: list[pd.DataFrame] = []
    monkeypatch.setattr(
        insiders_command, "print_insider_results", lambda df, *a, **k: shown.append(df)
    )

    # Must not raise KeyError: 'yf_net_pct_6m' in the sort.
    insiders_command.run_promoter_buys(**_run_kwargs())

    assert len(shown) == 1
    assert list(shown[0]["name"]) == ["AAA"]


def test_promoter_buys_india_csv_stdout_is_clean(monkeypatch) -> None:
    _patch_sources(monkeypatch, os_df=_OS_FRAME, yf_df=pd.DataFrame())

    result = CliRunner().invoke(cli, ["promoter-buys", "-m", "india", "--csv"])

    assert result.exit_code == 0, result.output
    parsed = pd.read_csv(io.StringIO(result.stdout))
    assert list(parsed["name"]) == ["AAA"]
    # Status/progress lines belong on stderr so stdout stays parseable.
    assert "Universe:" in result.stderr
    assert "Universe:" not in result.stdout
