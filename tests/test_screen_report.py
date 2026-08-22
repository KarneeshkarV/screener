"""Screen HTML report: charts must have a pixel height and numbers stay readable."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from screener.commands.screen_report import render_screen_report


def _screen_rows(n: int = 30) -> pd.DataFrame:
    tickers = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "BAJFINANCE",
        "BHARTIARTL",
        "LT",
    ]
    rows = []
    for i in range(n):
        rows.append(
            {
                "name": tickers[i % len(tickers)]
                + ("" if i < len(tickers) else str(i)),
                "description": "Alpha Corp",
                "close": 0.57 if i == 0 else (1_105_000.0 if i == 1 else 100.0 + i),
                "change": 14.84 - i * 0.4,
                "volume": 1.0 if i == 0 else (4.255241e11 if i == 1 else 1_500_000 + i),
                "market_cap_basic": (
                    1.938e7 if i == 0 else (1.200652e13 if i == 1 else 5_000_000_000)
                ),
                "setup_score": 50.0 + (i % 20),
                "RSI": 40.0 + (i % 10),
            }
        )
    return pd.DataFrame(rows)


def _section(html: str, start: str, end: str) -> str:
    i = html.index(start)
    j = html.index(end, i)
    return html[i:j]


def test_screen_report_avoids_scientific_notation_and_collapsed_charts(tmp_path: Path):
    path = render_screen_report(
        _screen_rows(),
        total=2828,
        market="india",
        criteria_name="momentum_12_1",
        output_file=tmp_path / "screen.html",
        order_by="setup_score",
    )
    html = path.read_text(encoding="utf-8")

    numeric = _section(html, 'id="numeric-summary"', 'id="screen-results"')
    results = _section(html, 'id="screen-results"', 'id="added-tickers"')
    assert html.index('id="screen-results"') < html.index('id="added-tickers"')
    assert html.index('id="added-tickers"') < html.index('id="removed-tickers"')
    assert html.index('id="removed-tickers"') < html.index('id="report-notes"')
    sci = re.compile(r"\d\.?\d*e[+-]\d+", re.I)
    assert sci.search(numeric) is None
    assert sci.search(results) is None
    assert "12.01 Lakh Cr" in numeric
    assert "12.01 Lakh Cr" in results
    assert "1.94 Cr" in results
    assert "market_cap_basic" not in results
    assert "Mkt Cap" in results
    assert "425.52B" in numeric
    assert "+14.84%" in numeric
    assert "Symbol" in results

    assert "name=%{x}" not in html
    assert "Ticker=%{y}" in html
    assert '"orientation":"h"' in html
    assert 'id="screen-top-change"' in html
    assert 'style="height:320px;width:100%"' in html
    assert 'style="height:630px;width:100%"' in html


def test_screen_report_us_market_cap_uses_billions(tmp_path: Path):
    path = render_screen_report(
        _screen_rows(n=2),
        total=2,
        market="us",
        criteria_name="ema",
        output_file=tmp_path / "screen-us.html",
    )
    html = path.read_text(encoding="utf-8")
    results = _section(html, 'id="screen-results"', 'id="added-tickers"')
    assert "12.01T" in results
    assert "19.4M" in results
    assert " Lakh Cr" not in results
    assert " Cr" not in results
