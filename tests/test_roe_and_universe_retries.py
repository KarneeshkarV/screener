from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import requests

from screener import resilience, universes
from screener.backtester import fundamentals

_SP500_HTML = """
<html><body>
<table>
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td>AAA</td><td>Alpha Corp</td></tr>
</table>
</body></html>
"""

_NIFTY_CSV = "Symbol\nALPHA\nBETA\n"

_SENSEX_HTML = (
    "<table><tr><th>Company</th><th>Symbol</th></tr>"
    + "".join(f"<tr><td>Company {i}</td><td>STOCK{i}.BO</td></tr>" for i in range(30))
    + "</table>"
)


def _fmp_payload(return_on_equity: float) -> dict[str, Any]:
    return {
        "income": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-05", "revenue": 100.0}
        ],
        "balance": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-20", "totalDebt": 40.0}
        ],
        "ratios": [{"date": "2023-12-31", "returnOnEquity": return_on_equity}],
    }


@pytest.mark.parametrize(
    ("raw_roe", "expected"),
    [
        (2.05, 205.0),
        (-2.5, -250.0),
        (0.18, 18.0),
    ],
)
def test_fmp_roe_is_a_decimal_ratio_even_above_200_percent(
    raw_roe: float, expected: float
) -> None:
    frame = fundamentals._normalize_fmp_payload(
        _fmp_payload(raw_roe), fields=("roe_ttm",), lag_days=1
    )

    # The ratio row is dated from both statements of its fiscal period.
    assert frame.loc[pd.Timestamp("2024-02-21"), "roe_ttm"] == pytest.approx(expected)


def _http_response(text: str, status: int = 200) -> SimpleNamespace:
    def raise_for_status() -> None:
        if status >= 400:
            response = requests.Response()
            response.status_code = status
            raise requests.HTTPError(f"{status} error", response=response)

    return SimpleNamespace(text=text, raise_for_status=raise_for_status)


def _transient_get(
    success_text: str, failure_status: int
) -> tuple[Callable[..., SimpleNamespace], list[str]]:
    calls: list[str] = []

    def get(url: str, **kwargs: Any) -> SimpleNamespace:
        calls.append(url)
        if len(calls) == 1:
            return _http_response("", failure_status)
        return _http_response(success_text)

    return get, calls


def _patch_no_sleep_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    def call(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("sleep", lambda _seconds: None)
        kwargs.setdefault("rate_limiter", None)
        return resilience.call_with_resilience(*args, **kwargs)

    monkeypatch.setattr(universes, "call_with_resilience", call)


def _load_sp500() -> list[str]:
    return universes._read_sp500_html()[0]["Symbol"].tolist()


def _load_nifty50() -> list[str]:
    symbols, _source = universes._fetch_nifty50()
    return symbols


def _load_nifty500() -> list[str]:
    symbols, _source = universes._fetch_nifty500()
    return symbols


def _load_sensex() -> list[str]:
    symbols, _source = universes._fetch_sensex()
    return symbols


@pytest.mark.parametrize("failure_status", [429, 503])
@pytest.mark.parametrize(
    ("loader", "success_text", "expected"),
    [
        (_load_sp500, _SP500_HTML, ["AAA"]),
        (_load_nifty50, _NIFTY_CSV, ["ALPHA", "BETA"]),
        (_load_nifty500, _NIFTY_CSV, ["ALPHA.NS", "BETA.NS"]),
        (
            _load_sensex,
            _SENSEX_HTML,
            [f"STOCK{i}.BO" for i in range(30)],
        ),
    ],
)
def test_universe_fetchers_retry_transient_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_status: int,
    loader: Callable[[], list[str]],
    success_text: str,
    expected: list[str],
) -> None:
    _patch_no_sleep_resilience(monkeypatch)
    get, calls = _transient_get(success_text, failure_status)
    monkeypatch.setattr(universes.requests, "get", get)

    assert loader() == expected
    assert len(calls) == 2
