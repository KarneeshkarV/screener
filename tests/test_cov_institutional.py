"""Offline coverage tests for core utility/CLI modules.

Drives several small modules to (near) 100% line coverage without any
network access. All external seams — Turso/libSQL client, price fetchers,
HTTP/Wikipedia/NSE calls, the FMP provider — are stubbed or monkeypatched.
"""

from __future__ import annotations


import json


import pandas as pd


import pytest


from click.testing import CliRunner


from screener import history as history_mod


from screener import universes as universes_mod


from screener.cli import cli


class _FakeRows:
    def __init__(self, rows):
        self.rows = rows


class _FakeClient:
    def __init__(self, select_rows=None):
        self.statements: list[tuple[str, list[object] | None]] = []
        self.closed = False
        self._select_rows = select_rows or []

    def execute(self, stmt: str, args: list[object] | None = None):
        self.statements.append((stmt, args))
        if stmt.lstrip().upper().startswith("SELECT"):
            return _FakeRows(self._select_rows)
        return _FakeRows([])

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


class _StubQuery:
    """Mimics tradingview_screener.Query's fluent chaining."""

    def __init__(self, result):
        self._result = result

    def set_markets(self, *a):
        return self

    def select(self, *a):
        return self

    def where(self, *a):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a):
        return self

    def get_scanner_data(self):
        return self._result


def _scanner_frame():
    return pd.DataFrame(
        {
            "name": ["AAA", "BBB"],
            "description": ["Alpha", "Beta"],
            "close": [100.0, 50.0],
            "change": [1.0, -1.0],
            "volume": [10000.0, 20000.0],
            "market_cap_basic": [1e9, 2e9],
            "EMA5": [101, 51],
            "EMA20": [100, 50],
            "EMA100": [98, 48],
            "EMA200": [95, 45],
            "RSI": [60, 40],
        }
    )


@pytest.fixture
def universes_dir(tmp_path, monkeypatch):
    d = tmp_path / "universes"
    monkeypatch.setattr(universes_mod, "CACHE_DIR", d)
    return d


class _Resp:
    def __init__(self, text=None, status=None):
        self.text = text or ""
        self._status = status

    def raise_for_status(self):
        if self._status:
            raise RuntimeError("bad status")


def test_institutional_no_api_key(monkeypatch):
    import screener.commands.institutional as cmd

    monkeypatch.setattr(cmd, "resolve_api_key", lambda: None)
    res = CliRunner().invoke(cli, ["institutional", "--tickers", "AAPL"])
    assert res.exit_code != 0
    assert "FMP_API_KEY is not set" in res.output


def test_institutional_renders_results(monkeypatch):
    import screener.institutional as inst_mod
    import screener.commands.institutional as cmd

    monkeypatch.setattr(cmd, "resolve_api_key", lambda: "key")
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "holders": [10],
            "total_shares": [1000.0],
            "qoq_change_shares": [50.0],
            "qoq_change_pct": [5.0],
        }
    )
    monkeypatch.setattr(inst_mod, "fetch_fmp_institutional", lambda *a, **k: df)
    res = CliRunner().invoke(
        cli, ["institutional", "--tickers", "AAPL,MSFT"], env={"COLUMNS": "250"}
    )
    assert res.exit_code == 0
    # MSFT missing -> reported on stderr (mixed into output by CliRunner).
    assert "AAPL" in res.output


def test_fetch_fmp_institutional_one_empty_rows(monkeypatch):
    import screener.institutional as inst_mod

    captured = {}

    def fake_provider_fetch(key, fn, **kwargs):
        captured["result"] = fn()
        return captured["result"]

    monkeypatch.setattr(
        inst_mod._FMP_INSTITUTIONAL_PROVIDER, "fetch", fake_provider_fetch
    )

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps([]).encode()  # empty list -> not rows -> None

    monkeypatch.setattr(inst_mod.urllib.request, "urlopen", lambda *a, **k: _R())
    out = inst_mod._fetch_fmp_institutional_one(
        "AAPL", api_key="k", cache_ttl=10, refresh=False
    )
    assert out is None


def test_fetch_fmp_institutional_one_aggregation_none(monkeypatch):
    """Non-empty rows that aggregate to None (no numeric shares) -> None."""
    import screener.institutional as inst_mod

    monkeypatch.setattr(
        inst_mod._FMP_INSTITUTIONAL_PROVIDER, "fetch", lambda key, fn, **kw: fn()
    )

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            # One row, but its 'shares' is non-numeric so aggregation yields None.
            return json.dumps([{"holder": "F", "shares": "not-a-number"}]).encode()

    monkeypatch.setattr(inst_mod.urllib.request, "urlopen", lambda *a, **k: _R())
    out = inst_mod._fetch_fmp_institutional_one(
        "AAPL", api_key="k", cache_ttl=10, refresh=False
    )
    assert out is None
