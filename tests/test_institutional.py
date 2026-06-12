"""Tests for the FMP institutional ownership module and CLI — offline."""

from __future__ import annotations

import json

import pandas as pd
from click.testing import CliRunner

from screener import cache
from screener import insiders as insiders_module
from screener import institutional as institutional_module
from screener.cli import cli
from screener.institutional import (
    _aggregate_institutional_holders,
    _fetch_fmp_institutional_one,
    fetch_fmp_institutional,
)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _holder(shares, change=None, holder="Fund"):
    row = {"holder": holder, "shares": shares, "dateReported": "2026-03-31"}
    if change is not None:
        row["change"] = change
    return row


# ── aggregation ─────────────────────────────────────────────────────────────


def test_aggregate_sums_holders_shares_and_qoq_change():
    agg = _aggregate_institutional_holders(
        [
            _holder(1_000, change=200),
            _holder(500, change=-100),
            _holder(250, change=0),
        ]
    )
    assert agg == {
        "holders": 3,
        "total_shares": 1750.0,
        "qoq_change_shares": 100.0,
        # prev total = 1750 - 100 = 1650
        "qoq_change_pct": 100.0 / 1650.0 * 100.0,
    }


def test_aggregate_without_change_field_reports_none_qoq():
    agg = _aggregate_institutional_holders([_holder(1_000), _holder(500)])
    assert agg == {
        "holders": 2,
        "total_shares": 1500.0,
        "qoq_change_shares": None,
        "qoq_change_pct": None,
    }


def test_aggregate_skips_bad_rows_and_handles_empty():
    assert _aggregate_institutional_holders([]) is None
    assert _aggregate_institutional_holders([{"shares": "not-a-number"}]) is None

    agg = _aggregate_institutional_holders(
        [
            {"shares": "not-a-number", "change": 999},
            _holder(100, change="also-bad"),
            _holder(400, change=50),
        ]
    )
    assert agg["holders"] == 2
    assert agg["total_shares"] == 500.0
    assert agg["qoq_change_shares"] == 50.0


def test_aggregate_pct_is_none_when_previous_total_not_positive():
    # All shares are new this quarter -> previous total is 0.
    agg = _aggregate_institutional_holders([_holder(300, change=300)])
    assert agg["qoq_change_shares"] == 300.0
    assert agg["qoq_change_pct"] is None


# ── fetch (stubbed urlopen, no network) ─────────────────────────────────────


def test_fetch_one_returns_summary_from_stubbed_payload(monkeypatch, fake_provider):
    # Inject the provider seam instead of monkeypatching cache.CACHE_ROOT: the
    # FakeProvider runs the fetch directly (no disk cache, no resilience).
    monkeypatch.setattr(
        institutional_module, "_FMP_INSTITUTIONAL_PROVIDER", fake_provider()
    )
    seen: list[str] = []

    def fake_urlopen(req, timeout=20):
        seen.append(req.full_url)
        return _Resp([_holder(1_000, change=250), _holder(2_000, change=-50)])

    monkeypatch.setattr(institutional_module.urllib.request, "urlopen", fake_urlopen)

    out = _fetch_fmp_institutional_one(
        "AAPL", api_key="key", cache_ttl=None, refresh=True
    )

    assert out == {
        "symbol": "AAPL",
        "holders": 2,
        "total_shares": 3000.0,
        "qoq_change_shares": 200.0,
        "qoq_change_pct": 200.0 / 2800.0 * 100.0,
    }
    assert seen and "institutional-holder/AAPL" in seen[0]
    assert "apikey=key" in seen[0]


def test_fetch_one_returns_none_for_empty_or_non_list_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)

    monkeypatch.setattr(
        institutional_module.urllib.request,
        "urlopen",
        lambda req, timeout=20: _Resp([]),
    )
    assert (
        _fetch_fmp_institutional_one(
            "ZZZZ", api_key="key", cache_ttl=None, refresh=True
        )
        is None
    )

    monkeypatch.setattr(
        institutional_module.urllib.request,
        "urlopen",
        lambda req, timeout=20: _Resp({"Error Message": "Invalid API KEY."}),
    )
    assert (
        _fetch_fmp_institutional_one(
            "AAPL", api_key="key", cache_ttl=None, refresh=True
        )
        is None
    )


def test_fetch_many_skips_symbols_without_data(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)

    def fake_urlopen(req, timeout=20):
        if "institutional-holder/AAPL" in req.full_url:
            return _Resp([_holder(1_000, change=100)])
        return _Resp([])

    monkeypatch.setattr(institutional_module.urllib.request, "urlopen", fake_urlopen)

    df = fetch_fmp_institutional(
        ["AAPL", "ZZZZ"], api_key="key", cache_ttl=None, refresh=True
    )

    assert list(df["symbol"]) == ["AAPL"]
    assert df.iloc[0]["holders"] == 1


def test_fetch_many_empty_symbol_list_returns_empty_frame():
    assert fetch_fmp_institutional([], api_key="key").empty


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_errors_gracefully_without_api_key(monkeypatch):
    monkeypatch.setattr(insiders_module, "_fmp_api_key", lambda: None)
    runner = CliRunner()
    res = runner.invoke(cli, ["institutional", "-m", "us", "--tickers", "AAPL"])
    assert res.exit_code != 0
    assert "FMP_API_KEY" in res.output


def test_cli_ranks_by_qoq_change_and_reports_missing(monkeypatch):
    monkeypatch.setattr(insiders_module, "_fmp_api_key", lambda: "key")

    def fake_fetch(symbols, *, api_key, max_workers=8, cache_ttl=86400, refresh=False):
        assert api_key == "key"
        assert symbols == ["AAPL", "MSFT", "ZZZZ"]
        return pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "holders": 2,
                    "total_shares": 3000.0,
                    "qoq_change_shares": 200.0,
                    "qoq_change_pct": 7.14,
                },
                {
                    "symbol": "MSFT",
                    "holders": 3,
                    "total_shares": 9000.0,
                    "qoq_change_shares": 500.0,
                    "qoq_change_pct": 5.88,
                },
            ]
        )

    monkeypatch.setattr(institutional_module, "fetch_fmp_institutional", fake_fetch)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["institutional", "-m", "us", "--tickers", "aapl,msft,zzzz", "--csv"],
        catch_exceptions=False,
    )

    assert res.exit_code == 0
    assert "No institutional data for: ZZZZ" in res.output
    lines = [ln for ln in res.output.splitlines() if "," in ln]
    header = lines[0].split(",")
    assert header[:5] == [
        "symbol",
        "holders",
        "total_shares",
        "qoq_change_shares",
        "qoq_change_pct",
    ]
    # Ranked by QoQ share change descending: MSFT (+500) before AAPL (+200).
    assert lines[1].startswith("MSFT")
    assert lines[2].startswith("AAPL")


def test_cli_table_output_and_all_missing(monkeypatch):
    monkeypatch.setattr(insiders_module, "_fmp_api_key", lambda: "key")
    monkeypatch.setattr(
        institutional_module,
        "fetch_fmp_institutional",
        lambda symbols, **kwargs: pd.DataFrame(),
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["institutional", "--tickers", "ZZZZ"])

    assert res.exit_code == 0
    assert "No institutional data for: ZZZZ" in res.output
    assert "No institutional ownership data returned." in res.output


def test_cli_rejects_market_other_than_us():
    runner = CliRunner()
    res = runner.invoke(cli, ["institutional", "-m", "india", "--tickers", "AAPL"])
    assert res.exit_code != 0
