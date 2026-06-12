from __future__ import annotations

import json
import urllib.parse

import pandas as pd

from screener import cache
from screener import insiders as insiders_module
from screener.backtester.data import tv_to_yf
from screener.insiders import (
    _aggregate_fmp_transactions,
    _fetch_fmp_insider_one,
    _row_value,
    filter_promoter_increased,
)


def _txn(
    days_ago: int,
    disposition: str,
    shares: float,
    transaction_type: str | None = None,
) -> dict:
    date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days_ago)).date()
    # Default to a genuine open-market purchase/sale so existing tests keep
    # exercising the buy/sell paths under the stricter transactionType logic.
    if transaction_type is None:
        transaction_type = "P-Purchase" if disposition == "A" else "S-Sale"
    return {
        "transactionDate": date.isoformat(),
        "acquistionOrDisposition": disposition,
        "transactionType": transaction_type,
        "securitiesTransacted": shares,
    }


def test_aggregate_nets_buys_against_sells_within_window():
    agg = _aggregate_fmp_transactions(
        [
            _txn(10, "A", 1000),
            _txn(20, "A", 500),
            _txn(30, "D", 200),
        ]
    )
    assert agg == {
        "fmp_net_shares_6m": 1300.0,
        "fmp_buy_shares_6m": 1500.0,
        "fmp_sell_shares_6m": 200.0,
        "fmp_buy_trans_6m": 2,
        "fmp_sell_trans_6m": 1,
    }


def test_row_value_returns_none_when_yfinance_schema_is_missing_label_column():
    df = pd.DataFrame({"Breakdown": ["Net Shares Purchased (Sold)"], "Shares": [10]})

    assert _row_value(df, "Net Shares Purchased (Sold)", "Shares") is None


def test_aggregate_excludes_transactions_outside_window():
    agg = _aggregate_fmp_transactions([_txn(10, "A", 100), _txn(400, "A", 9999)])
    assert agg["fmp_buy_shares_6m"] == 100.0
    assert agg["fmp_buy_trans_6m"] == 1


def test_aggregate_returns_none_when_no_dated_rows():
    assert _aggregate_fmp_transactions([]) is None
    assert _aggregate_fmp_transactions([_txn(500, "A", 100)]) is None


def test_aggregate_excludes_awards_and_non_purchase_acquisitions():
    # An "A" acquisition that is an Award/Gift/Option-exercise must NOT count
    # as a buy — only P-Purchase rows are genuine open-market buys.
    agg = _aggregate_fmp_transactions(
        [
            _txn(5, "A", 5000, transaction_type="A-Award"),
            _txn(6, "A", 3000, transaction_type="G-Gift"),
            _txn(7, "A", 2000, transaction_type="M-Exempt"),
            _txn(8, "A", 1000, transaction_type="P-Purchase"),
        ]
    )
    assert agg == {
        "fmp_net_shares_6m": 1000.0,
        "fmp_buy_shares_6m": 1000.0,
        "fmp_sell_shares_6m": 0.0,
        "fmp_buy_trans_6m": 1,
        "fmp_sell_trans_6m": 0,
    }


def test_aggregate_excludes_non_sale_dispositions_and_handles_missing_type():
    # An "D" disposition that is not an S-Sale (e.g. F-Payment of Exercise)
    # must not count as a sell; a missing transactionType is skipped, not raised.
    agg = _aggregate_fmp_transactions(
        [
            _txn(5, "D", 4000, transaction_type="F-Payment of Exercise"),
            _txn(6, "A", 4000, transaction_type=None) | {"transactionType": None},
            _txn(7, "D", 250, transaction_type="S-Sale"),
            _txn(8, "A", 750, transaction_type="P-Purchase"),
        ]
    )
    assert agg == {
        "fmp_net_shares_6m": 500.0,
        "fmp_buy_shares_6m": 750.0,
        "fmp_sell_shares_6m": 250.0,
        "fmp_buy_trans_6m": 1,
        "fmp_sell_trans_6m": 1,
    }


def test_aggregate_skips_non_numeric_shares_and_uses_filing_date():
    filing = (pd.Timestamp.now().normalize() - pd.Timedelta(days=4)).date()
    agg = _aggregate_fmp_transactions(
        [
            _txn(5, "A", 100, transaction_type="P-Purchase")
            | {"securitiesTransacted": "not-a-number"},
            {
                "filingDate": filing.isoformat(),
                "acquistionOrDisposition": "A",
                "transactionType": "P-Purchase",
                "securitiesTransacted": "250",
            },
        ]
    )

    assert agg == {
        "fmp_net_shares_6m": 250.0,
        "fmp_buy_shares_6m": 250.0,
        "fmp_sell_shares_6m": 0.0,
        "fmp_buy_trans_6m": 1,
        "fmp_sell_trans_6m": 0,
    }


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _fmp_row(days_ago: int, shares: float = 10.0) -> dict:
    d = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days_ago)).date()
    return {
        "transactionDate": d.isoformat(),
        "acquistionOrDisposition": "A",
        "transactionType": "P-Purchase",
        "securitiesTransacted": shares,
    }


def test_fetch_fmp_warns_when_page_cap_may_truncate(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)

    def fake_urlopen(req, timeout=20):
        page = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)["page"][0]
        )
        return _Resp([_fmp_row(page), _fmp_row(page + 1)])

    monkeypatch.setattr(insiders_module.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("WARNING", logger="screener.insiders"):
        out = _fetch_fmp_insider_one(
            "AAA", "AAA", api_key="key", cache_ttl=None, refresh=True
        )

    assert out is not None
    assert "may be truncated at 10 pages" in caplog.text
    # Truncation is surfaced to callers as a flagged field, not just a log.
    assert out["fmp_truncated"] is True


def test_fetch_fmp_stops_after_out_of_window_page(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    calls: list[int] = []

    def fake_urlopen(req, timeout=20):
        page = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)["page"][0]
        )
        calls.append(page)
        if page == 0:
            return _Resp([_fmp_row(1), _fmp_row(2)])
        if page == 1:
            return _Resp([_fmp_row(3), _fmp_row(400)])
        return _Resp([_fmp_row(4)])

    monkeypatch.setattr(insiders_module.urllib.request, "urlopen", fake_urlopen)
    out = _fetch_fmp_insider_one(
        "AAA", "AAA", api_key="key", cache_ttl=None, refresh=True
    )

    assert out is not None
    assert calls == [0, 1]
    assert out["fmp_truncated"] is False


def test_tv_to_yf_bse_maps_to_bo_not_ns():
    """BSE-prefixed tickers must resolve to .BO, not the old insiders .NS bug."""
    assert tv_to_yf("BSE:TCS", "india") == "TCS.BO"


def test_tv_to_yf_nse_maps_to_ns():
    assert tv_to_yf("NSE:RELIANCE", "india") == "RELIANCE.NS"


def test_tv_to_yf_bare_india_symbol_defaults_to_ns():
    assert tv_to_yf("RELIANCE", "india") == "RELIANCE.NS"


def test_tv_to_yf_presuffixed_symbol_passes_through():
    assert tv_to_yf("RELIANCE.NS", "india") == "RELIANCE.NS"


def test_tv_to_yf_us_symbol_strips_exchange():
    assert tv_to_yf("NASDAQ:AAPL", "us") == "AAPL"


def test_tv_to_yf_lowercases_input():
    assert tv_to_yf("nse:reliance", "india") == "RELIANCE.NS"


def test_us_filter_prefers_fmp_and_falls_back_to_yfinance():
    df = pd.DataFrame(
        [
            # FMP positive -> kept on FMP signal
            {"name": "AAA", "fmp_net_shares_6m": 500.0, "yf_net_shares_6m": -10.0},
            # FMP negative -> dropped despite positive yfinance
            {"name": "BBB", "fmp_net_shares_6m": -100.0, "yf_net_shares_6m": 50.0},
            # FMP missing -> falls back to yfinance (positive -> kept)
            {"name": "CCC", "fmp_net_shares_6m": None, "yf_net_shares_6m": 20.0},
            # FMP zero is no signal -> falls back to yfinance (positive -> kept)
            {"name": "DDD", "fmp_net_shares_6m": 0.0, "yf_net_shares_6m": 30.0},
        ]
    )

    out = filter_promoter_increased(df, market="us")

    assert sorted(out["name"]) == ["AAA", "CCC", "DDD"]
