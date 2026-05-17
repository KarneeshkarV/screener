from __future__ import annotations

import pandas as pd

from screener.insiders import (
    _aggregate_fmp_transactions,
    filter_promoter_increased,
)


def _txn(days_ago: int, disposition: str, shares: float) -> dict:
    date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days_ago)).date()
    return {
        "transactionDate": date.isoformat(),
        "acquistionOrDisposition": disposition,
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


def test_aggregate_excludes_transactions_outside_window():
    agg = _aggregate_fmp_transactions([_txn(10, "A", 100), _txn(400, "A", 9999)])
    assert agg["fmp_buy_shares_6m"] == 100.0
    assert agg["fmp_buy_trans_6m"] == 1


def test_aggregate_returns_none_when_no_dated_rows():
    assert _aggregate_fmp_transactions([]) is None
    assert _aggregate_fmp_transactions([_txn(500, "A", 100)]) is None


def test_us_filter_prefers_fmp_and_falls_back_to_yfinance():
    df = pd.DataFrame(
        [
            # FMP positive -> kept on FMP signal
            {"name": "AAA", "fmp_net_shares_6m": 500.0, "yf_net_shares_6m": -10.0},
            # FMP negative -> dropped despite positive yfinance
            {"name": "BBB", "fmp_net_shares_6m": -100.0, "yf_net_shares_6m": 50.0},
            # FMP missing -> falls back to yfinance (positive -> kept)
            {"name": "CCC", "fmp_net_shares_6m": None, "yf_net_shares_6m": 20.0},
        ]
    )

    out = filter_promoter_increased(df, market="us")

    assert sorted(out["name"]) == ["AAA", "CCC"]
