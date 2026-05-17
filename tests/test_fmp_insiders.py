from __future__ import annotations

import pandas as pd

from screener.insiders import (
    _aggregate_fmp_transactions,
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
