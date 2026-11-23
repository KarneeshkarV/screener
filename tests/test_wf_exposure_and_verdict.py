"""Regressions for walk-forward exposure and non-positive OOS verdicts.

B1: a stitched walk-forward ledger concatenates folds whose portfolios each
number lots from 1, so ``(ticker, open_seq)`` alone collides across folds and
merged disjoint OOS positions into one year-long interval. B6a: a strategy with
a non-positive OOS metric could still be reported as PASS when the raw return
kept the Monte Carlo 5th percentile above zero.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.metrics import _exposure
from screener.backtester.models import Trade
from screener.backtester.optimization import research_report as rr

_INDEX = pd.date_range("2024-01-02", "2024-12-31", freq="B")


def _trade(
    *,
    entry: str,
    exit_: str,
    open_seq: int = 1,
    ticker: str = "AAA",
) -> Trade:
    return Trade(
        ticker=ticker,
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=pd.Timestamp(entry).date(),
        entry_price=100.0,
        exit_date=pd.Timestamp(exit_).date(),
        exit_price=101.0,
        exit_reason="time",
        shares=1.0,
        entry_cost=100.0,
        exit_value=101.0,
        pnl=1.0,
        return_pct=0.01,
        open_seq=open_seq,
    )


def test_exposure_separates_cross_fold_lot_collisions():
    """Two disjoint 4-session positions from different folds, both lot 1.

    They must stay two short intervals, not one Jan-to-Dec span. The buggy
    ``(ticker, open_seq)`` key collapsed them into roughly the whole year.
    """
    trades = [
        _trade(entry="2024-01-02", exit_="2024-01-05"),
        _trade(entry="2024-12-02", exit_="2024-12-05"),
    ]

    exposure = _exposure(_INDEX, trades, slot_count=1)

    # 4 inclusive sessions each, 8 of ~261 business days.
    assert exposure == pytest.approx(8 / len(_INDEX))
    assert exposure < 0.05
    # The merged-interval bug would have occupied most of the year.
    assert exposure != pytest.approx((len(_INDEX) - 8) / len(_INDEX), abs=0.05)


def test_exposure_still_merges_partial_exit_tranches_of_one_lot():
    """Tranches share ``(ticker, open_seq, entry_date)`` and collapse to one."""
    tranches = [
        _trade(entry="2024-01-02", exit_="2024-01-04"),
        _trade(entry="2024-01-02", exit_="2024-01-10"),
    ]
    whole_position = _trade(entry="2024-01-02", exit_="2024-01-10")

    merged = _exposure(_INDEX, tranches, slot_count=1)

    assert merged == _exposure(_INDEX, [whole_position], slot_count=1)
    # Jan 2..Jan 10 inclusive is 7 business sessions.
    assert merged == pytest.approx(7 / len(_INDEX))


@pytest.mark.parametrize("oos_metric", [-0.2, 0.0])
def test_verdict_fails_non_positive_oos_with_positive_mc_tail(oos_metric):
    verdict = rr._verdict(
        insufficient_data=False,
        overfit_flag=False,
        degradation=rr._degradation_ratio(-0.5, oos_metric),
        mc_return_p05=0.05,
        oos_metric=oos_metric,
    )

    assert verdict == "FAIL: non-positive OOS performance"
    assert verdict != rr.PASS_VERDICT


def test_verdict_keeps_weak_oos_message_when_mc_tail_negative():
    verdict = rr._verdict(
        insufficient_data=False,
        overfit_flag=False,
        degradation=0.0,
        mc_return_p05=-0.01,
        oos_metric=-0.2,
    )

    assert verdict == "FAIL: weak OOS and left-tail MC returns negative"
