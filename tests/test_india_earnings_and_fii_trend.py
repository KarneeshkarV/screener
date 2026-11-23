"""Regression tests for India upcoming-earnings availability and FII trend sign.

Two independent bugs are pinned here:

* Indian ``next_earnings_date`` used to read only published NSE filings, so it
  could never return an upcoming date. ``--earnings-buffer`` then passed every
  Indian row without a word. The resolver now asks Yahoo (which carries future
  board-meeting dates for ``.NS`` symbols) before falling back to NSE filings,
  and warns when no upcoming date exists for any row.
* ``fii_trend`` divided today's flow by the *signed* 20-day mean, so heavy
  selling during a selling regime looked bullish, a buying reversal looked
  bearish, and a near-zero signed mean exploded. It now normalizes by the
  average absolute flow, preserving today's direction.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from screener.criteria import FilterCriteriaSelection
from screener.earnings_backtest.earnings_dates import fetch_next_earnings_dates
from screener.screen_workflow import ScreenRequest, run_screen_workflow
from screener.unusual_volume import fii_dii


def _yf_frame(day: str) -> pd.DataFrame:
    return pd.DataFrame({"EPS Estimate": [1.0]}, index=pd.DatetimeIndex([day]))


def test_india_upcoming_earnings_resolve_from_yahoo_calendar():
    """Yahoo's scheduled meeting date is a genuine upcoming date for India."""
    out = fetch_next_earnings_dates(
        ["RELIANCE", "NSE:TCS", "INFY"],
        "india",
        as_of=date(2024, 6, 1),
        yf_fetcher=lambda ticker: (
            _yf_frame("2024-07-10") if ticker == "RELIANCE.NS" else None
        ),
        nse_fetcher=lambda: None,
    )

    assert out["RELIANCE"] == date(2024, 7, 10)
    assert out["NSE:TCS"] is None
    assert out["INFY"] is None


def test_india_falls_back_to_nse_filings_when_yahoo_is_empty():
    nse = pd.DataFrame(
        {
            "ticker": ["TCS.NS"],
            "earnings_date": [pd.Timestamp("2024-07-12")],
        }
    )

    out = fetch_next_earnings_dates(
        ["NSE:TCS"],
        "india",
        as_of=date(2024, 6, 1),
        yf_fetcher=lambda _ticker: None,
        nse_fetcher=lambda: nse,
    )

    assert out == {"NSE:TCS": date(2024, 7, 12)}


def test_india_no_upcoming_earnings_logs_a_clear_warning(caplog):
    with caplog.at_level(logging.WARNING):
        out = fetch_next_earnings_dates(
            ["INFY"],
            "india",
            as_of=date(2024, 6, 1),
            yf_fetcher=lambda _ticker: None,
            nse_fetcher=lambda: None,
        )

    assert out == {"INFY": None}
    assert "india_upcoming_earnings_unavailable" in caplog.text


def test_india_earnings_buffer_warns_instead_of_silently_keeping_everything(
    monkeypatch, caplog
):
    import screener.screen_workflow as sw

    frame = pd.DataFrame({"name": ["AAA", "BBB"]})
    as_of = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    monkeypatch.setattr(
        sw,
        "resolve_criteria",
        lambda names: FilterCriteriaSelection(tuple(names), "value", ["FILTER"]),
    )
    monkeypatch.setattr(sw, "scan", lambda **kwargs: (2, frame, as_of))
    monkeypatch.setattr(
        sw,
        "enrich_days_to_earnings",
        lambda df, market: df.assign(days_to_earnings=[None, None]),
    )
    request = ScreenRequest(
        market="india",
        criteria_names=("value",),
        limit=5,
        order_by="volume",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
        earnings_buffer=5,
    )

    with caplog.at_level(logging.WARNING):
        outcome = run_screen_workflow(request)

    assert outcome.df is not None
    # Unknown dates are kept, but the user is told the buffer could not run.
    assert outcome.df["name"].tolist() == ["AAA", "BBB"]
    assert "no upcoming earnings dates" in caplog.text


def _fii_panel(flows: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=len(flows)),
            "fii_net": flows,
            "dii_net": [0.0] * len(flows),
        }
    )


def test_fii_trend_is_negative_when_selling_accelerates_in_a_selling_regime():
    panel = _fii_panel([-1000.0] * 20 + [-3000.0])
    metrics = fii_dii.compute_fii_dii_metrics(panel, panel["date"].max().date())

    # -3000 against a typical 1000 outflow magnitude is heavier selling.
    assert metrics["fii_trend"] == pytest.approx(-3.0)


def test_fii_trend_is_positive_for_a_buying_reversal_in_a_selling_regime():
    panel = _fii_panel([-1000.0] * 20 + [2000.0])
    metrics = fii_dii.compute_fii_dii_metrics(panel, panel["date"].max().date())

    assert metrics["fii_trend"] == pytest.approx(2.0)


def test_fii_trend_survives_a_zero_signed_mean_without_exploding():
    # Signed mean is exactly zero, but actual flow magnitude is 1000/day.
    flows = [1000.0 if i % 2 else -1000.0 for i in range(20)] + [1000.0]
    panel = _fii_panel(flows)
    series = fii_dii.fii_dii_metric_series(panel)
    trend = series["fii_trend"].iloc[-1]

    assert trend == pytest.approx(1.0)
    assert not pd.isna(trend)


def test_fii_trend_is_none_not_inf_when_denominator_is_zero():
    panel = _fii_panel([0.0] * 21)
    series = fii_dii.fii_dii_metric_series(panel)
    metrics = fii_dii.compute_fii_dii_metrics(panel, panel["date"].max().date())

    trend = series["fii_trend"].iloc[-1]
    assert pd.isna(trend)
    assert trend != float("inf")
    assert metrics["fii_trend"] is None
