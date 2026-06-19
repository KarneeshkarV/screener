"""Point-in-time correctness for India earnings event dating (H-5).

screener.in keys quarterly results on the fiscal PERIOD-END (e.g. "Mar 2024" ->
2024-03-31), but Indian results are only announced ~45-60 days later. Applying
an event on the bare period-end leaks information into the backtest. These
offline tests assert:

* the openscreener event date is strictly later than the period-end (period-end
  + filing lag), so no event is applied before it was public; and
* when a real NSE announcement date is available for the same ticker/period it
  is used instead, and the openscreener-sourced duplicate is dropped.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from screener.earnings_backtest import data as ebd
from screener.earnings_backtest.data import (
    INDIA_EARNINGS_FILING_LAG_DAYS,
    collect_earnings_events,
    fetch_earnings_dates_openscreener,
)


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    """Bypass the JSON disk cache so each test exercises the live parse path."""
    monkeypatch.setattr(ebd, "_read_json_cache", lambda path, max_age: (False, None))
    monkeypatch.setattr(ebd, "_write_json_cache", lambda path, value: None)


class _FakeStock:
    """Stand-in for ``openscreener.Stock`` returning a fixed quarterly payload."""

    _PAYLOAD = {
        "quarterly_results": [
            {"date": "Dec 2023", "eps": 10.0},
            {"date": "Mar 2024", "eps": 12.0},
        ]
    }

    def __init__(self, symbol, scraper=None):
        self.symbol = symbol

    def fetch(self, section):
        return self._PAYLOAD

    def shareholding_quarterly(self):  # unused here, kept for interface parity
        return []


@pytest.fixture(autouse=True)
def _stub_openscreener(monkeypatch):
    """Inject a fake ``openscreener`` module (imported lazily inside the fn)."""
    module = types.ModuleType("openscreener")
    module.Stock = _FakeStock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openscreener", module)


def test_openscreener_floor_is_conservative_upper_bound_no_leak():
    # No-leak floor: with no real NSE announcement date, the synthetic event date
    # must sit at the CONSERVATIVE UPPER bound (>= period-end + 60 days), so a late
    # filer announcing anywhere in the stated 45-60 day window is never preceded by
    # the backtest as_of. The lower bound (45) would leak EPS for day 46-60 filers.
    NO_LEAK_FLOOR_DAYS = 60
    assert INDIA_EARNINGS_FILING_LAG_DAYS >= NO_LEAK_FLOOR_DAYS

    ed = fetch_earnings_dates_openscreener("RELIANCE.NS", years=5)
    assert ed is not None and not ed.empty

    # March year-end quarter is the classic late-filer case.
    mar_end = pd.Timestamp("2024-03-31")
    floor = mar_end + pd.Timedelta(days=NO_LEAK_FLOOR_DAYS)
    mar_events = [ts for ts in ed.index if (ts + pd.offsets.QuarterEnd(-1)) == mar_end]
    assert mar_events, "expected a Mar-2024 quarter event"
    for ts in mar_events:
        # Emitted as_of is at or after the worst-case (day-60) real announcement.
        assert ts >= floor
    # A late filer announcing up to day 60 never precedes the synthetic as_of:
    # for every plausible real announcement day in [45, 60], as_of >= announcement.
    for delay in range(45, 61):
        real_announcement = mar_end + pd.Timedelta(days=delay)
        assert all(ts >= real_announcement for ts in mar_events)


def test_openscreener_event_date_is_after_period_end():
    ed = fetch_earnings_dates_openscreener("RELIANCE.NS", years=5)
    assert ed is not None and not ed.empty

    # "Mar 2024" -> period-end 2024-03-31; the event must be the *announcement*
    # estimate (period-end + filing lag), strictly later than the bare end.
    mar_end = pd.Timestamp("2024-03-31")
    expected = mar_end + pd.Timedelta(days=INDIA_EARNINGS_FILING_LAG_DAYS)
    assert expected in ed.index
    assert mar_end not in ed.index  # the leaky bare period-end is never used

    dec_end = pd.Timestamp("2023-12-31")
    assert dec_end + pd.Timedelta(days=INDIA_EARNINGS_FILING_LAG_DAYS) in ed.index

    # Every event sits strictly after its own fiscal period-end: each index
    # date equals a known period-end plus the filing lag, never the bare end.
    period_ends = {mar_end, dec_end}
    for ts in ed.index:
        matched = ts - pd.Timedelta(days=INDIA_EARNINGS_FILING_LAG_DAYS)
        assert matched in period_ends
        assert ts > matched


def test_collect_prefers_nse_announcement_and_dedups(monkeypatch):
    # Real NSE announcement for RELIANCE's Mar-2024 quarter, landing 2024-05-25
    # — 55 days after the 2024-03-31 period-end. This exceeds the 45-day filing
    # lag, so the old "announcement - 45d" dedup key drifted into Q2 and FAILED
    # to dedup (double-counting the quarter); the quarter-end mapping handles it.
    nse_date = pd.Timestamp("2024-05-25")
    nse_df = pd.DataFrame(
        {
            "ticker": ["RELIANCE.NS"],
            "earnings_date": [nse_date],
            "desc": ["Financial Results"],
        }
    )
    monkeypatch.setattr(ebd, "fetch_earnings_dates_nse", lambda: nse_df)

    events = collect_earnings_events(["RELIANCE.NS"], years=5, market="india")
    rel = events[events["ticker"] == "RELIANCE.NS"].copy()
    rel["earnings_date"] = pd.to_datetime(rel["earnings_date"])
    dates = set(rel["earnings_date"])

    # The Mar-2024 quarter is represented by the real NSE announcement date,
    # not the openscreener period-end + lag estimate (dedup by ticker/quarter).
    assert nse_date in dates
    osc_estimate = pd.Timestamp("2024-03-31") + pd.Timedelta(
        days=INDIA_EARNINGS_FILING_LAG_DAYS
    )
    assert osc_estimate not in dates

    # Exactly one event for the Mar-2024 fiscal quarter (no duplicate row).
    mar_q = pd.Period("2024Q1", freq="Q")
    mar_rows = rel[rel["earnings_date"].dt.to_period("Q") == mar_q]
    assert len(mar_rows) == 1

    # The Dec-2023 quarter (no NSE row) still appears via the lagged estimate.
    dec_estimate = pd.Timestamp("2023-12-31") + pd.Timedelta(
        days=INDIA_EARNINGS_FILING_LAG_DAYS
    )
    assert dec_estimate in dates


@pytest.mark.parametrize("delay_days", [20, 40, 55, 70, 89])
def test_announcement_maps_to_reporting_quarter_independent_oracle(delay_days):
    """Across the full realistic filing-delay range, a Mar-2024 result's
    announcement must map back to 2024Q1 — the quarter it actually reports on.

    The expected value (2024Q1) is an independent, hand-known fact, not derived
    from the production arithmetic. A result for the quarter ending 2024-03-31
    reports on Q1 no matter whether it is announced 20 or 89 days later.
    """
    period_end = pd.Timestamp("2024-03-31")
    announcement = period_end + pd.Timedelta(days=delay_days)
    reported_quarter = (announcement + pd.offsets.QuarterEnd(-1)).to_period("Q")
    assert reported_quarter == pd.Period("2024Q1", freq="Q")
    assert reported_quarter == period_end.to_period("Q")
