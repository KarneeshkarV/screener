"""Unit tests for earnings blackout gates and screen days_to_earnings enrichment.

All network access is stubbed — no live yfinance / NSE calls.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.rolling_candidates import _build_rolling_candidate_matrices
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.earnings_backtest.earnings_dates import (
    events_to_dates_map,
    fetch_next_earnings_dates,
    load_earnings_dates_map,
    next_earnings_date,
)
from tests.conftest import make_bars


def test_build_rolling_candidate_matrices_earnings_blackout():
    # Calendar-day range so blackout windows are exact day offsets (not bdays).
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    bars = make_bars(n=20, open_base=100.0)
    bars.index = idx
    bars_by = {"AAA": bars, "BBB": bars.copy()}
    entry_sig = {
        "AAA": pd.Series(True, index=idx),
        "BBB": pd.Series(True, index=idx),
    }
    master = list(idx)
    # Earnings on day 10 → with N=2, blackout covers days 8, 9, 10 (inclusive).
    earn_day = idx[10].date()
    warnings: list[str] = []
    mats = _build_rolling_candidate_matrices(
        bars_by,
        entry_sig,
        {},
        master,
        lookback_required=3,
        earnings_blackout={"AAA": [earn_day]},  # BBB has no data
        earnings_blackout_days=2,
        warnings=warnings,
    )
    # Outside window: untouched.
    assert bool(mats.signal_mat.iloc[5]["AAA"])
    assert bool(mats.signal_mat.iloc[12]["AAA"])
    # Inside window (E-2, E-1, E): suppressed.
    assert not bool(mats.signal_mat.iloc[8]["AAA"])
    assert not bool(mats.signal_mat.iloc[9]["AAA"])
    assert not bool(mats.signal_mat.iloc[10]["AAA"])
    # Ticker without earnings data is never gated.
    assert bool(mats.signal_mat.iloc[8]["BBB"])
    assert bool(mats.signal_mat.iloc[10]["BBB"])
    assert any("lack earnings dates" in w and "BBB" in w for w in warnings)


def test_build_rolling_candidate_matrices_earnings_blackout_includes_day_zero():
    """N=0 still blackouts the earnings date itself."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    bars = make_bars(n=10, open_base=100.0)
    bars.index = idx
    mats = _build_rolling_candidate_matrices(
        {"AAA": bars},
        {"AAA": pd.Series(True, index=idx)},
        {},
        list(idx),
        lookback_required=1,
        earnings_blackout={"AAA": [idx[5].date()]},
        earnings_blackout_days=0,
    )
    assert bool(mats.signal_mat.iloc[4]["AAA"])
    assert not bool(mats.signal_mat.iloc[5]["AAA"])
    assert bool(mats.signal_mat.iloc[6]["AAA"])


def test_events_to_dates_map_and_load_stub():
    events = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "earnings_date": ["2024-01-15", "2024-04-20", "2024-02-01"],
        }
    )
    mapping = events_to_dates_map(events)
    assert mapping["AAPL"] == [date(2024, 1, 15), date(2024, 4, 20)]
    assert mapping["MSFT"] == [date(2024, 2, 1)]

    loaded = load_earnings_dates_map(
        ["AAPL", "MSFT"],
        "us",
        collect_fn=lambda tickers, years=5, market="us": events,
    )
    assert loaded == mapping

    empty = load_earnings_dates_map([], "us", collect_fn=lambda *a, **k: events)
    assert empty == {}


def test_next_earnings_date():
    as_of = date(2024, 3, 1)
    assert next_earnings_date([date(2024, 1, 1), date(2024, 4, 15)], as_of) == date(
        2024, 4, 15
    )
    assert next_earnings_date([date(2024, 1, 1)], as_of) is None
    assert next_earnings_date(None, as_of) is None
    # On the day itself counts as upcoming.
    assert next_earnings_date([date(2024, 3, 1)], as_of) == date(2024, 3, 1)


def test_fetch_next_earnings_dates_us_stub():
    ed = pd.DataFrame(
        {"EPS Estimate": [1.0]},
        index=pd.DatetimeIndex(["2024-06-01"]),
    )

    out = fetch_next_earnings_dates(
        ["AAPL", "ZZZ"],
        "us",
        as_of=date(2024, 5, 1),
        yf_fetcher=lambda t: ed if t == "AAPL" else None,
    )
    assert out["AAPL"] == date(2024, 6, 1)
    assert out["ZZZ"] is None


def test_fetch_next_earnings_dates_india_stub():
    nse = pd.DataFrame(
        {
            "ticker": ["RELIANCE.NS", "TCS.NS"],
            "earnings_date": [pd.Timestamp("2024-07-10"), pd.Timestamp("2024-01-01")],
        }
    )
    out = fetch_next_earnings_dates(
        ["RELIANCE", "NSE:TCS", "INFY"],
        "india",
        as_of=date(2024, 6, 1),
        nse_fetcher=lambda: nse,
    )
    assert out["RELIANCE"] == date(2024, 7, 10)
    # Past-only earnings → unknown.
    assert out["NSE:TCS"] is None
    assert out["INFY"] is None


def test_fetch_next_earnings_dates_provider_failure():
    def boom():
        raise RuntimeError("offline")

    out = fetch_next_earnings_dates(
        ["AAA"], "india", as_of=date(2024, 1, 1), nse_fetcher=boom
    )
    assert out == {"AAA": None}


def test_enrich_days_to_earnings_and_buffer_filter():
    df = pd.DataFrame(
        {
            "name": ["AAA", "BBB", "CCC"],
            "close": [10.0, 20.0, 30.0],
        }
    )
    as_of = date(2024, 5, 1)

    def provider(symbols, market, as_of=None):
        return {
            "AAA": date(2024, 5, 6),  # 5 days
            "BBB": date(2024, 6, 1),  # 31 days
            "CCC": None,
        }

    enriched = enrich_days_to_earnings(df, "us", as_of=as_of, provider=provider)
    assert enriched["days_to_earnings"].tolist() == [5, 31, None]

    filtered = filter_earnings_buffer(enriched, 5)
    # AAA has dte=5 <= 5 → dropped; BBB and CCC kept.
    assert filtered["name"].tolist() == ["BBB", "CCC"]
    assert filtered["days_to_earnings"].tolist() == [31, None]


def test_enrich_days_to_earnings_provider_failure_leaves_column():
    df = pd.DataFrame({"name": ["AAA"]})

    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    out = enrich_days_to_earnings(df, "us", provider=boom)
    assert "days_to_earnings" in out.columns
    assert out["days_to_earnings"].tolist() == [None]


def test_filter_earnings_buffer_rejects_negative():
    with pytest.raises(ValueError):
        filter_earnings_buffer(pd.DataFrame({"days_to_earnings": [1]}), -1)


def test_screen_workflow_earnings_buffer(tmp_path, monkeypatch):
    from screener.criteria import FilterCriteriaSelection
    from screener.screen_workflow import (
        ScreenMode,
        ScreenRequest,
        ScreenWorkflowDeps,
        run_screen_workflow,
    )

    frame = pd.DataFrame(
        {"name": ["AAA", "BBB", "CCC"], "description": ["a", "b", "c"]}
    )

    def fake_enrich(df, market, **kwargs):
        out = df.copy()
        out["days_to_earnings"] = [3, 20, None]
        return out

    monkeypatch.setattr("screener.enrich.enrich_days_to_earnings", fake_enrich)

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        parse_cache_ttl=lambda raw: 900.0,
        scan=lambda **kwargs: (3, frame),
        save_run=lambda *args: 1,
        previous_run=lambda *args: None,
        diff=lambda current, previous: ([], []),
        temp_report_path=lambda prefix: tmp_path / f"{prefix}.html",
        render_report=lambda *args, **kwargs: tmp_path / "unused.html",
    )
    request = ScreenRequest(
        market="us",
        criteria_names=("ema",),
        limit=5,
        order_by="volume",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
        earnings_buffer=5,
    )
    outcome = run_screen_workflow(request, deps)
    assert outcome.mode is ScreenMode.CSV
    assert outcome.df is not None
    assert outcome.df["name"].tolist() == ["BBB", "CCC"]
