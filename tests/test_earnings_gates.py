"""Unit tests for earnings blackout gates and screen days_to_earnings enrichment.

All network access is stubbed — no live yfinance / NSE calls.
"""

from __future__ import annotations

from datetime import date, timedelta
from threading import Barrier, get_ident

import pandas as pd
import pytest
from click.testing import CliRunner

from main import cli
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.rolling_candidates import _build_rolling_candidate_matrices
from screener.backtester.rolling_candidates import _candidate_rows_for_day
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.earnings_backtest.earnings_dates import (
    events_to_dates_map,
    fetch_next_earnings_dates,
    load_earnings_dates_map,
    next_earnings_date,
)
from tests.conftest import StubPriceFetcher, make_bars


def _rolling_cfg(**overrides) -> BacktestConfig:
    values = {
        "market": "us",
        "as_of": date(2024, 3, 1),
        "hold": 3,
        "top": 1,
        "entry_expr": "entry_signal > 0",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA",),
        "min_price": None,
        "min_avg_dollar_volume": None,
    }
    values.update(overrides)
    return BacktestConfig(**values)


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


def test_candidate_rows_empty_when_factor_scores_are_unknown():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    bars = make_bars(n=4)
    bars.index = idx
    bars["rank_score"] = float("nan")
    mats = _build_rolling_candidate_matrices(
        {"AAA": bars},
        {"AAA": pd.Series(True, index=idx)},
        {},
        list(idx),
        lookback_required=1,
    )

    rows, warnings = _candidate_rows_for_day(idx[2], mats, exclude=set())

    assert rows == []
    assert warnings == []


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


def test_earnings_date_mapping_handles_invalid_rows_and_provider_failure():
    assert events_to_dates_map(pd.DataFrame()) == {}
    assert events_to_dates_map(pd.DataFrame({"ticker": ["AAA"]})) == {}
    assert (
        events_to_dates_map(
            pd.DataFrame({"ticker": ["AAA"], "earnings_date": [pd.NaT]})
        )
        == {}
    )

    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    assert load_earnings_dates_map(["AAA"], "us", collect_fn=boom) == {}


def test_next_earnings_date():
    as_of = date(2024, 3, 1)
    assert next_earnings_date([date(2024, 1, 1), date(2024, 4, 15)], as_of) == date(
        2024, 4, 15
    )
    assert next_earnings_date([date(2024, 1, 1)], as_of) is None
    assert next_earnings_date(None, as_of) is None
    # On the day itself counts as upcoming.
    assert next_earnings_date([date(2024, 3, 1)], as_of) == date(2024, 3, 1)
    assert next_earnings_date([pd.NaT, date(2024, 4, 1)], as_of) == date(2024, 4, 1)


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


def test_fetch_next_earnings_dates_us_fetches_tickers_concurrently():
    barrier = Barrier(2, timeout=2)
    thread_ids: set[int] = set()

    def yf_fetcher(ticker):
        thread_ids.add(get_ident())
        barrier.wait()
        return pd.DataFrame(
            {"EPS Estimate": [1.0]}, index=pd.to_datetime(["2024-05-10"])
        )

    out = fetch_next_earnings_dates(
        ["AAA", "BBB"],
        "us",
        as_of=date(2024, 5, 1),
        yf_fetcher=yf_fetcher,
    )

    assert out == {
        "AAA": date(2024, 5, 10),
        "BBB": date(2024, 5, 10),
    }
    assert len(thread_ids) == 2


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


def test_fetch_next_earnings_dates_empty_and_provider_edge_cases():
    assert fetch_next_earnings_dates([], "us") == {}
    assert fetch_next_earnings_dates(
        ["AAA"], "india", nse_fetcher=lambda: pd.DataFrame()
    ) == {"AAA": None}

    def yf_boom(_ticker):
        raise RuntimeError("offline")

    assert fetch_next_earnings_dates(["AAA"], "us", yf_fetcher=yf_boom) == {"AAA": None}
    assert fetch_next_earnings_dates(
        ["AAA"],
        "us",
        as_of=date(2024, 1, 1),
        yf_fetcher=lambda _ticker: [date(2024, 2, 1)],
    ) == {"AAA": date(2024, 2, 1)}


def test_rolling_backtest_loads_earnings_blackout_map(monkeypatch):
    bars = make_bars(n=20, seed=1)
    bars["entry_signal"] = 0.0
    bars.iloc[8, bars.columns.get_loc("entry_signal")] = 1.0
    benchmark = make_bars(n=20, seed=2, open_base=400.0)
    earnings_day = bars.index[8].date()
    calls: list[tuple[list[str], str, int]] = []

    def load_map(tickers, market, *, years):
        calls.append((tickers, market, years))
        return {"AAA": [earnings_day]}

    monkeypatch.setattr(
        "screener.earnings_backtest.earnings_dates.load_earnings_dates_map", load_map
    )
    result = run_rolling_backtest(
        _rolling_cfg(earnings_blackout_days=0),
        StubPriceFetcher({"AAA": bars, "SPY": benchmark}),
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
    )

    assert calls == [(["AAA"], "us", 3)]
    assert result.trades == []


def test_enrich_days_to_earnings_and_buffer_filter():
    df = pd.DataFrame(
        {
            "name": ["AAA", "BBB", "CCC"],
            "close": [10.0, 20.0, 30.0],
        }
    )
    as_of = date(2024, 5, 1)

    def provider(symbols, market, *, as_of):
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


def test_enrich_days_to_earnings_symbol_fallbacks_and_provider_protocol():
    assert enrich_days_to_earnings(pd.DataFrame(), "us").empty
    assert enrich_days_to_earnings(pd.DataFrame({"close": [1.0]}), "us")[
        "days_to_earnings"
    ].tolist() == [None]

    def provider(symbols, market, *, as_of):
        assert symbols == ["AAA"]
        assert market == "us"
        assert as_of == date(2024, 5, 1)
        return {"AAA": date(2024, 5, 1)}

    out = enrich_days_to_earnings(
        pd.DataFrame({"ticker": ["AAA"]}),
        "us",
        as_of=date(2024, 5, 1),
        provider=provider,
    )
    assert out["days_to_earnings"].tolist() == [0]


def test_filter_earnings_buffer_rejects_negative():
    with pytest.raises(ValueError):
        filter_earnings_buffer(pd.DataFrame({"days_to_earnings": [1]}), -1)


def test_filter_earnings_buffer_leaves_unenriched_frames_unchanged():
    empty = pd.DataFrame()
    plain = pd.DataFrame({"name": ["AAA"]})
    assert filter_earnings_buffer(empty, 1) is empty
    assert filter_earnings_buffer(plain, 1) is plain


def test_earnings_blackout_validation_errors():
    result = CliRunner().invoke(cli, ["backtest-rolling", "--earnings-blackout", "-1"])
    assert result.exit_code == 2
    assert "--earnings-blackout must be >= 0" in result.output

    with pytest.raises(ValueError, match="unsupported interval"):
        _rolling_cfg(interval="2h")


def test_screen_workflow_earnings_buffer(tmp_path):
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

    def fake_enrich(df, market):
        out = df.copy()
        out["days_to_earnings"] = [3, 20, None]
        return out

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
        enrich_days_to_earnings=fake_enrich,
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


def test_enrich_days_to_earnings_default_provider(monkeypatch):
    """provider=None imports the real earnings-dates fetcher lazily."""
    import screener.earnings_backtest.earnings_dates as earnings_dates

    as_of = date(2024, 3, 1)
    seen: list[tuple] = []

    def fake_fetch(symbols, market, *, as_of):
        seen.append((tuple(symbols), market, as_of))
        return {sym: as_of + timedelta(days=3) for sym in symbols}

    monkeypatch.setattr(earnings_dates, "fetch_next_earnings_dates", fake_fetch)
    df = pd.DataFrame([{"name": "AAA"}])
    enriched = enrich_days_to_earnings(df, "us", as_of=as_of)
    assert seen and seen[0][1] == "us"
    assert enriched["days_to_earnings"].tolist() == [3]
