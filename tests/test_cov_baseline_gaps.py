"""Coverage for baseline gaps left by PRs #74/#79 — offline, no network."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from screener import history as history_mod
from screener import minervini
from screener.backtester.data import (
    FMPPriceFetcher,
    YFinancePriceFetcher,
    _naive_normalized_index,
    _needs_tail_refresh,
    _normalize_fmp_historical,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import collect_names, parse
from screener.backtester.rolling_candidates import (
    _build_rolling_candidate_matrices,
    _candidate_rows_for_day,
)
from screener.backtester.tearsheet import _trade_ledger_frame, _trade_timeline_html
from screener.cache import cache_area_path, set_cache_area_path
from screener.cli import cli
from screener.commands.screen_report import _describe_numeric, _fmt
from screener.financials import first_number
from screener.garp import US_THRESHOLDS, FmpGarpAdapter, YFinanceGarpAdapter
from screener.reporting import open_report
from screener.resilience import (
    ProviderRateLimiter,
    _is_http_429,
    _retry_after,
    set_provider_rates,
)
from screener.screen_workflow import default_screen_workflow_deps

from tests.conftest import StubPriceFetcher, make_bars

# ───────────────────────── backtester/data.py ─────────────────────────


def test_naive_normalized_index_intraday_tz_aware():
    idx = pd.DatetimeIndex(["2024-03-01 09:30", "2024-03-01 09:35"], tz="US/Eastern")
    out = _naive_normalized_index(idx, "5m")
    assert out.tz is None
    # 09:30 US/Eastern == 14:30 UTC.
    assert out[0] == pd.Timestamp("2024-03-01 14:30")


def test_needs_tail_refresh_invalid_ttl_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENER_PRICE_TAIL_TTL_SECONDS", "not-a-number")
    p = tmp_path / "fresh.parquet"
    p.write_bytes(b"x")
    # Invalid env falls back to the default TTL; a just-written file is fresh.
    assert _needs_tail_refresh(p, pd.Timestamp(date.today())) is False


def test_needs_tail_refresh_missing_file(tmp_path):
    missing = tmp_path / "nope.parquet"
    assert _needs_tail_refresh(missing, pd.Timestamp(date.today())) is False


def test_intraday_chunks_unknown_interval_passthrough(tmp_path):
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, interval="1d")
    s, e = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")
    assert fetcher._intraday_chunks(s, e) == [(s, e)]


def test_intraday_chunks_clamps_and_warns(tmp_path):
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, interval="1m")
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=400)
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    chunks = fetcher._intraday_chunks(start, end)
    assert chunks
    # Start was clamped to the yfinance availability cap.
    assert chunks[0][0] > start


def test_intraday_chunks_fully_before_cap_returns_empty(tmp_path):
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, interval="1m")
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=400)
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=300)
    assert fetcher._intraday_chunks(start, end) == []


def test_yfinance_fetch_no_jobs_after_clamp(tmp_path):
    """A stale intraday range clamps to zero windows — no downloads happen."""
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, interval="1m", refresh=True)
    start = (pd.Timestamp.now() - pd.Timedelta(days=400)).date()
    end = (pd.Timestamp.now() - pd.Timedelta(days=300)).date()
    out = fetcher.fetch(["AAA"], start, end)
    assert out["AAA"].empty


def test_normalize_fmp_historical_daily_tz_aware_dates():
    payload = {
        "historical": [
            {
                "date": "2024-03-01T00:00:00+00:00",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
            }
        ]
    }
    out = _normalize_fmp_historical(payload, auto_adjust=True, interval="1d")
    assert out.index.tz is None
    assert out.index[0] == pd.Timestamp("2024-03-01")


def test_fmp_fetch_empty_tickers():
    fetcher = FMPPriceFetcher(api_key="k")
    assert fetcher.fetch(["", ""], date(2024, 1, 1), date(2024, 1, 5)) == {}


# ───────────────────────── backtester CLI / models / pine ─────────────────────────


def test_backtest_historical_from_run_unknown_run(tmp_path, monkeypatch):
    monkeypatch.setattr(history_mod, "DB_PATH", tmp_path / "history.db")
    history_mod.save_run("us", "ema", 1, pd.DataFrame([{"name": "AAA"}]))
    res = CliRunner().invoke(cli, ["backtest-historical", "--from-run", "999"])
    assert res.exit_code == 2
    assert "999" in res.output


def test_backtest_config_rejects_unknown_interval():
    with pytest.raises(ValidationError, match="unsupported interval"):
        BacktestConfig(
            market="us",
            as_of=date(2024, 3, 1),
            hold=5,
            top=2,
            interval="45m",
            entry_expr="close > 0",
            exit_expr=None,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=10_000.0,
            benchmark="SPY",
            tickers=("AAA",),
        )


def test_collect_names_visits_unary_operand():
    assert "close" in collect_names(parse("not (close > 0)"))


def test_candidate_rows_all_nan_rank_score():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bars = make_bars(n=20)
    bars.index = idx
    bars["rank_score"] = np.nan
    mats = _build_rolling_candidate_matrices(
        {"AAA": bars},
        {"AAA": pd.Series(True, index=idx)},
        {},
        list(idx),
        lookback_required=3,
    )
    assert mats.rank_score_np is not None
    rows, warnings = _candidate_rows_for_day(idx[15], mats, exclude=set())
    assert rows == []
    assert warnings == []


def test_tearsheet_empty_trades_short_circuits():
    empty = pd.DataFrame()
    assert _trade_ledger_frame(empty).empty
    assert "No trades" in _trade_timeline_html(empty)


# ───────────────────────── cache / criteria / financials ─────────────────────────


def test_cache_area_unknown_names():
    with pytest.raises(KeyError, match="unknown cache area"):
        cache_area_path("bogus")
    with pytest.raises(KeyError, match="unknown cache area"):
        set_cache_area_path("bogus", "/tmp/nope")


def test_first_number_case_sensitive():
    mapping = {"Revenue": "5", "eps": None}
    assert first_number(mapping, "Revenue", case_insensitive=False) == 5.0
    assert first_number(mapping, "revenue", case_insensitive=False) is None


# ───────────────────────── garp adapters / history / reporting ─────────────────────────


def test_us_adapter_thresholds_and_fmp_non_dict(monkeypatch):
    assert YFinanceGarpAdapter().thresholds is US_THRESHOLDS
    adapter = FmpGarpAdapter(api_key="k")
    assert adapter.thresholds is US_THRESHOLDS
    monkeypatch.setattr("screener.garp._fetch_fmp_us_cached", lambda *a, **kw: None)
    assert adapter.load_row("AAA", None, cache_ttl=None, refresh=False) is None


def test_list_runs_criteria_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(history_mod, "DB_PATH", tmp_path / "history.db")
    history_mod.save_run("us", "ema", 1, pd.DataFrame([{"name": "AAA"}]))
    out = history_mod.list_runs(market="us", criteria="ema", limit=5)
    assert len(out) == 1
    assert history_mod.list_runs(criteria="garp", limit=5).empty


def test_open_report_uses_browser(tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda uri: opened.append(uri))
    report = tmp_path / "r.html"
    report.write_text("<html></html>")
    open_report(report)
    assert opened and opened[0].startswith("file://")


# ───────────────────────── screen_report helpers ─────────────────────────


def test_fmt_floats():
    assert _fmt(float("nan")) == "-"
    assert _fmt(1234.5) == "1,234.50"


def test_describe_numeric_skips_all_nan_column():
    df = pd.DataFrame({"close": [np.nan, np.nan]})
    assert _describe_numeric(df, ["close"]).empty


# ───────────────────────── resilience ─────────────────────────


def test_rate_limiter_zero_rate_is_noop():
    calls: list[float] = []
    ProviderRateLimiter().wait(
        "p", 0.0, clock=lambda: 0.0, sleep=lambda s: calls.append(s)
    )
    assert calls == []


def test_set_provider_rates_rejects_non_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        set_provider_rates({"yfinance": 0.0})


def test_is_http_429_urllib():
    err = HTTPError("http://x", 429, "too many", {}, None)
    assert _is_http_429(err) is True


def test_retry_after_parsing():
    # urllib HTTPError carries headers on the exception itself.
    err = HTTPError("http://x", 429, "too many", {"Retry-After": "7"}, None)
    assert _retry_after(err) == 7.0
    # No response and not an HTTPError → nothing to parse.
    assert _retry_after(ValueError("x")) is None
    # Response without the header.
    no_header = SimpleNamespace(response=SimpleNamespace(headers={}))
    assert _retry_after(no_header) is None
    # Unparseable value.
    bad = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "soon"}))
    assert _retry_after(bad) is None


# ───────────────────────── screen_workflow ─────────────────────────


def test_default_screen_workflow_deps_builds():
    deps = default_screen_workflow_deps()
    assert callable(deps.scan)
    assert callable(deps.render_report)


# ───────────────────────── minervini ─────────────────────────


def _rising_bars(periods: int = 300, end: str = "2024-06-28") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=periods)
    close = np.linspace(50.0, 100.0, periods)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1_000_000.0),
        },
        index=idx,
    )


def _flat_bars(periods: int = 300, end: str = "2024-06-28") -> pd.DataFrame:
    frame = _rising_bars(periods, end)
    for col in ("open", "high", "low", "close"):
        frame[col] = 50.0
    return frame


def test_add_rs_rank_column_degenerate_inputs():
    # All-empty universe: nothing to rank, input returned untouched.
    empty = {"E": pd.DataFrame()}
    assert minervini.add_rs_rank_column(empty) is empty
    # A symbol without a close column gets NA rank alongside ranked peers.
    out = minervini.add_rs_rank_column(
        {
            "GOOD": _rising_bars(),
            "NOCLOSE": pd.DataFrame(
                {"volume": [1.0, 2.0]},
                index=pd.bdate_range("2024-01-01", periods=2),
            ),
        }
    )
    assert out["NOCLOSE"]["rs_rank"].isna().all()


def test_evaluate_symbol_rejections():
    as_of = date(2024, 6, 28)
    assert minervini.evaluate_symbol("X", pd.DataFrame(), as_of) is None
    assert minervini.evaluate_symbol("X", _rising_bars(periods=10), as_of) is None
    # Enough history but no rs_rank column → NaN value → rejected.
    assert minervini.evaluate_symbol("X", _rising_bars(), as_of) is None


def test_scan_minervini_offline(monkeypatch):
    monkeypatch.setattr(minervini, "load_universe", lambda market, **kw: ["AAA", "BBB"])
    fetcher = StubPriceFetcher({"AAA": _rising_bars(), "BBB": _flat_bars()})
    rows = minervini.scan_minervini(
        "us",
        as_of=date(2024, 6, 28),
        limit=5,
        cache_ttl="900",
        refresh=False,
        fetcher=fetcher,
    )
    assert [row.symbol for row in rows] == ["AAA"]
    assert rows[0].rs_rank == 100.0
