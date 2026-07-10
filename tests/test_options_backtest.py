from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from screener import cache
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import parse
from screener.backtester.rolling import run_rolling_backtest
from screener.options import backtest as options_backtest
from screener.options.backtest import (
    merge_options_into_bars,
    merge_referenced_options,
    referenced_options_fields,
)
from tests.conftest import StubPriceFetcher, make_bars


@pytest.fixture
def panel_root(tmp_path: Path):
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "panels"
    finally:
        cache.reset_cache_area_paths()


def _cfg(**overrides) -> BacktestConfig:
    defaults = {
        "market": "us",
        "as_of": date(2024, 3, 1),
        "hold": 5,
        "top": 1,
        "entry_expr": "close > 0 and pcr < 0.8",
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
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_referenced_options_fields_only_selects_known_panel_identifiers():
    entry = parse("close > sma(close, 20) and pcr < 0.8 and iv_rank < 30")
    exit_ast = parse("oi_chg_ratio > 2 or close < 0")
    assert referenced_options_fields(entry, exit_ast) == {
        "pcr",
        "iv_rank",
        "oi_chg_ratio",
    }
    assert referenced_options_fields(parse("close > 0")) == set()


def test_point_in_time_join_never_backfills_future_rows():
    index = pd.bdate_range("2026-07-01", periods=6)
    bars = make_bars(start="2026-07-01", n=6)
    bars["pcr"] = 0.9  # injected fallback before canonical coverage begins
    panel = pd.DataFrame(
        [
            {"as_of": index[2], "SYMBOL": "AAPL", "pcr": 0.7, "iv_rank": 20},
            {"as_of": index[4], "SYMBOL": "AAPL", "pcr": 1.2, "iv_rank": 90},
        ]
    )
    result = merge_options_into_bars(
        {"NASDAQ:AAPL": bars},
        market="us",
        fields={"pcr", "iv_rank"},
        panel=panel,
    )
    joined = result.bars_by_tv["NASDAQ:AAPL"]
    assert joined.loc[index[0], "pcr"] == 0.9
    assert pd.isna(joined.loc[index[1], "iv_rank"])
    assert joined.loc[index[2], "pcr"] == 0.7
    assert joined.loc[index[3], "pcr"] == 0.7
    assert joined.loc[index[4], "pcr"] == 1.2
    assert result.covered_symbols == ("NASDAQ:AAPL",)

    # Adding a future observation cannot alter any prior joined value.
    future = pd.concat(
        [
            panel,
            pd.DataFrame(
                [
                    {
                        "as_of": index[-1] + pd.Timedelta(days=7),
                        "SYMBOL": "AAPL",
                        "pcr": 0.1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rerun = merge_options_into_bars(
        {"NASDAQ:AAPL": bars}, market="us", fields={"pcr"}, panel=future
    ).bars_by_tv["NASDAQ:AAPL"]
    pd.testing.assert_series_equal(joined["pcr"], rerun["pcr"])


def test_join_handles_india_symbols_timezone_and_missing_panel():
    index = pd.date_range("2026-07-01 09:15", periods=3, freq="h", tz="Asia/Kolkata")
    bars = make_bars(start="2026-07-01", n=3)
    bars.index = index
    panel = pd.DataFrame([{"as_of": "2026-07-01", "SYMBOL": "RELIANCE", "pcr": 1.1}])
    joined = merge_options_into_bars(
        {"NSE:RELIANCE": bars}, market="india", fields={"pcr"}, panel=panel
    )
    assert joined.bars_by_tv["NSE:RELIANCE"]["pcr"].tolist() == [1.1, 1.1, 1.1]

    missing = merge_options_into_bars(
        {"NSE:RELIANCE": bars},
        market="india",
        fields={"iv_rank"},
        panel=pd.DataFrame(),
    )
    assert missing.covered_symbols == ()
    assert missing.bars_by_tv["NSE:RELIANCE"]["iv_rank"].isna().all()
    untouched = merge_options_into_bars(
        {"NSE:RELIANCE": bars}, market="india", fields=set(), panel=panel
    )
    assert untouched.fields == ()


def test_merge_referenced_options_emits_honest_coverage_warning(monkeypatch):
    bars = {"AAPL": make_bars(n=5)}
    warnings: list[str] = []
    monkeypatch.setattr(
        options_backtest, "read_options_panel", lambda _market: pd.DataFrame()
    )
    merged = merge_referenced_options(
        bars,
        market="us",
        entry_ast=parse("pcr < 1"),
        exit_ast=None,
        warnings=warnings,
    )
    assert merged["AAPL"]["pcr"].isna().all()
    assert "no point-in-time coverage" in warnings[0]

    warnings.clear()
    same = merge_referenced_options(
        bars,
        market="us",
        entry_ast=parse("close > 0"),
        exit_ast=None,
        warnings=warnings,
    )
    assert same is bars and warnings == []


def test_historical_backtest_evaluates_options_panel_fields(panel_root: Path):
    bars = make_bars(start="2024-01-01", n=80, open_base=100)
    benchmark = make_bars(start="2024-01-01", n=80, open_base=400)
    signal_day = bars.index[50]
    cache.append_panel_snapshot(
        "options_metrics_us",
        pd.DataFrame(
            [{"as_of": signal_day - pd.Timedelta(days=1), "SYMBOL": "AAA", "pcr": 0.7}]
        ),
        dedupe_keys=["as_of", "SYMBOL"],
    )
    result = run_backtest(
        _cfg(as_of=signal_day.date()),
        StubPriceFetcher({"AAA": bars, "SPY": benchmark}),
    )
    assert result.selection["ticker"].tolist() == ["AAA"]
    assert any(
        "options panel joined point-in-time" in warning for warning in result.warnings
    )


def test_rolling_backtest_starts_options_signal_only_after_panel_date(
    panel_root: Path,
):
    bars = make_bars(start="2024-01-01", n=70, open_base=100)
    benchmark = make_bars(start="2024-01-01", n=70, open_base=400)
    effective = bars.index[25]
    cache.append_panel_snapshot(
        "options_metrics_us",
        pd.DataFrame([{"as_of": effective, "SYMBOL": "AAA", "pcr": 0.5}]),
        dedupe_keys=["as_of", "SYMBOL"],
    )
    result = run_rolling_backtest(
        _cfg(as_of=bars.index[10].date(), hold=3),
        StubPriceFetcher({"AAA": bars, "SPY": benchmark}),
        start_date=bars.index[10].date(),
        end_date=bars.index[50].date(),
    )
    assert not result.selection.empty
    assert result.selection["signal_date"].min() >= effective.date()
