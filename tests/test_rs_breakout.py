from __future__ import annotations

from datetime import date

import pandas as pd
from click.testing import CliRunner

import main
from main import cli
from screener.rs_breakout import (
    delivery_lookup,
    evaluate_symbol,
    previous_completed_week_high,
    relative_strength_55,
    scan_rs_breakouts,
    supertrend,
)
from tests.conftest import StubPriceFetcher


def _trend_bars(
    start: float = 100.0,
    end: float = 150.0,
    volume: float = 100_000.0,
    n: int = 90,
) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-04-30", periods=n)
    close = pd.Series(
        [start + (end - start) * i / (n - 1) for i in range(n)],
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(start)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _delivery_panel(symbol: str, latest: float, previous: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SYMBOL": symbol,
                "date": date(2026, 4, 29),
                "TTL_TRD_QNTY": 100_000.0,
                "DELIV_QTY": previous * 1_000,
                "DELIV_PER": previous,
            },
            {
                "SYMBOL": symbol,
                "date": date(2026, 4, 30),
                "TTL_TRD_QNTY": 100_000.0,
                "DELIV_QTY": latest * 1_000,
                "DELIV_PER": latest,
            },
        ]
    )


def test_relative_strength_positive_and_negative():
    idx = pd.bdate_range(end="2026-04-30", periods=70)
    benchmark = pd.Series(100.0, index=idx)
    benchmark.iloc[-1] = 110.0
    strong = pd.Series(100.0, index=idx)
    strong.iloc[-1] = 130.0
    weak = pd.Series(100.0, index=idx)
    weak.iloc[-1] = 105.0

    assert relative_strength_55(strong, benchmark).iloc[-1] > 0
    assert relative_strength_55(weak, benchmark).iloc[-1] < 0


def test_supertrend_bullish_and_bearish_states():
    bullish = _trend_bars(100.0, 150.0)
    bearish = _trend_bars(150.0, 100.0)

    assert bullish["close"].iloc[-1] > supertrend(bullish).iloc[-1]
    assert bearish["close"].iloc[-1] < supertrend(bearish).iloc[-1]


def test_previous_completed_week_high_excludes_current_week():
    bars = _trend_bars(100.0, 140.0)
    current_week_mask = bars.index >= pd.Timestamp("2026-04-27")
    previous_week_mask = (bars.index >= pd.Timestamp("2026-04-20")) & (
        bars.index <= pd.Timestamp("2026-04-24")
    )
    bars.loc[current_week_mask, "high"] = 1_000.0
    bars.loc[previous_week_mask, "high"] = 123.0

    assert previous_completed_week_high(bars, date(2026, 4, 30)) == 123.0


def test_evaluate_symbol_applies_volume_and_delivery_filters():
    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 160_000.0
    benchmark = _trend_bars(100.0, 110.0)["close"]
    delivery = delivery_lookup(_delivery_panel("AAA", latest=55.0, previous=45.0))

    evaluated = evaluate_symbol(
        "AAA",
        bars,
        benchmark,
        date(2026, 4, 30),
        delivery=delivery["AAA"],
    )

    assert evaluated is not None
    row, price_pass, delivery_pass = evaluated
    assert row.volume_ratio == 1.6
    assert price_pass is True
    assert delivery_pass is True


def test_scan_returns_relaxed_when_price_and_delivery_fail():
    full_bars = _trend_bars(100.0, 150.0)
    full_bars.iloc[-1, full_bars.columns.get_loc("volume")] = 160_000.0
    relaxed_only = _trend_bars(100.0, 150.0)
    relaxed_only.iloc[-1, relaxed_only.columns.get_loc("volume")] = 160_000.0
    relaxed_only.loc[
        (relaxed_only.index >= pd.Timestamp("2026-04-20"))
        & (relaxed_only.index <= pd.Timestamp("2026-04-24")),
        "high",
    ] = 155.0
    benchmark = _trend_bars(100.0, 110.0)
    panel = pd.concat(
        [
            _delivery_panel("FULL", latest=60.0, previous=50.0),
            _delivery_panel("RELAX", latest=40.0, previous=45.0),
        ],
        ignore_index=True,
    )

    result = scan_rs_breakouts(
        {"FULL": full_bars, "RELAX": relaxed_only},
        benchmark,
        date(2026, 4, 30),
        delivery_panel=panel,
    )

    assert [row.symbol for row in result.full] == ["FULL"]
    assert {row.symbol for row in result.relaxed} == {"FULL", "RELAX"}


def test_rs_breakout_cli_runs_offline(monkeypatch):
    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 160_000.0
    benchmark = _trend_bars(100.0, 110.0)
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    monkeypatch.setattr(
        main,
        "load_india_delivery_for_scan",
        lambda symbols, as_of: _delivery_panel("AAA", latest=55.0, previous=45.0),
    )

    res = CliRunner().invoke(
        cli,
        [
            "rs-breakout",
            "--tickers",
            "AAA",
            "--as-of",
            "2026-04-30",
            "--no-output-files",
        ],
        obj=fetcher,
    )

    assert res.exit_code == 0, res.output
    assert "India RS Breakout Screen" in res.output
    assert "AAA" in res.output
