from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.commands import rs_breakout as rs_breakout_cli
from screener.relative_strength import relative_strength_ratio
from screener.rs_breakout import (
    build_signal_frame,
    delivery_lookup,
    evaluate_symbol,
    normalize_bars,
    previous_completed_week_high,
    scan_rs_breakouts,
    supertrend,
    write_json,
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

    assert relative_strength_ratio(strong, benchmark).iloc[-1] > 0
    assert relative_strength_ratio(weak, benchmark).iloc[-1] < 0


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


def test_run_rs_breakout_screen_offline(monkeypatch):
    from rich.console import Console

    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 160_000.0
    benchmark = _trend_bars(100.0, 110.0)
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    monkeypatch.setattr(
        rs_breakout_cli,
        "load_india_delivery_for_scan",
        lambda symbols, as_of: _delivery_panel("AAA", latest=55.0, previous=45.0),
    )

    result = rs_breakout_cli.run_rs_breakout_screen(
        "india",
        as_of=date(2026, 4, 30),
        benchmark=None,
        history_days=220,
        cache_ttl=None,
        refresh=False,
        console=Console(),
        tickers="AAA",
        fetcher=fetcher,
    )

    assert result.as_of == date(2026, 4, 30)
    assert any(row.symbol == "AAA" for row in result.full + result.relaxed)


def test_rs_breakout_cli_runs_offline(monkeypatch):
    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 160_000.0
    benchmark = _trend_bars(100.0, 110.0)
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    monkeypatch.setattr(
        rs_breakout_cli,
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
    assert "INDIA RS Breakout Screen" in res.output
    assert "AAA" in res.output


def test_write_json_serializes_result_dates(tmp_path) -> None:
    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 160_000.0
    benchmark = _trend_bars(100.0, 110.0)
    result = scan_rs_breakouts(
        {"AAA": bars},
        benchmark,
        date(2026, 4, 30),
        delivery_panel=_delivery_panel("AAA", latest=55.0, previous=45.0),
    )

    path = tmp_path / "rs_breakout.json"
    write_json(result, path)

    payload = json.loads(path.read_text())
    assert payload["as_of"] == "2026-04-30"
    assert payload["full"][0]["date"] == "2026-04-30"


def _equivalence_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Noisy bars, benchmark and a per-bar delivery panel for one symbol.

    Deliberately choppy, with occasional volume spikes, so the entry rule flips
    on and off inside the compared window: a monotone trend would let both
    paths agree by always saying yes.
    """
    rng = np.random.default_rng(11)
    periods = 220
    index = pd.bdate_range(end="2026-04-30", periods=periods)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0015, 0.02, periods)),
        index=index,
        dtype=float,
    )
    openp = close.shift(1).fillna(100.0)
    volume = rng.uniform(80_000.0, 200_000.0, periods)
    volume = np.where(rng.random(periods) < 0.35, volume * 2.5, volume)
    bars = pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) * 1.01,
            "low": pd.concat([openp, close], axis=1).min(axis=1) * 0.99,
            "close": close,
            "volume": pd.Series(volume, index=index, dtype=float),
        }
    )
    benchmark_close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, periods)),
        index=index,
        dtype=float,
    )
    delivery = pd.DataFrame(
        {
            "SYMBOL": ["AAA"] * periods,
            "date": index,
            "DELIV_PER": rng.uniform(20.0, 80.0, periods),
        }
    )
    return bars, benchmark_close, delivery


@pytest.mark.parametrize("require_delivery", [True, False])
def test_scalar_scan_and_vectorized_backtest_entries_agree(require_delivery: bool):
    """The live scan's verdict must equal the backtest plugin's entry flag.

    ``evaluate_symbol`` (the scan, one bar at a time) and ``build_signal_frame``
    (the backtest plugin, the whole history at once) now share one expression of
    the rule. Nothing pinned them equivalent while it was written twice, so walk
    the last 120 bars and require the same answer on every one, for both the
    India variant (delivery increase required) and the US variant (not).
    """
    bars, benchmark_close, delivery = _equivalence_dataset()
    as_of = bars.index[-1].date()
    vectorized = build_signal_frame(
        normalize_bars(bars, as_of),
        benchmark_close,
        delivery_panel=delivery,
        symbol="AAA",
        require_delivery=require_delivery,
    )

    entries: list[bool] = []
    for timestamp in bars.index[-120:]:
        evaluated = evaluate_symbol(
            "AAA",
            bars.loc[bars.index <= timestamp],
            benchmark_close.loc[benchmark_close.index <= timestamp],
            timestamp.date(),
            delivery=delivery_lookup(delivery[delivery["date"] <= timestamp]).get(
                "AAA"
            ),
        )
        if evaluated is None:
            # No row at all means the base filters failed, which is the scan's
            # way of saying "no entry".
            scalar_entry = False
        else:
            _row, price_pass, delivery_pass = evaluated
            scalar_entry = price_pass and (delivery_pass or not require_delivery)
        assert scalar_entry == bool(vectorized.loc[timestamp, "rs_breakout_entry"]), (
            f"scalar/vectorized RS-breakout entry disagree on {timestamp.date()}"
        )
        entries.append(scalar_entry)

    # Guard the guard: an all-False window would agree vacuously.
    assert 0 < sum(entries) < len(entries)
