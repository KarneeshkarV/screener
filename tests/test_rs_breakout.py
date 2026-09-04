from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.commands import rs_breakout as rs_breakout_cli
from screener.indicators.frames import wilder_atr
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


def _supertrend_pandas_reference(
    bars: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """The Series.iloc recurrence ``supertrend`` was written as.

    Kept verbatim so the numpy rewrite has something to be exactly equal to.
    Nothing in the package calls it; if it ever disagrees with
    :func:`~screener.rs_breakout.supertrend`, trades moved.
    """
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    atr = wilder_atr(high, low, close, period, min_periods=period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(np.nan, index=bars.index, dtype=float)
    final_lower = pd.Series(np.nan, index=bars.index, dtype=float)
    st = pd.Series(np.nan, index=bars.index, dtype=float)
    for i in range(len(bars)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            st.iloc[i] = (
                final_lower.iloc[i]
                if close.iloc[i] >= hl2.iloc[i]
                else final_upper.iloc[i]
            )
            continue
        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
            else final_lower.iloc[i - 1]
        )
        if st.iloc[i - 1] == final_upper.iloc[i - 1]:
            st.iloc[i] = (
                final_lower.iloc[i]
                if close.iloc[i] > final_upper.iloc[i]
                else final_upper.iloc[i]
            )
        else:
            st.iloc[i] = (
                final_upper.iloc[i]
                if close.iloc[i] < final_lower.iloc[i]
                else final_lower.iloc[i]
            )
    st.name = "supertrend"
    return st


def _random_walk_bars(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n)))
    idx = pd.bdate_range("2023-01-02", periods=n)
    span = np.abs(rng.normal(0.0, 0.012, n)) * close
    openp = close * (1.0 + rng.normal(0.0, 0.004, n))
    return pd.DataFrame(
        {
            "open": openp,
            "high": np.maximum(openp, close) + span,
            "low": np.minimum(openp, close) - span,
            "close": close,
            "volume": rng.random(n) * 1e6,
        },
        index=idx,
    )


@pytest.mark.parametrize("seed", range(12))
def test_supertrend_is_bit_identical_to_the_pandas_recurrence(seed: int) -> None:
    """Whipsaw walks hit every branch: seeding, both carry-forwards, both flips."""
    bars = _random_walk_bars(seed, 260)
    result = supertrend(bars)
    expected = _supertrend_pandas_reference(bars)

    assert result.name == expected.name
    assert result.index.equals(expected.index)
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("n", [0, 1, 9, 10, 11])
def test_supertrend_matches_the_reference_on_frames_shorter_than_the_atr(
    n: int,
) -> None:
    """Below ``period`` bars the ATR is all NaN, so every value stays NaN."""
    bars = _random_walk_bars(99, n) if n else pd.DataFrame()
    result = supertrend(bars)
    if n == 0:
        assert result.empty
        return
    np.testing.assert_array_equal(
        result.to_numpy(), _supertrend_pandas_reference(bars).to_numpy()
    )


def test_supertrend_tolerates_a_gap_in_the_middle_of_the_atr_warmup() -> None:
    """A NaN high re-seeds the bands; the numpy form must re-seed identically."""
    bars = _random_walk_bars(7, 120)
    bars.iloc[40:44, bars.columns.get_loc("high")] = np.nan
    np.testing.assert_array_equal(
        supertrend(bars).to_numpy(), _supertrend_pandas_reference(bars).to_numpy()
    )


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
