"""The one-day candidate entry point reads the rolling engine's own matrices.

Stage 2 of ``docs/plans/unify-screen-backtest.md``: a screen must not own a
second implementation of "who is a candidate today". These tests pin that
:func:`day_candidates_from_panel` is a reshape of
:func:`~screener.backtester.rolling_candidates._candidate_rows_for_day` over
the same matrices, not a parallel path that happens to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import parse
from screener.backtester.price_panel import PricePanel
from screener.backtester.rolling_candidates import _candidate_rows_for_day
from screener.backtester.signal_panel import (
    SignalPanelInputs,
    SignalProgram,
    build_day_candidates,
    build_signal_panel,
    day_candidates_from_panel,
)

ENTRY_EXPR = "close > sma(close, 5)"
_TICKERS = ("NSE:AAA", "NSE:BBB", "NSE:CCC", "NSE:DDD")


def _bars(seed: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic OHLCV bars with enough movement to fire the entry."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, size=len(dates)).cumsum()
    close = 100.0 + steps * 3.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(dates), 1_000_000.0 + seed * 10_000.0),
        },
        index=dates,
    )


@pytest.fixture
def window() -> tuple[SignalPanelInputs, PricePanel, SignalProgram]:
    dates = pd.bdate_range("2024-01-01", periods=60)
    bars_by_tv = {tv: _bars(i, dates) for i, tv in enumerate(_TICKERS)}
    panel = PricePanel(
        tv_symbols=list(_TICKERS),
        yf_by_tv={tv: tv.split(":")[-1] for tv in _TICKERS},
        bars_by_tv=bars_by_tv,
        benchmark=bars_by_tv["NSE:AAA"]["close"],
        lookback=5,
        master_dates=list(dates),
    )
    inputs = SignalPanelInputs(
        market="india",
        entry_expr=ENTRY_EXPR,
        exit_expr=None,
        regime_filter=(),
        earnings_blackout_days=None,
        sector_neutral=False,
        min_price=None,
        min_avg_dollar_volume=None,
        avg_dollar_volume_window=20,
        membership_added=(),
        membership_windows=(),
        dynamic_universe_size=None,
        dynamic_universe_lookback=20,
        dynamic_universe_rebalance="never",
    )
    program = SignalProgram(entry_ast=parse(ENTRY_EXPR), exit_ast=None, lookback=5)
    return inputs, panel, program


def _panel(window):
    inputs, price_panel, program = window
    return build_signal_panel(
        inputs,
        price_panel,
        program=program,
        start_ts=price_panel.master_dates[0],
        end_ts=price_panel.master_dates[-1],
        warnings=[],
    )


def test_every_day_matches_the_rolling_engine_row_for_row(window) -> None:
    """The one-day path returns the rolling engine's rows, every day, in order."""
    signals = _panel(window)
    matrices = signals.candidate_matrices
    assert matrices is not None
    days = list(matrices.signal_mat.index)
    assert len(days) == 60

    seen_any = False
    for day in days:
        expected, _ = _candidate_rows_for_day(day, matrices, exclude=set(), limit=None)
        actual = day_candidates_from_panel(signals, day)
        assert actual.as_of == day
        assert len(actual.candidates) == len(expected)
        for got, want in zip(actual.candidates, expected, strict=True):
            assert got.ticker == want["ticker"]
            assert got.rank == want["rank"]
            assert got.role == want["role"]
            assert got.signal_idx == want["signal_idx"]
            assert got.as_of_close == want["as_of_close"]
            assert got.as_of_volume == want["as_of_volume"]
            assert got.as_of_dollar_vol == want["as_of_dollar_vol"]
        seen_any = seen_any or bool(expected)

    # A window where nothing ever fires would make the assertions vacuous.
    assert seen_any, "fixture produced no candidates on any day"


def test_build_day_candidates_matches_the_prebuilt_panel_path(window) -> None:
    """Building the panel internally gives the same answer as reusing one."""
    inputs, price_panel, program = window
    signals = _panel(window)
    as_of = price_panel.master_dates[40]
    from_prebuilt = day_candidates_from_panel(signals, as_of)
    built = build_day_candidates(
        inputs,
        price_panel,
        program=program,
        as_of=as_of,
        start_ts=price_panel.master_dates[0],
        end_ts=price_panel.master_dates[-1],
        warnings=[],
    )
    assert built == from_prebuilt


def test_as_of_snaps_back_to_the_last_bar_on_or_before(window) -> None:
    """A non-trading date resolves to the previous master-calendar bar."""
    signals = _panel(window)
    last_bar = list(signals.candidate_matrices.signal_mat.index)[-1]
    weekend = (last_bar + pd.Timedelta(days=1)).date()
    snapped = day_candidates_from_panel(signals, weekend)
    assert snapped.as_of == last_bar
    assert snapped == day_candidates_from_panel(signals, last_bar)


def test_date_before_the_window_has_no_candidates(window) -> None:
    signals = _panel(window)
    first_bar = list(signals.candidate_matrices.signal_mat.index)[0]
    before = (first_bar - pd.Timedelta(days=3)).date()
    empty = day_candidates_from_panel(signals, before)
    assert empty.as_of is None
    assert empty.candidates == ()


def test_exclude_and_limit_reach_the_rolling_scan(window) -> None:
    """``exclude`` and ``limit`` are the rolling engine's own, not reimplemented."""
    signals = _panel(window)
    matrices = signals.candidate_matrices
    day = next(
        d
        for d in matrices.signal_mat.index
        if len(_candidate_rows_for_day(d, matrices, exclude=set(), limit=None)[0]) >= 2
    )
    full = day_candidates_from_panel(signals, day)

    capped = day_candidates_from_panel(signals, day, limit=1)
    assert len(capped.candidates) == 1
    assert capped.candidates[0] == full.candidates[0]

    dropped = day_candidates_from_panel(
        signals, day, exclude=[full.candidates[0].ticker]
    )
    assert full.candidates[0].ticker not in {c.ticker for c in dropped.candidates}
    # Ranks are recomputed by the rolling scan, so the survivor is now rank 1.
    assert [c.rank for c in dropped.candidates] == list(
        range(1, len(dropped.candidates) + 1)
    )


def test_rank_basis_reports_dollar_volume_without_a_factor_score(window) -> None:
    """No ``rank_score`` column means the legacy dollar-volume ordering."""
    signals = _panel(window)
    day = next(
        d
        for d in signals.candidate_matrices.signal_mat.index
        if day_candidates_from_panel(signals, d).candidates
    )
    result = day_candidates_from_panel(signals, day)
    assert {c.rank_basis for c in result.candidates} == {"as_of_dollar_vol"}
    assert all(c.rank_score is None for c in result.candidates)


def test_factor_ranking_reports_the_as_of_score_for_each_name(window) -> None:
    """With a ``rank_score`` column the reported score is that day's own value.

    The dollar-volume branch leaves ``rank_score`` at ``None``, so without this
    the ``[row_position, col_by_ticker[ticker]]`` lookup in the one-day reshape
    is never exercised and a transposed or stale index would go unnoticed.
    """
    inputs, price_panel, program = window
    scored_bars = {}
    for offset, (tv, bars) in enumerate(price_panel.bars_by_tv.items()):
        frame = bars.copy()
        # Distinct per ticker and per bar, so a wrong row or column shows up.
        frame["rank_score"] = np.arange(len(frame), dtype=float) + offset * 1000.0
        scored_bars[tv] = frame
    scored_panel = PricePanel(
        tv_symbols=price_panel.tv_symbols,
        yf_by_tv=price_panel.yf_by_tv,
        bars_by_tv=scored_bars,
        benchmark=price_panel.benchmark,
        lookback=price_panel.lookback,
        master_dates=price_panel.master_dates,
    )
    signals = build_signal_panel(
        inputs,
        scored_panel,
        program=program,
        start_ts=scored_panel.master_dates[0],
        end_ts=scored_panel.master_dates[-1],
        warnings=[],
    )
    matrices = signals.candidate_matrices
    assert matrices is not None
    assert matrices.rank_score_np is not None, "fixture failed to install rank_score"

    day = next(
        d
        for d in matrices.signal_mat.index
        if len(day_candidates_from_panel(signals, d).candidates) >= 2
    )
    result = day_candidates_from_panel(signals, day)
    bar_position = scored_panel.master_dates.index(day)

    assert {c.rank_basis for c in result.candidates} == {"rank_score"}
    for candidate in result.candidates:
        expected = scored_bars[candidate.ticker]["rank_score"].iloc[bar_position]
        assert candidate.rank_score == pytest.approx(expected)

    # Ranking is by score descending, which is the whole point of the branch.
    scores = [c.rank_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
