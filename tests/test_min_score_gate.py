"""``min_score`` is a candidate gate, not a screen-side post-filter.

Step 1 of the flag-parity work. ``setup_score`` used to be computed in
``screener.screen_candidates`` after the candidate layer had already answered,
so a backtest could never see it and the screen could never gate on it. It now
lives where every other gate lives - the per-day candidate scan the rolling
engine and the screen both call - which is what lets one ``--min-score`` mean
the same thing on both commands.

The percentile is taken over the eligible set *before* ``exclude`` and before
``limit``: a name's score must say where it stands in the field, not where it
stands among whatever the caller had room for.
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
    build_signal_panel,
    day_candidates_from_panel,
)

ENTRY_EXPR = "close > 0"
_TICKERS = ("NSE:AAA", "NSE:BBB", "NSE:CCC", "NSE:DDD")


def _bars(dates: pd.DatetimeIndex, volume: float) -> pd.DataFrame:
    """Flat bars whose only distinguishing feature is dollar volume.

    The entry fires on every name every day, so the field is the whole
    universe and the ranking basis is dollar volume - the legacy basis, and
    the one a strategy without a factor score uses.
    """
    close = np.full(len(dates), 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(dates), volume),
        },
        index=dates,
    )


def _panel(volumes: dict[str, float], *, min_score: float | None = None):
    dates = pd.bdate_range("2024-01-01", periods=30)
    bars_by_tv = {tv: _bars(dates, vol) for tv, vol in volumes.items()}
    price_panel = PricePanel(
        tv_symbols=list(volumes),
        yf_by_tv={tv: tv.split(":")[-1] for tv in volumes},
        bars_by_tv=bars_by_tv,
        benchmark=next(iter(bars_by_tv.values()))["close"],
        lookback=1,
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
        min_score=min_score,
    )
    program = SignalProgram(entry_ast=parse(ENTRY_EXPR), exit_ast=None, lookback=1)
    signals = build_signal_panel(
        inputs,
        price_panel,
        program=program,
        start_ts=dates[0],
        end_ts=dates[-1],
        warnings=[],
        require_next_bar=False,
    )
    return signals, dates[-1]


_VOLUMES = {
    "NSE:AAA": 1_000.0,
    "NSE:BBB": 2_000.0,
    "NSE:CCC": 3_000.0,
    "NSE:DDD": 4_000.0,
}


def _scores_by_ticker(signals, as_of, **kwargs) -> dict[str, float]:
    day = day_candidates_from_panel(signals, as_of, **kwargs)
    return {c.ticker: c.setup_score for c in day.candidates}


def test_setup_score_is_a_percentile_of_the_eligible_field() -> None:
    signals, as_of = _panel(_VOLUMES)

    assert _scores_by_ticker(signals, as_of) == {
        "NSE:DDD": 100.0,
        "NSE:CCC": 75.0,
        "NSE:BBB": 50.0,
        "NSE:AAA": 25.0,
    }


def test_min_score_drops_the_names_below_the_threshold() -> None:
    signals, as_of = _panel(_VOLUMES, min_score=60.0)

    assert list(_scores_by_ticker(signals, as_of)) == ["NSE:DDD", "NSE:CCC"]


def test_the_percentile_is_taken_before_exclude_and_before_limit() -> None:
    # Holding the top name already must not promote the rest of the field: an
    # excluded name is still part of the field it was ranked against.
    signals, as_of = _panel(_VOLUMES)

    excluded = _scores_by_ticker(signals, as_of, exclude={"NSE:DDD"})
    limited = _scores_by_ticker(signals, as_of, limit=2)

    assert excluded == {"NSE:CCC": 75.0, "NSE:BBB": 50.0, "NSE:AAA": 25.0}
    assert limited == {"NSE:DDD": 100.0, "NSE:CCC": 75.0}


def test_a_field_of_one_scores_full_marks_and_clears_any_threshold() -> None:
    # A percentile of a singleton is 100 by definition. The gate must not read
    # that as "no information" and drop the only name there is.
    signals, as_of = _panel({"NSE:AAA": 1_000.0}, min_score=99.0)

    assert _scores_by_ticker(signals, as_of) == {"NSE:AAA": 100.0}


def test_ties_share_the_average_rank() -> None:
    # ``rank(pct=True)`` averages ties, so two names on the same dollar volume
    # get the same score. Anything else would make the score depend on column
    # order, which is not a property of the name.
    signals, as_of = _panel(
        {"NSE:AAA": 1_000.0, "NSE:BBB": 1_000.0, "NSE:CCC": 3_000.0}
    )

    scores = _scores_by_ticker(signals, as_of)
    assert scores["NSE:AAA"] == scores["NSE:BBB"] == pytest.approx(50.0)
    assert scores["NSE:CCC"] == 100.0


def test_the_rolling_scan_and_the_one_day_reshape_agree() -> None:
    # ``day_candidates_from_panel`` is a reshape of the rolling engine's own
    # per-day scan; the score has to travel with the row, not be recomputed.
    signals, as_of = _panel(_VOLUMES, min_score=60.0)
    matrices = signals.candidate_matrices
    assert matrices is not None

    rows, _ = _candidate_rows_for_day(as_of, matrices, exclude=set())

    assert [(r["ticker"], r["setup_score"]) for r in rows] == [
        (c.ticker, c.setup_score)
        for c in day_candidates_from_panel(signals, as_of).candidates
    ]
