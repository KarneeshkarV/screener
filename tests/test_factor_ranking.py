"""Core test: a ``rank_score`` column overrides dollar-volume candidate ranking.

The rolling backtester historically ranks same-day entry candidates by signal-day
dollar volume. Factor portfolios instead need cross-sectional factor ranking, so
``screener.backtester.rolling`` now ranks by a ``rank_score`` column when one is
present in the prepared bars. This test pins that override: the two tickers are
built so dollar-volume ordering and ``rank_score`` ordering DISAGREE, and only one
slot is available, so whichever ranks first is the only one that trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from tests.conftest import StubPriceFetcher

_START = "2024-01-01"
_N = 30
_INDEX = pd.bdate_range(_START, periods=_N)


def _frame(close_px: float, volume: float, score: float | None) -> pd.DataFrame:
    close = pd.Series(np.linspace(close_px, close_px + 20.0, _N), index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    data = {
        "open": openp,
        "high": high,
        "low": low,
        "close": close,
        "volume": pd.Series(volume, index=_INDEX, dtype=float),
    }
    if score is not None:
        data["rank_score"] = pd.Series(score, index=_INDEX, dtype=float)
    return pd.DataFrame(data)


def _cfg() -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=3,
        top=1,  # single slot -> only the top-ranked candidate ever trades
        strategy_name=None,
        entry_expr="close > 0",  # always eligible
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("LOWVOL_HISCORE", "HIVOL_LOSCORE"),
        min_price=None,
        min_avg_dollar_volume=None,
    )


def test_rank_score_overrides_dollar_volume() -> None:
    # HIVOL_LOSCORE has the larger dollar volume but the LOWER factor score, so
    # the legacy ranker would pick it; the factor ranker must pick the other.
    data = {
        "LOWVOL_HISCORE": _frame(100.0, volume=10_000.0, score=9.0),
        "HIVOL_LOSCORE": _frame(100.0, volume=999_000.0, score=1.0),
        "SPY": _frame(400.0, volume=1_000_000.0, score=None),
    }
    result = run_rolling_backtest(
        _cfg(),
        StubPriceFetcher(data),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"LOWVOL_HISCORE"}, traded
    # Every recorded selection ranked the high-score name first.
    assert all(t.ticker == "LOWVOL_HISCORE" for t in result.trades)


def test_no_rank_score_keeps_dollar_volume_ranking() -> None:
    # Same prices, NO score column anywhere -> legacy dollar-volume ordering
    # picks the high-volume name.
    data = {
        "LOWVOL": _frame(100.0, volume=10_000.0, score=None),
        "HIVOL": _frame(100.0, volume=999_000.0, score=None),
        "SPY": _frame(400.0, volume=1_000_000.0, score=None),
    }
    cfg = _cfg().model_copy(update={"tickers": ("LOWVOL", "HIVOL")})
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    assert {t.ticker for t in result.trades} == {"HIVOL"}


def test_equal_rank_score_breaks_tie_by_dollar_volume() -> None:
    # Two names share the SAME factor score, so the cross-sectional ordering is a
    # tie; it must resolve by signal-day dollar volume (liquidity) rather than by
    # arbitrary universe/column order. The high-volume name should win the slot.
    data = {
        "TIE_LOWVOL": _frame(100.0, volume=10_000.0, score=5.0),
        "TIE_HIVOL": _frame(100.0, volume=999_000.0, score=5.0),
        "SPY": _frame(400.0, volume=1_000_000.0, score=None),
    }
    cfg = _cfg().model_copy(update={"tickers": ("TIE_LOWVOL", "TIE_HIVOL")})
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    assert {t.ticker for t in result.trades} == {"TIE_HIVOL"}


def test_mixed_universe_missing_rank_score_warns() -> None:
    # Factor ranking is active (one name carries a score), but another candidate
    # lacks the column entirely. That name is silently excluded from selection;
    # the run must surface a warning naming it rather than dropping it without
    # trace.
    data = {
        "HAS_SCORE": _frame(100.0, volume=10_000.0, score=9.0),
        "NO_SCORE": _frame(100.0, volume=999_000.0, score=None),
        "SPY": _frame(400.0, volume=1_000_000.0, score=None),
    }
    cfg = _cfg().model_copy(update={"tickers": ("HAS_SCORE", "NO_SCORE")})
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    # Only the scored name is selectable.
    assert {t.ticker for t in result.trades} == {"HAS_SCORE"}
    assert any("rank_score" in w and "NO_SCORE" in w for w in result.warnings), (
        result.warnings
    )
