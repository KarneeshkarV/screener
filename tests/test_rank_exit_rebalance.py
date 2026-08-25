"""Rank-based exit rebalance (rolling engine).

Covers the ``--rank-exit`` feature: a global trading-bar schedule, strict
top-N membership, force-close with reason ``"rank"``, same-day refill of the
freed slot, prepared-panel reuse across rank knobs, and CLI flag parsing.
"""

from __future__ import annotations

import click
import numpy as np
import pandas as pd
import pytest

from screener.backtester.cli_common import RankExitPeriod
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import (
    _DailyRankingSource,
    run_rolling_backtest,
)
from tests.conftest import StubPriceFetcher

_INDEX = pd.bdate_range("2024-01-01", periods=20)


def _ramp(start_px: float, end_px: float, volume: float) -> pd.DataFrame:
    close = pd.Series(np.linspace(start_px, end_px, len(_INDEX)), index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _rise_then_fall(peak: float, volume: float) -> pd.DataFrame:
    """Rises one point per bar for five bars, then falls five points per bar.

    The decline puts every close below its own sma(close, 3), so the entry
    signal - and therefore candidate eligibility - dies on the first falling
    bar while the position is still held.
    """
    vals = [100.0, 101.0, 102.0, 103.0, 104.0]
    vals += [peak - 5 * k for k in range(1, len(_INDEX) - len(vals) + 1)]
    close = pd.Series(vals, index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# Distinct dollar volumes fix the ranking order: AAA > BBB > CCC.
_DATA = {
    "AAA": _rise_then_fall(104.0, 500_000.0),
    "BBB": _ramp(100.0, 130.0, 300_000.0),
    "CCC": _ramp(100.0, 130.0, 200_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}

# Both holdings die together while CCC stays eligible.
_THIN_DATA = {
    "AAA": _rise_then_fall(104.0, 500_000.0),
    "BBB": _rise_then_fall(104.0, 300_000.0),
    "CCC": _ramp(100.0, 130.0, 200_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}


def _cfg(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {
        "market": "us",
        "as_of": _INDEX[-1].date(),
        "hold": 40,  # longer than the window: only rank/eod exits can fire
        "top": 1,
        "strategy_name": None,
        "entry_expr": "close > sma(close, 3)",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA", "BBB", "CCC"),
        "rank_exit_every": 3,
        "rank_universe_size": 2,
    }
    values.update(overrides)
    return BacktestConfig(**values)  # type: ignore[arg-type]


def _run(cfg: BacktestConfig, data: dict[str, pd.DataFrame] | None = None) -> list[object]:
    return run_rolling_backtest(
        cfg,
        StubPriceFetcher(data if data is not None else _DATA),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    ).trades


def test_rank_exit_closes_holding_that_left_the_top_list():
    trades = _run(_cfg())

    # AAA enters first (highest ADV), its signal dies on the first falling bar,
    # and the next scheduled sweep (bar index 5) closes it with reason "rank".
    aaa = trades[0]
    assert str(aaa.ticker) == "AAA"  # type: ignore[attr-defined]
    assert str(aaa.exit_reason) == "rank"  # type: ignore[attr-defined]
    assert pd.Timestamp(aaa.exit_date) == _INDEX[5]  # type: ignore[attr-defined]

    # The freed slot refills the same day from that day's ranking: BBB enters.
    bbb = trades[1]
    assert str(bbb.ticker) == "BBB"  # type: ignore[attr-defined]
    assert str(bbb.exit_reason) == "eod"  # type: ignore[attr-defined]


def test_rank_exit_is_off_by_default():
    trades = _run(_cfg(rank_exit_every=None))
    assert trades
    assert {str(t.exit_reason) for t in trades}.isdisjoint({"rank"})  # type: ignore[attr-defined]


def test_thin_top_list_is_strict():
    """Both holdings leave a top-1 list that holds neither name: both close."""
    trades = _run(_cfg(top=2, rank_universe_size=1), data=_THIN_DATA)

    rank_exits = [t for t in trades if str(t.exit_reason) == "rank"]  # type: ignore[attr-defined]
    assert {str(t.ticker) for t in rank_exits} == {"AAA", "BBB"}  # type: ignore[attr-defined]


def test_schedule_fires_on_every_nth_trading_bar(monkeypatch: pytest.MonkeyPatch):
    """The counter is global: first sweep on bar N, then every N bars after."""
    fired: list[int] = []
    monkeypatch.setattr(
        "screener.backtester.rolling_simulation._rank_exit_sweep",
        lambda **kwargs: fired.append(1),
    )
    source = _DailyRankingSource(
        candidate_matrices=None,  # type: ignore[arg-type]
        bars_by_tv={},
        cfg=_cfg(rank_exit_every=5),
        exit_ast=None,
        fill_model=None,  # type: ignore[arg-type]
        portfolio=None,  # type: ignore[arg-type]
        slot_states={},
        slot_bars={},
        end_ts=_INDEX[-1],
        selection_rows=[],
        warnings=[],
        exit_signals={},
        frame_caches={},
    )
    for _ in range(12):
        source.before_exits(_INDEX[0])

    assert len(fired) == 2  # bars 5 and 10


def test_rank_knobs_reuse_prepared_panels():
    from screener.backtester.rolling_simulation import prepare_rolling_backtest

    cfg = _cfg()
    window = {
        "start_date": _INDEX[0].date(),
        "end_date": _INDEX[-1].date(),
    }
    prepared = prepare_rolling_backtest(cfg, StubPriceFetcher(_DATA), **window)
    swept = cfg.model_copy(update={"rank_universe_size": 7})
    changed_period = cfg.model_copy(update={"rank_exit_every": 9})

    assert prepared.supports(swept)
    assert prepared.supports(changed_period)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("weekly", 5), ("monthly", 21), ("WEEKLY", 5), ("10", 10), (None, None)],
)
def test_rank_exit_flag_parsing(raw: str | None, expected: int | None):
    assert RankExitPeriod().convert(raw, None, None) == expected


def test_rank_exit_flag_rejects_garbage():
    with pytest.raises(click.UsageError):
        RankExitPeriod().convert("fortnightly", None, None)
