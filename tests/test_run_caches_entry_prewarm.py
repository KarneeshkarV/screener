"""``_RunCaches.prewarm_entry_signals`` must store what the lazy path stores.

The historical engine prewarms entry signals with one panel pass instead of
walking the entry AST once per ticker. A prewarmed cache that held a different
value, or a different failure, than :meth:`_RunCaches.entry_signal` would
change who is bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from screener.backtester.core import _RunCaches
from screener.backtester.historical import select_candidates
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import PineError, parse

_DAYS = pd.bdate_range("2024-01-01", periods=60)


def _bars(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    close = 100 + np.random.default_rng(seed).normal(0, 1, len(index)).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1e6,
        },
        index=index,
    )


def _assert_same_cache(bars: dict[str, pd.DataFrame], entry) -> None:
    warm = _RunCaches()
    warm.prewarm_entry_signals(bars, entry)
    lazy = _RunCaches()
    for ticker, frame in bars.items():
        expected = lazy.entry_signal(ticker, frame, entry)
        got = warm.entry_signal(ticker, frame, entry)
        if isinstance(expected, PineError):
            assert isinstance(got, PineError)
            assert str(got) == str(expected)
        else:
            assert isinstance(got, pd.Series)
            pdt.assert_series_equal(got, expected, check_index_type=True)
            assert got.name == expected.name
            assert got.index.name == expected.index.name


def test_prewarmed_entries_keep_each_tickers_index_and_series_metadata() -> None:
    # ``close > sma(close, 5)`` comes back named "close" on the lazy path, and
    # the panel stacks B under A's index, whose name differs.
    b = _bars(_DAYS, 2)
    b.index = b.index.rename("session")
    _assert_same_cache({"A": _bars(_DAYS, 1), "B": b}, parse("close > sma(close, 5)"))
    _assert_same_cache({"A": _bars(_DAYS, 1), "B": b}, parse("sma(close, 5)"))


def test_a_panel_build_overflow_keeps_each_tickers_own_error() -> None:
    bars = {}
    for ticker, seed in (("A", 1), ("B", 2)):
        frame = _bars(_DAYS, seed)
        frame["volume"] = pd.Series([10**400] * len(_DAYS), index=_DAYS, dtype=object)
        bars[ticker] = frame
    _assert_same_cache(bars, parse("bogus(volume) > 0"))


def test_a_cached_entry_failure_does_not_hold_the_panel_alive() -> None:
    """A traceback through the panel pass keeps its frames and AST cache alive.

    The cached error must come from the ticker's own evaluation instead.
    """
    entry = parse("close > sma(close, 5) and score > 0")
    bars = {
        "A": _bars(_DAYS, 1),
        "B": _bars(_DAYS, 2),
        "C": _bars(_DAYS, 3).drop(columns="volume"),
    }
    caches = _RunCaches()
    caches.prewarm_entry_signals(bars, entry)
    for ticker, frame in bars.items():
        caches.entry_signal(ticker, frame, entry)

    for error in caches.entry_signals.values():
        assert isinstance(error, PineError)
        assert "evaluate_panel_many" not in _frame_names(error.__traceback__)


def _frame_names(tb) -> list[str]:
    names = []
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return names


def test_a_name_the_price_filter_rejects_is_never_evaluated() -> None:
    good = _bars(_DAYS, 1)
    cheap = _bars(_DAYS, 2) * 0 + 1.0
    cheap["volume"] = pd.Series(["bad"] * len(_DAYS), index=_DAYS, dtype=object)
    cfg = BacktestConfig(
        market="us",
        as_of=_DAYS[-2].date(),
        hold=1,
        top=2,
        entry_expr="volume > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        min_price=5.0,
        min_avg_dollar_volume=None,
    )

    picks, warnings = select_candidates(
        {"GOOD": good, "CHEAP": cheap},
        parse("volume > 0"),
        _DAYS[-2],
        top_n=2,
        lookback_required=0,
        cfg=cfg,
    )

    assert list(picks["ticker"]) == ["GOOD"]
    assert warnings == ["filtered 1 tickers on price/liquidity filters"]


def test_prewarmed_entries_match_the_lazy_evaluation() -> None:
    bars = {
        "A": _bars(_DAYS, 1),
        "B": _bars(_DAYS, 2),
        "SHORT": _bars(_DAYS[10:], 3),
        "BROKEN": _bars(_DAYS, 4).drop(columns="volume"),
    }
    entry = parse("close > sma(close, 5) and volume > 0")
    warm = _RunCaches()
    warm.prewarm_entry_signals(bars, entry)
    lazy = _RunCaches()
    for ticker, frame in bars.items():
        expected = lazy.entry_signal(ticker, frame, entry)
        got = warm.entry_signal(ticker, frame, entry)
        if isinstance(expected, PineError):
            assert isinstance(got, PineError)
            assert str(got) == str(expected)
        else:
            assert isinstance(got, pd.Series)
            pdt.assert_series_equal(got, expected)


def test_prewarm_keeps_an_entry_already_cached() -> None:
    bars = {"A": _bars(_DAYS, 1)}
    entry = parse("close > 0")
    caches = _RunCaches()
    sentinel = pd.Series(False, index=_DAYS)
    caches.entry_signals["A"] = sentinel
    caches.prewarm_entry_signals(bars, entry)
    assert caches.entry_signals["A"] is sentinel
