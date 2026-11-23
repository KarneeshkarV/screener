"""Regressions for historical reserve rotation and strict-refresh fallbacks.

Bug 1 (B5b): an India reserve whose signal-bar budget cannot buy one whole
share used to be queued anyway, cancelled on its execution bar, and pushed
back to the front of ``reserve_queue``. Every later day it was popped again,
blocking every reserve behind it for the rest of the run.

Bug 2 (B8): under ``strict=True, refresh=True`` the primary price fetcher
raises ``StaleDataError`` instead of serving leftover cache. Neither
``FallbackPriceFetcher`` nor ``ExchangeFallbackPriceFetcher`` caught it, so the
FMP and BSE fallbacks were bypassed entirely.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.data import (
    ExchangeFallbackPriceFetcher,
    FallbackPriceFetcher,
)
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.providers import StaleDataError
from tests.conftest import StubPriceFetcher


def _frame(
    index: pd.DatetimeIndex,
    *,
    close: float,
    volume: float,
) -> pd.DataFrame:
    close_arr = np.full(len(index), float(close))
    open_arr = np.concatenate([[close_arr[0]], close_arr[:-1]])
    return pd.DataFrame(
        {
            "open": open_arr,
            "high": np.maximum(open_arr, close_arr) + 1.0,
            "low": np.minimum(open_arr, close_arr) - 1.0,
            "close": close_arr,
            "volume": np.full(len(index), float(volume)),
        },
        index=index,
    )


def test_unaffordable_india_reserve_does_not_starve_the_slot():
    """A rich reserve ahead of an affordable one must not block it forever.

    ``UNAFF`` costs more than the whole slot budget, so no whole share is
    affordable. It ranks ahead of ``AFF`` because it prints more dollar
    volume. Once the single active slot frees, the affordable reserve must be
    promoted promptly instead of the queue looping on ``UNAFF`` every day.
    """
    n = 30
    index = pd.bdate_range("2024-01-01", periods=n)
    as_of_i = 20

    # Dollar volume ranks ACTIVE > UNAFF > AFF; UNAFF is affordable only at a
    # fractional share, which an India whole-share market cannot buy.
    panel = {
        "ACTIVE.NS": _frame(index, close=100.0, volume=5_000_000.0),
        "UNAFF.NS": _frame(index, close=20_000.0, volume=10_000.0),
        "AFF.NS": _frame(index, close=100.0, volume=10_000.0),
        "^NSEI": _frame(index, close=100.0, volume=10_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=index[as_of_i].date(),
        benchmark="^NSEI",
        tickers=("ACTIVE", "UNAFF", "AFF"),
        entry_expr="close > 0",
        exit_expr=None,
        hold=1,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        top=1,
        strategy_name=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        reinvest=True,
        compounding=False,
        reserve_multiple=3,
    )

    result = run_backtest(cfg, StubPriceFetcher(panel))

    traded = {trade.ticker for trade in result.trades}
    assert "AFF" in traded, (
        "the affordable reserve behind an unaffordable one never filled: "
        f"trades={sorted(traded)}"
    )
    assert "UNAFF" not in traded, "an unaffordable whole-share India lot filled"

    # The affordable reserve is promoted out of the free slot without waiting
    # for the queue to loop through the unaffordable name day after day.
    aff_entries = [trade.entry_date for trade in result.trades if trade.ticker == "AFF"]
    assert aff_entries
    assert min(aff_entries) <= index[as_of_i + 5].date()


class _RaisingFetcher:
    """Primary stand-in for a strict refresh that refused stale cache."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    def fetch(self, tickers, start, end):
        self.calls.append(list(tickers))
        raise self.error


class _FramesFetcher:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[list[str]] = []

    def fetch(self, tickers, start, end):
        ticker_list = list(tickers)
        self.calls.append(ticker_list)
        return {
            ticker: self.frames.get(ticker, pd.DataFrame()) for ticker in ticker_list
        }


def _fresh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )


def test_fallback_fetcher_uses_fallback_when_primary_is_stale():
    stale = StaleDataError("strict refresh could not refresh bars for AAA")
    primary = _RaisingFetcher(stale)
    fallback = _FramesFetcher({"AAA": _fresh_frame()})
    fetcher = FallbackPriceFetcher(primary, fallback)

    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))

    assert fallback.calls == [["AAA"]]
    assert out["AAA"].equals(_fresh_frame())


def test_fallback_fetcher_reraises_when_fallback_is_empty():
    stale = StaleDataError("strict refresh could not refresh bars for AAA")
    primary = _RaisingFetcher(stale)
    fallback = _FramesFetcher({})
    fetcher = FallbackPriceFetcher(primary, fallback)

    with pytest.raises(StaleDataError, match="strict refresh could not refresh"):
        fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))


def test_fallback_fetcher_reraises_when_fallback_is_also_stale():
    primary = _RaisingFetcher(StaleDataError("primary stale"))
    secondary = _RaisingFetcher(StaleDataError("fallback stale"))
    fetcher = FallbackPriceFetcher(primary, secondary)

    with pytest.raises(StaleDataError, match="primary stale"):
        fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))


class _StaleThenBseFetcher:
    """Inner fetcher that refuses the first batch and serves BSE on the retry."""

    def __init__(self, bse: dict[str, pd.DataFrame]) -> None:
        self.bse = bse
        self.calls: list[list[str]] = []

    def fetch(self, tickers, start, end):
        ticker_list = list(tickers)
        self.calls.append(ticker_list)
        if len(self.calls) == 1:
            raise StaleDataError("strict refresh could not refresh bars for AAA.NS")
        return {ticker: self.bse.get(ticker, pd.DataFrame()) for ticker in ticker_list}


def test_exchange_fallback_retries_bse_after_a_stale_nse_refusal():
    frame = _fresh_frame()
    inner = _StaleThenBseFetcher({"AAA.BO": frame})
    fetcher = ExchangeFallbackPriceFetcher(inner)

    out = fetcher.fetch(["AAA.NS"], date(2024, 1, 1), date(2024, 1, 5))

    assert inner.calls == [["AAA.NS"], ["AAA.BO"]]
    assert out["AAA.NS"].equals(frame)


def test_exchange_fallback_reraises_when_bse_is_also_empty():
    inner = _StaleThenBseFetcher({})
    fetcher = ExchangeFallbackPriceFetcher(inner)

    with pytest.raises(StaleDataError, match="strict refresh could not refresh"):
        fetcher.fetch(["AAA.NS"], date(2024, 1, 1), date(2024, 1, 5))
