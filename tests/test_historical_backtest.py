"""Tests for screener/historical_backtest.py.

All network calls are monkeypatched — no real HTTP requests.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import make_ohlcv

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_universe_file(tmp_path: Path, tickers: list[str]) -> Path:
    p = tmp_path / "universe.txt"
    p.write_text("\n".join(tickers) + "\n")
    return p


def _build_ohlcv_map(tickers: list[str], n_days: int = 400, trend: str = "up") -> dict:
    """Return {ticker: DataFrame} for monkeypatching fetch_ohlcv."""
    return {t: make_ohlcv(n_days=n_days, trend=trend) for t in tickers}


# ── entry / exit date logic ───────────────────────────────────────────────────

class TestEntryExitDates:
    def test_entry_date_is_after_asof(self, tmp_path, tmp_db):
        tickers = ["AAA", "BBB", "CCC"]
        ohlcv_map = _build_ohlcv_map(tickers, n_days=400)
        as_of = date(2023, 10, 1)

        with (
            patch("screener.historical_backtest.fetch_ohlcv", side_effect=lambda t, *a, **kw: ohlcv_map.get(t)),
            patch("screener.historical_backtest.fetch_ohlcv", side_effect=lambda t, *a, **kw: ohlcv_map.get(t)),
            patch("screener.backtest._fetch_benchmark", return_value=(
                pd.Series(1.0, index=pd.date_range("2023-01-02", periods=300)),
                pd.Series(0.0, index=pd.date_range("2023-01-02", periods=300)),
            )),
        ):
            universe_file = _make_universe_file(tmp_path, tickers)
            from screener.historical_backtest import run_historical_backtest
            result = run_historical_backtest(
                market="us",
                criteria_name="ema",
                as_of=as_of,
                hold_days=10,
                top=3,
                universe_path=str(universe_file),
            )

        assert result.entry_date > as_of, (
            f"entry_date {result.entry_date} must be strictly after as_of {as_of}"
        )

    def test_trading_days_le_hold_days(self, tmp_path, tmp_db):
        tickers = ["AAA", "BBB"]
        ohlcv_map = _build_ohlcv_map(tickers, n_days=400)
        hold = 20
        as_of = date(2023, 10, 1)

        with (
            patch("screener.historical_backtest.fetch_ohlcv", side_effect=lambda t, *a, **kw: ohlcv_map.get(t)),
            patch("screener.backtest._fetch_benchmark", return_value=(
                pd.Series(1.0, index=pd.date_range("2023-01-02", periods=300)),
                pd.Series(0.0, index=pd.date_range("2023-01-02", periods=300)),
            )),
        ):
            universe_file = _make_universe_file(tmp_path, tickers)
            from screener.historical_backtest import run_historical_backtest
            result = run_historical_backtest(
                market="us",
                criteria_name="ema",
                as_of=as_of,
                hold_days=hold,
                top=5,
                universe_path=str(universe_file),
            )

        for _, row in result.per_ticker.iterrows():
            assert row["trading_days"] <= hold, (
                f"ticker {row['ticker']}: trading_days {row['trading_days']} > hold {hold}"
            )


# ── basket equal-weighting ────────────────────────────────────────────────────

class TestBasketEqualWeighting:
    def test_basket_is_mean_of_ticker_returns(self, tmp_path, tmp_db):
        """Basket total return should equal the mean of per-ticker returns."""
        tickers = ["AAA", "BBB", "CCC"]
        ohlcv_map = _build_ohlcv_map(tickers, n_days=400, trend="up")
        as_of = date(2023, 10, 1)
        hold = 30

        with (
            patch("screener.historical_backtest.fetch_ohlcv", side_effect=lambda t, *a, **kw: ohlcv_map.get(t)),
            patch("screener.backtest._fetch_benchmark", return_value=(
                pd.Series(1.0, index=pd.date_range("2023-01-02", periods=400)),
                pd.Series(0.0, index=pd.date_range("2023-01-02", periods=400)),
            )),
        ):
            universe_file = _make_universe_file(tmp_path, tickers)
            from screener.historical_backtest import run_historical_backtest
            result = run_historical_backtest(
                market="us",
                criteria_name="ema",
                as_of=as_of,
                hold_days=hold,
                top=3,
                universe_path=str(universe_file),
            )

        pt = result.per_ticker.dropna(subset=["return_pct"])
        if pt.empty:
            pytest.skip("no returns to check")

        mean_ticker_return = float(pt["return_pct"].mean())
        basket_total_return = float(result.summary["basket"]["total_return"])
        assert abs(basket_total_return - mean_ticker_return) < 0.05, (
            f"basket total_return {basket_total_return:.4f} should be close to "
            f"mean ticker return {mean_ticker_return:.4f}"
        )


# ── no-match guard ────────────────────────────────────────────────────────────

class TestNoMatches:
    def test_raises_on_no_matches(self, tmp_path, tmp_db):
        """When no tickers pass the criteria, run_historical_backtest raises RuntimeError."""
        tickers = ["AAA", "BBB"]
        # downtrend → ema criterion never passes
        ohlcv_map = _build_ohlcv_map(tickers, n_days=400, trend="down")
        as_of = date(2023, 10, 1)

        with (
            patch("screener.historical_backtest.fetch_ohlcv", side_effect=lambda t, *a, **kw: ohlcv_map.get(t)),
        ):
            universe_file = _make_universe_file(tmp_path, tickers)
            from screener.historical_backtest import run_historical_backtest
            with pytest.raises(RuntimeError, match="No tickers matched"):
                run_historical_backtest(
                    market="us",
                    criteria_name="ema",
                    as_of=as_of,
                    hold_days=10,
                    top=5,
                    universe_path=str(universe_file),
                )
