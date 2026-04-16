"""Tests that repeated saves do not leave orphaned rows in the DB.

Uses a tmp SQLite file (redirected via tmp_db fixture) so production data
is untouched.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from screener import history
from tests.conftest import make_ohlcv


def _integrity_check(db_path) -> list:
    """Return any FK violations via PRAGMA foreign_key_check."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()
    return violations


def _make_per_ticker(tickers: list[str], return_val: float = 0.05) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": t,
            "entry_close": 100.0,
            "exit_close": 100.0 * (1 + return_val),
            "return_pct": return_val,
            "trading_days": 10,
        }
        for t in tickers
    ])


def _make_hist_per_ticker(tickers: list[str], return_val: float = 0.05) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": t,
            "rank": i + 1,
            "score": 0.5,
            "entry_close": 100.0,
            "exit_close": 100.0 * (1 + return_val),
            "return_pct": return_val,
            "trading_days": 10,
        }
        for i, t in enumerate(tickers)
    ])


def _dummy_run(tmp_db, market: str = "us", criteria: str = "ema") -> int:
    """Insert a minimal run row and return its id."""
    df = make_ohlcv(n_days=5)
    df = df[["date"]].rename(columns={"date": "date"})  # just need something
    # Use history directly — save a synthetic run
    import sqlite3 as _sq
    conn = _sq.connect(tmp_db)
    conn.execute("PRAGMA foreign_keys = ON")
    history._SCHEMA  # ensure schema is applied via _connect
    conn.close()
    # Use history.save_run with a minimal DataFrame
    screen_df = pd.DataFrame([
        {"name": "AAA", "description": "Ticker A", "close": 100.0,
         "change": 0.0, "volume": 1e6, "market_cap_basic": 1e9, "setup_score": 0.5},
    ])
    return history.save_run(market, criteria, 1, screen_df)


class TestBacktestOrphans:
    def test_no_orphans_on_first_save(self, tmp_db):
        run_id = _dummy_run(tmp_db)
        per_ticker = _make_per_ticker(["AAA", "BBB"])
        history.save_backtest(
            run_id=run_id,
            scope="next",
            start_date="2024-01-01",
            end_date="2024-06-01",
            hold_days=120,
            benchmark="AMEX:SPY",
            summary={"total_return": 0.1, "cagr": 0.1, "sharpe": 1.0,
                     "max_drawdown": -0.05, "hit_rate": 0.5, "alpha": 0.02, "beta": 0.9},
            per_ticker=per_ticker,
        )
        assert _integrity_check(tmp_db) == []

    def test_no_orphans_on_resave_same_key(self, tmp_db):
        """Saving with the same (run_id, scope, hold_days) twice must not orphan rows."""
        run_id = _dummy_run(tmp_db)
        kwargs = dict(
            run_id=run_id, scope="next",
            start_date="2024-01-01", end_date="2024-06-01",
            hold_days=120, benchmark="AMEX:SPY",
            summary={"total_return": 0.1, "cagr": 0.1, "sharpe": 1.0,
                     "max_drawdown": -0.05, "hit_rate": 0.5, "alpha": 0.02, "beta": 0.9},
        )
        # First save with 3 tickers
        history.save_backtest(**kwargs, per_ticker=_make_per_ticker(["AAA", "BBB", "CCC"]))
        # Second save with different tickers (simulate re-run)
        history.save_backtest(**kwargs, per_ticker=_make_per_ticker(["DDD", "EEE"]))

        assert _integrity_check(tmp_db) == []

        # Confirm only the second set of tickers remains
        conn = sqlite3.connect(tmp_db)
        conn.execute("PRAGMA foreign_keys = ON")
        rows = conn.execute("SELECT ticker FROM backtest_tickers").fetchall()
        conn.close()
        tickers = {r[0] for r in rows}
        assert tickers == {"DDD", "EEE"}, f"unexpected tickers: {tickers}"


class TestHistoricalBacktestOrphans:
    def test_no_orphans_on_first_save(self, tmp_db):
        per_ticker = _make_hist_per_ticker(["AAA", "BBB"])
        history.save_historical_backtest(
            market="us", criteria="ema",
            as_of_date="2024-01-01", entry_date="2024-01-02",
            exit_date="2024-06-30", hold_days=120, top_n=10,
            universe_label="sp500", universe_size=500,
            matches_total=50, benchmark="AMEX:SPY",
            basket_summary={"total_return": 0.1, "cagr": 0.1, "sharpe": 1.0,
                            "max_drawdown": -0.05, "hit_rate": 0.5, "alpha": 0.02, "beta": 0.9},
            bench_summary={"total_return": 0.08, "cagr": 0.08},
            per_ticker=per_ticker,
        )
        assert _integrity_check(tmp_db) == []

    def test_no_orphans_on_resave_same_key(self, tmp_db):
        kwargs = dict(
            market="us", criteria="ema",
            as_of_date="2024-01-01", entry_date="2024-01-02",
            exit_date="2024-06-30", hold_days=120, top_n=10,
            universe_label="sp500", universe_size=500,
            matches_total=50, benchmark="AMEX:SPY",
            basket_summary={"total_return": 0.1, "cagr": 0.1, "sharpe": 1.0,
                            "max_drawdown": -0.05, "hit_rate": 0.5, "alpha": 0.02, "beta": 0.9},
            bench_summary={"total_return": 0.08, "cagr": 0.08},
        )
        # First save
        history.save_historical_backtest(**kwargs, per_ticker=_make_hist_per_ticker(["AAA", "BBB", "CCC"]))
        # Re-save with different tickers
        history.save_historical_backtest(**kwargs, per_ticker=_make_hist_per_ticker(["DDD", "EEE"]))

        assert _integrity_check(tmp_db) == []

        conn = sqlite3.connect(tmp_db)
        conn.execute("PRAGMA foreign_keys = ON")
        rows = conn.execute("SELECT ticker FROM historical_backtest_tickers").fetchall()
        conn.close()
        tickers = {r[0] for r in rows}
        assert tickers == {"DDD", "EEE"}, f"unexpected tickers: {tickers}"
