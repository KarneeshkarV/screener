"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener import history


def make_ohlcv(
    n_days: int = 300,
    start_date: date = date(2023, 1, 2),
    trend: str = "up",   # "up" | "down" | "flat"
    base_close: float = 100.0,
    base_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Return a synthetic daily OHLCV DataFrame.

    ``trend="up"``  — close rises 0.1% each bar  → EMA5 > EMA20 > EMA100 > EMA200
    ``trend="down"``— close falls 0.1% each bar  → bearish stack
    ``trend="flat"``— close is constant           → all EMAs equal
    """
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    closes: list[float] = []
    c = base_close
    for i in range(n_days):
        if trend == "up":
            c *= 1.001
        elif trend == "down":
            c *= 0.999
        closes.append(round(c, 4))

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "adj_close": closes,
        "volume": [base_volume] * n_days,
    })


@pytest.fixture
def uptrend_ohlcv() -> pd.DataFrame:
    """300 bars of a steady uptrend — EMA stack is bullish."""
    return make_ohlcv(n_days=300, trend="up")


@pytest.fixture
def downtrend_ohlcv() -> pd.DataFrame:
    """300 bars of a steady downtrend — EMA stack is bearish."""
    return make_ohlcv(n_days=300, trend="down")


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """300 bars flat — all EMAs converge to the same value."""
    return make_ohlcv(n_days=300, trend="flat")


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect history.DB_PATH to a temp file and return its path."""
    db = tmp_path / "test_history.db"
    monkeypatch.setattr(history, "DB_PATH", db)
    return db
