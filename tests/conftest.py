from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import pytest


def make_bars(
    start: str = "2024-01-01",
    n: int = 60,
    open_base: float = 100.0,
    drift: float = 0.0,
    seed: int = 0,
    spikes: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """Build a deterministic synthetic OHLCV frame.

    ``spikes`` maps positional index → {"open","high","low","close","volume"}
    overrides (any subset) applied to that bar.
    """
    rng = np.random.default_rng(seed)
    close = open_base + np.cumsum(rng.normal(drift, 0.5, n))
    openp = np.concatenate(([open_base], close[:-1]))
    high = np.maximum(openp, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(openp, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(10_000, 50_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    if spikes:
        for i, overrides in spikes.items():
            for k, v in overrides.items():
                df.iat[i, df.columns.get_loc(k)] = v
    return df


class StubPriceFetcher:
    """In-memory price fetcher for offline tests."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = {k: v.copy() for k, v in data.items()}

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for t in tickers:
            frame = self._data.get(t, pd.DataFrame())
            if frame.empty:
                out[t] = frame
                continue
            out[t] = frame.loc[(frame.index >= s) & (frame.index <= e)]
        return out


@pytest.fixture
def stub_fetcher_factory():
    def _make(data: dict[str, pd.DataFrame]) -> StubPriceFetcher:
        return StubPriceFetcher(data)
    return _make
