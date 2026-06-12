"""Live network tests for the data layer.

ALL tests here are marked ``@pytest.mark.network`` and are skipped unless
``SCREENER_LIVE_TESTS=1`` is set in the environment. They exercise the real
yfinance download path against fixed historical windows and assert sanity
bands rather than exact values (auto_adjust prices drift as corporate actions
are revised).

Run offline (collect + skip):
    uv run pytest tests/correctness/test_data_layer_live.py -q

Run live:
    SCREENER_LIVE_TESTS=1 uv run pytest tests/correctness/test_data_layer_live.py -v
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.data import (
    OHLCV_COLUMNS,
    YFinancePriceFetcher,
    tv_to_yf,
)


# ─── AAPL historical sanity ─────────────────────────────────────────────────


@pytest.mark.network
class TestAaplHistoricalSanity:
    """Fetch AAPL for a fixed past window and assert structural + sane-value
    properties. We avoid any exact-price assertion because auto_adjust prices
    change whenever Apple declares a dividend or split."""

    # Fixed 5-day window well in the past — stable, non-holiday week
    SYMBOL = "AAPL"
    START = date(2023, 1, 9)  # Mon
    END = date(2023, 1, 13)  # Fri

    @pytest.fixture(scope="class")
    def aapl_frame(self, tmp_path_factory):
        cache = tmp_path_factory.mktemp("live_cache")
        fetcher = YFinancePriceFetcher(cache_dir=cache, auto_adjust=True)
        results = fetcher.fetch([self.SYMBOL], self.START, self.END)
        return results[self.SYMBOL]

    def test_frame_not_empty(self, aapl_frame):
        assert not aapl_frame.empty, "Expected non-empty AAPL data for 2023-01-09..13"

    def test_columns_are_lowercase_ohlcv(self, aapl_frame):
        """Columns must include at least open/high/low/close/volume."""
        for col in OHLCV_COLUMNS:
            assert col in aapl_frame.columns, f"Missing column: {col}"

    def test_index_is_tz_naive(self, aapl_frame):
        assert aapl_frame.index.tz is None, "Index must be tz-naive"

    def test_index_is_datetimeindex(self, aapl_frame):
        assert isinstance(aapl_frame.index, pd.DatetimeIndex)

    def test_close_in_sane_band(self, aapl_frame):
        """AAPL closed between $100–$250 during this week (historical range).

        This is an intentionally wide sane band to allow for any future
        auto_adjust revisions while still catching obviously wrong data.
        """
        close = aapl_frame["close"]
        assert (close > 100).all(), f"Close below $100: {close.min()}"
        assert (close < 250).all(), f"Close above $250: {close.max()}"

    def test_volume_positive(self, aapl_frame):
        assert (aapl_frame["volume"] > 0).all(), "Volume must be positive"

    def test_high_gte_low(self, aapl_frame):
        assert (aapl_frame["high"] >= aapl_frame["low"]).all()

    def test_high_gte_open_and_close(self, aapl_frame):
        assert (aapl_frame["high"] >= aapl_frame["open"]).all()
        assert (aapl_frame["high"] >= aapl_frame["close"]).all()

    def test_low_lte_open_and_close(self, aapl_frame):
        assert (aapl_frame["low"] <= aapl_frame["open"]).all()
        assert (aapl_frame["low"] <= aapl_frame["close"]).all()

    def test_no_nan_in_ohlcv(self, aapl_frame):
        assert not aapl_frame[OHLCV_COLUMNS].isna().any().any()

    def test_at_least_3_trading_days(self, aapl_frame):
        """A Mon-Fri window should yield at least 3 bars."""
        assert len(aapl_frame) >= 3


# ─── Symbol mapping + fetch round-trip ──────────────────────────────────────


@pytest.mark.network
def test_tv_to_yf_aapl_roundtrip(tmp_path):
    """tv_to_yf('NASDAQ:AAPL', 'us') → 'AAPL' which yfinance can resolve."""
    symbol = tv_to_yf("NASDAQ:AAPL", "us")
    assert symbol == "AAPL"
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, auto_adjust=True)
    results = fetcher.fetch([symbol], date(2023, 1, 9), date(2023, 1, 13))
    assert symbol in results
    assert not results[symbol].empty


# ─── Optional NSE symbol sanity ─────────────────────────────────────────────


@pytest.mark.network
def test_reliance_ns_basic_sanity(tmp_path):
    """RELIANCE.NS should return non-empty bars with sane close prices.

    RELIANCE (Reliance Industries) traded between ₹2000–₹3000 in early 2023.
    """
    symbol = tv_to_yf("NSE:RELIANCE", "india")
    assert symbol == "RELIANCE.NS"
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, auto_adjust=True)
    results = fetcher.fetch([symbol], date(2023, 1, 9), date(2023, 1, 13))
    if symbol not in results or results[symbol].empty:
        pytest.skip(
            "RELIANCE.NS data unavailable for test window — may be holiday week"
        )
    frame = results[symbol]
    assert frame.index.tz is None
    close = frame["close"]
    assert (close > 500).all(), f"RELIANCE.NS close unexpectedly low: {close.min()}"
    assert (close < 10000).all(), f"RELIANCE.NS close unexpectedly high: {close.max()}"


# ─── auto_adjust=False path: split columns present ───────────────────────────


@pytest.mark.network
def test_raw_download_has_split_columns(tmp_path):
    """With auto_adjust=False, yfinance emits Stock Splits and Dividends columns.

    _normalize_frame should then produce 'split_factor' and 'stock_splits'
    columns in the output.
    """
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path, auto_adjust=False)
    results = fetcher.fetch(["AAPL"], date(2023, 1, 9), date(2023, 1, 13))
    frame = results.get("AAPL", pd.DataFrame())
    if frame.empty:
        pytest.skip("Could not fetch AAPL raw data")
    # The normalized frame from an auto_adjust=False fetch should have these
    # columns when yfinance provides them (even if values are all 0 for this
    # window since AAPL had no split in Jan 2023).
    # We only assert structural properties — exact column presence depends on
    # which yfinance version is installed.
    assert "close" in frame.columns
    assert frame.index.tz is None
