"""Offline correctness tests for the data-layer transforms.

Covers:
- tv_to_yf symbol mapping
- _normalize_frame: split_factor derivation, NaN-OHLCV drop, dedupe-by-date
  keep-last, tz-naive index, column renaming, dividend handling
- _naive_normalized_index: tz-stripping and midnight normalisation
- _load_cached: re-drops NaN-OHLCV rows on cache hit
- NSE fetch.py: _parse_bhavcopy_date dayfirst, fetch_cash_bhavcopy EQ filter,
  fetch_fo_bhavcopy STF filter, near_month_oi aggregation

No network calls are made anywhere in this file.
"""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from screener.backtester.data import (
    OHLCV_COLUMNS,
    _empty_ohlcv_frame,
    _load_cached,
    _naive_normalized_index,
    _normalize_frame,
    tv_to_yf,
)
from screener.operator.fetch import (
    _parse_bhavcopy_date,
    near_month_oi,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_raw_df(
    dates: list[str],
    ohlcv: list[tuple],
    splits: list[float] | None = None,
    dividends: list[float] | None = None,
) -> pd.DataFrame:
    """Build a raw (pre-normalise) DataFrame mimicking yfinance output."""
    idx = pd.DatetimeIndex(dates)
    opens, highs, lows, closes, vols = zip(*ohlcv)
    data: dict = {
        "Open": list(opens),
        "High": list(highs),
        "Low": list(lows),
        "Close": list(closes),
        "Volume": list(vols),
    }
    if splits is not None:
        data["Stock Splits"] = splits
    if dividends is not None:
        data["Dividends"] = dividends
    return pd.DataFrame(data, index=idx)


# ─── tv_to_yf ───────────────────────────────────────────────────────────────


class TestTvToYf:
    """Test the TradingView → yfinance symbol translation table."""

    def test_nse_prefix_india(self):
        assert tv_to_yf("NSE:RELIANCE", "india") == "RELIANCE.NS"

    def test_bse_prefix_india(self):
        assert tv_to_yf("BSE:TCS", "india") == "TCS.BO"

    def test_nasdaq_prefix_us(self):
        assert tv_to_yf("NASDAQ:AAPL", "us") == "AAPL"

    def test_bare_symbol_us(self):
        # No ":" and no ".", market=us → returned as-is (uppercased).
        assert tv_to_yf("AAPL", "us") == "AAPL"

    def test_bare_symbol_india_gets_ns_suffix(self):
        # No ":" and no "." with india market → appends .NS
        assert tv_to_yf("RELIANCE", "india") == "RELIANCE.NS"

    def test_already_ns_suffix_unchanged(self):
        # Has a "." → market suffix branch skipped → returned uppercased
        assert tv_to_yf("RELIANCE.NS", "india") == "RELIANCE.NS"

    def test_strip_and_upper(self):
        # Leading/trailing whitespace + lowercase → normalised
        assert tv_to_yf(" aapl ", "us") == "AAPL"

    def test_unknown_exchange_strips_prefix(self):
        # Exchange not NSE/BSE → just returns the part after ":"
        assert tv_to_yf("NASDAQ:TSLA", "india") == "TSLA"

    def test_bse_prefix_us_market(self):
        # BSE is always .BO regardless of market parameter
        assert tv_to_yf("BSE:WIPRO", "us") == "WIPRO.BO"

    def test_nse_prefix_us_market(self):
        # NSE is always .NS regardless of market parameter
        assert tv_to_yf("NSE:INFY", "us") == "INFY.NS"

    def test_bare_with_dot_india_unchanged(self):
        # Has a "." already (e.g. "RELIANCE.BO") → market branch not entered
        assert tv_to_yf("RELIANCE.BO", "india") == "RELIANCE.BO"


# ─── _normalize_frame: column rename / keep ──────────────────────────────────


class TestNormalizeFrameColumns:
    """Column rename, keep subset, and empty-input handling."""

    def test_lowercase_rename(self):
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03"],
            [(100, 102, 99, 101, 1000), (101, 103, 100, 102, 1100)],
        )
        out = _normalize_frame(df)
        assert list(out.columns) == OHLCV_COLUMNS

    def test_extra_columns_dropped(self):
        df = _make_raw_df(
            ["2024-01-02"],
            [(100, 102, 99, 101, 1000)],
        )
        df["SomeExtra"] = 99
        out = _normalize_frame(df)
        assert "someextra" not in out.columns
        assert set(out.columns).issubset(
            set(OHLCV_COLUMNS)
            | {"adj_close", "dividend", "split_factor", "stock_splits"}
        )

    def test_none_returns_empty_frame(self):
        out = _normalize_frame(None)
        assert out.empty
        assert list(out.columns) == OHLCV_COLUMNS

    def test_empty_df_returns_empty_frame(self):
        out = _normalize_frame(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == OHLCV_COLUMNS

    def test_multiindex_columns_droplevel(self):
        """yfinance multi-ticker download returns MultiIndex; normalize drops level."""
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        mi = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(
            [[100, 102, 99, 101, 1000], [101, 103, 100, 102, 1100]],
            index=dates,
            columns=mi,
        )
        out = _normalize_frame(df)
        assert not isinstance(out.columns, pd.MultiIndex)
        assert "open" in out.columns


# ─── _normalize_frame: split_factor derivation ───────────────────────────────


class TestNormalizeFrameSplitFactor:
    """Verify the reverse-cumulative split_factor column logic.

    The formula in _normalize_frame (line ~219):
        factor = splits.replace(0,1)[::-1].cumprod()[::-1].shift(-1).fillna(1)

    This gives the *back-adjustment multiplier* that would need to be applied
    to historical bars to make them split-adjusted relative to the present.
    Bars BEFORE a 2:1 split get factor 2; the bar ON which the split occurs
    and bars after get factor 1 (they are already in post-split units).
    """

    def _make_split_df(self, splits: list[float]) -> pd.DataFrame:
        n = len(splits)
        dates = pd.bdate_range("2024-01-02", periods=n)
        return pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [102.0] * n,
                "Low": [99.0] * n,
                "Close": [101.0] * n,
                "Volume": [1000.0] * n,
                "Stock Splits": splits,
            },
            index=dates,
        )

    def test_no_splits_factor_all_ones(self):
        df = self._make_split_df([0.0, 0.0, 0.0, 0.0, 0.0])
        out = _normalize_frame(df)
        assert "split_factor" in out.columns
        np.testing.assert_array_equal(
            out["split_factor"].values, [1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_single_split_2_at_index_2(self):
        """2:1 split on bar 2 → bars 0 and 1 get factor 2, bars 2,3,4 get factor 1."""
        # splits[2]=2.0, rest 0
        df = self._make_split_df([0.0, 0.0, 2.0, 0.0, 0.0])
        out = _normalize_frame(df)
        expected = [2.0, 2.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(out["split_factor"].values, expected)

    def test_double_split_compounds(self):
        """Two splits: 2:1 at bar 1, 3:1 at bar 3.

        Working (reversed cumprod then shift(-1)):
          splits       = [0, 2, 0, 3, 0]
          replace 0→1  = [1, 2, 1, 3, 1]
          reversed     = [1, 3, 1, 2, 1]
          cumprod      = [1, 3, 3, 6, 6]
          un-reverse   = [6, 6, 3, 3, 1]
          shift(-1)    = [6, 3, 3, 1, NaN]
          fillna(1)    = [6, 3, 3, 1, 1]
        """
        df = self._make_split_df([0.0, 2.0, 0.0, 3.0, 0.0])
        out = _normalize_frame(df)
        expected = [6.0, 3.0, 3.0, 1.0, 1.0]
        np.testing.assert_array_equal(out["split_factor"].values, expected)

    def test_split_at_first_bar(self):
        """Split on bar 0 → shift(-1) moves it, bar 0 itself gets factor 1."""
        df = self._make_split_df([2.0, 0.0, 0.0])
        out = _normalize_frame(df)
        # replace 0→1: [2,1,1]; reversed: [1,1,2]; cumprod: [1,1,2];
        # un-reverse: [2,1,1]; shift(-1): [1,1,NaN]; fillna: [1,1,1]
        expected = [1.0, 1.0, 1.0]
        np.testing.assert_array_equal(out["split_factor"].values, expected)

    def test_split_at_last_bar(self):
        """Split on last bar → the shift makes all pre-bars adjusted."""
        df = self._make_split_df([0.0, 0.0, 2.0])
        out = _normalize_frame(df)
        # replace: [1,1,2]; rev: [2,1,1]; cumprod: [2,2,2];
        # un-rev: [2,2,2]; shift(-1): [2,2,NaN]; fillna: [2,2,1]
        expected = [2.0, 2.0, 1.0]
        np.testing.assert_array_equal(out["split_factor"].values, expected)

    def test_stock_splits_column_preserved(self):
        """The raw stock_splits column is also kept in the output."""
        df = self._make_split_df([0.0, 2.0, 0.0])
        out = _normalize_frame(df)
        assert "stock_splits" in out.columns
        np.testing.assert_array_equal(out["stock_splits"].values, [0.0, 2.0, 0.0])

    def test_no_stock_splits_column_no_split_factor(self):
        """Without a Stock Splits column, split_factor is absent."""
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03"],
            [(100, 102, 99, 101, 1000), (101, 103, 100, 102, 1100)],
        )
        out = _normalize_frame(df)
        assert "split_factor" not in out.columns

    def test_normalize_does_not_apply_split_to_ohlc(self):
        """_normalize_frame stores split_factor but does NOT back-adjust OHLC.

        The close values in the output should be exactly the input close values
        — adjustment happens elsewhere (auto_adjust in yfinance or downstream).
        """
        df = self._make_split_df([0.0, 0.0, 2.0, 0.0, 0.0])
        out = _normalize_frame(df)
        np.testing.assert_array_equal(out["close"].values, [101.0] * 5)


# ─── _normalize_frame: dividend handling ────────────────────────────────────


class TestNormalizeFrameDividend:
    def test_dividends_column_renamed_fillna(self):
        """'Dividends' (yfinance spelling) → 'dividend', NaN → 0."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [103, 104, 105],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000, 1000, 1000],
                "Dividends": [0.0, float("nan"), 1.5],
            },
            index=dates,
        )
        out = _normalize_frame(df)
        assert "dividend" in out.columns
        assert out["dividend"].iloc[1] == 0.0  # NaN was filled
        assert out["dividend"].iloc[2] == 1.5

    def test_dividend_column_lowercase_preserved(self):
        """If input already has lowercase 'dividend', it is still kept."""
        dates = pd.bdate_range("2024-01-02", periods=2)
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [103, 104],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1000],
                "dividend": [0.5, float("nan")],
            },
            index=dates,
        )
        out = _normalize_frame(df)
        assert "dividend" in out.columns
        assert out["dividend"].iloc[1] == 0.0


# ─── _normalize_frame: NaN-OHLCV drop ───────────────────────────────────────


class TestNormalizeFrameNanDrop:
    """Bars where any OHLCV column is NaN must be dropped."""

    def test_nan_close_dropped(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        df = pd.DataFrame(
            {
                "Open": [100, 101, float("nan"), 103],
                "High": [102, 103, float("nan"), 105],
                "Low": [99, 100, float("nan"), 102],
                "Close": [101, 102, float("nan"), 104],
                "Volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        out = _normalize_frame(df)
        assert len(out) == 3
        assert pd.Timestamp("2024-01-04") not in out.index

    def test_nan_volume_dropped(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000, float("nan"), 1000],
            },
            index=dates,
        )
        out = _normalize_frame(df)
        assert len(out) == 2

    def test_all_valid_nothing_dropped(self):
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            [(100, 102, 99, 101, 1000)] * 3,
        )
        out = _normalize_frame(df)
        assert len(out) == 3


# ─── _normalize_frame: dedupe by date, keep last ────────────────────────────


class TestNormalizeFrameDedupe:
    """Duplicate index dates: last row wins."""

    def test_duplicate_date_keeps_last(self):
        # Two rows for 2024-01-02; second row has close=999 — that must survive
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame(
            {
                "Open": [100, 200, 105],
                "High": [102, 202, 107],
                "Low": [99, 199, 104],
                "Close": [101, 999, 106],
                "Volume": [1000, 2000, 1100],
            },
            index=idx,
        )
        out = _normalize_frame(df)
        assert len(out) == 2
        assert out.loc[pd.Timestamp("2024-01-02"), "close"] == 999.0

    def test_no_duplicates_unchanged_length(self):
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            [(100, 102, 99, 101, 1000)] * 3,
        )
        out = _normalize_frame(df)
        assert len(out) == 3


# ─── _normalize_frame: tz-naive index ───────────────────────────────────────


class TestNormalizeFrameIndex:
    """Output index must always be tz-naive."""

    def test_tz_aware_index_stripped(self):
        tz_idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("UTC")
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1000],
            },
            index=tz_idx,
        )
        out = _normalize_frame(df)
        assert out.index.tz is None

    def test_naive_index_unchanged(self):
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03"],
            [(100, 102, 99, 101, 1000)] * 2,
        )
        out = _normalize_frame(df)
        assert out.index.tz is None

    def test_index_is_datetime_index(self):
        df = _make_raw_df(
            ["2024-01-02", "2024-01-03"],
            [(100, 102, 99, 101, 1000)] * 2,
        )
        out = _normalize_frame(df)
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_index_normalised_to_midnight(self):
        """Even non-midnight timestamps are rounded down to 00:00."""
        idx = pd.DatetimeIndex(["2024-01-02T10:30:00", "2024-01-03T15:00:00"])
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1000],
            },
            index=idx,
        )
        out = _normalize_frame(df)
        for ts in out.index:
            assert ts.hour == 0 and ts.minute == 0 and ts.second == 0


# ─── _naive_normalized_index ────────────────────────────────────────────────


class TestNaiveNormalizedIndex:
    def test_tz_aware_stripped(self):
        idx = pd.DatetimeIndex(["2024-01-02"]).tz_localize("America/New_York")
        result = _naive_normalized_index(idx)
        assert result.tz is None

    def test_already_naive_returned_as_is_type(self):
        idx = pd.DatetimeIndex(["2024-01-02"])
        result = _naive_normalized_index(idx)
        assert isinstance(result, pd.DatetimeIndex)
        assert result.tz is None

    def test_string_index_converted(self):
        idx = pd.Index(["2024-01-02", "2024-01-03"])
        result = _naive_normalized_index(idx)
        assert isinstance(result, pd.DatetimeIndex)

    def test_midnight_preserved(self):
        idx = pd.DatetimeIndex(["2024-01-02T14:30:00"])
        result = _naive_normalized_index(idx)
        assert result[0].hour == 0


# ─── _empty_ohlcv_frame ─────────────────────────────────────────────────────


class TestEmptyOhlcvFrame:
    def test_columns(self):
        f = _empty_ohlcv_frame()
        assert list(f.columns) == OHLCV_COLUMNS

    def test_empty(self):
        f = _empty_ohlcv_frame()
        assert f.empty

    def test_datetimeindex(self):
        f = _empty_ohlcv_frame()
        assert isinstance(f.index, pd.DatetimeIndex)


# ─── _load_cached: re-drops NaN rows ────────────────────────────────────────


class TestLoadCached:
    """_load_cached must drop NaN-OHLCV rows that might have been written by
    an older cache version, ensuring a cache hit never re-introduces bad bars."""

    def test_nan_rows_dropped_on_cache_hit(self, tmp_path):
        """Write a parquet with a NaN close row; _load_cached must drop it."""
        ticker = "TEST"
        dates = pd.bdate_range("2024-01-02", periods=3)
        df = pd.DataFrame(
            {
                "open": [100.0, float("nan"), 102.0],
                "high": [102.0, float("nan"), 104.0],
                "low": [99.0, float("nan"), 101.0],
                "close": [101.0, float("nan"), 103.0],
                "volume": [1000.0, float("nan"), 1000.0],
            },
            index=dates,
        )
        path = tmp_path / f"{ticker}.parquet"
        df.to_parquet(path)

        result = _load_cached(ticker, cache_dir=tmp_path)
        assert result is not None
        assert len(result) == 2
        assert not result["close"].isna().any()

    def test_missing_file_returns_none(self, tmp_path):
        result = _load_cached("NONEXISTENT", cache_dir=tmp_path)
        assert result is None

    def test_index_is_tz_naive(self, tmp_path):
        ticker = "TEST2"
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )
        path = tmp_path / f"{ticker}.parquet"
        df.to_parquet(path)

        result = _load_cached(ticker, cache_dir=tmp_path)
        assert result is not None
        assert result.index.tz is None


# ─── NSE: _parse_bhavcopy_date ───────────────────────────────────────────────


class TestParseBhavcopDate:
    """_parse_bhavcopy_date uses dayfirst=True so '01-Mar-2024' → 2024-03-01."""

    def _df_with_date1(self, raw: str) -> pd.DataFrame:
        return pd.DataFrame({"DATE1": [raw], "SYMBOL": ["RELIANCE"]})

    def test_ddmmmyyyy_dayfirst(self):
        # NSE real format: "01-Mar-2024" — day first, not month first
        df = self._df_with_date1("01-Mar-2024")
        result = _parse_bhavcopy_date(df)
        assert result == date(2024, 3, 1)

    def test_ddmmmyyyy_no_hyphen(self):
        df = self._df_with_date1("15JAN2024")
        result = _parse_bhavcopy_date(df)
        assert result == date(2024, 1, 15)

    def test_iso_format(self):
        df = self._df_with_date1("2024-03-15")
        result = _parse_bhavcopy_date(df)
        assert result == date(2024, 3, 15)

    def test_missing_date1_returns_none(self):
        df = pd.DataFrame({"SYMBOL": ["RELIANCE"]})
        result = _parse_bhavcopy_date(df)
        assert result is None

    def test_empty_df_returns_none(self):
        df = pd.DataFrame({"DATE1": []})
        result = _parse_bhavcopy_date(df)
        assert result is None

    def test_unparseable_returns_none(self):
        df = self._df_with_date1("NOT_A_DATE")
        result = _parse_bhavcopy_date(df)
        assert result is None


# ─── NSE: fetch_cash_bhavcopy EQ filter ─────────────────────────────────────


class TestFetchCashBhavcopy:
    """fetch_cash_bhavcopy should keep only SERIES=='EQ' rows."""

    RAW_CSV = (
        "SYMBOL,SERIES,PREV_CLOSE,CLOSE_PRICE,AVG_PRICE,TTL_TRD_QNTY,DELIV_QTY,DELIV_PER,DATE1\n"
        "RELIANCE,EQ,2900.00,2950.00,2925.00,1000000,600000,60.0,01-Mar-2024\n"
        "RELIANCE,BE,2900.00,2950.00,2925.00,50000,50000,100.0,01-Mar-2024\n"
        "TCS,EQ,3500.00,3550.00,3525.00,800000,400000,50.0,01-Mar-2024\n"
        "WIPRO,N1,200.00,205.00,202.50,200000,100000,50.0,01-Mar-2024\n"
    )

    def _mock_read(self, d):
        """Return a parsed DataFrame as _read_cash_bhavcopy_raw would."""
        df = pd.read_csv(io.StringIO(self.RAW_CSV))
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
        return df

    def test_only_eq_rows_returned(self):
        from screener.operator.fetch import fetch_cash_bhavcopy

        with patch("screener.operator.fetch._read_cash_bhavcopy_raw", self._mock_read):
            result = fetch_cash_bhavcopy(date(2024, 3, 1))

        symbols = result["SYMBOL"].tolist()
        assert "RELIANCE" in symbols
        assert "TCS" in symbols
        assert len(result) == 2  # BE and N1 rows excluded

    def test_numeric_columns_coerced(self):
        from screener.operator.fetch import fetch_cash_bhavcopy

        with patch("screener.operator.fetch._read_cash_bhavcopy_raw", self._mock_read):
            result = fetch_cash_bhavcopy(date(2024, 3, 1))

        assert result["CLOSE_PRICE"].dtype in (float, np.float64)
        assert result["DELIV_PER"].dtype in (float, np.float64)

    def test_keep_columns_present(self):
        from screener.operator.fetch import fetch_cash_bhavcopy

        with patch("screener.operator.fetch._read_cash_bhavcopy_raw", self._mock_read):
            result = fetch_cash_bhavcopy(date(2024, 3, 1))

        expected_cols = [
            "SYMBOL",
            "PREV_CLOSE",
            "CLOSE_PRICE",
            "AVG_PRICE",
            "TTL_TRD_QNTY",
            "DELIV_QTY",
            "DELIV_PER",
        ]
        for col in expected_cols:
            assert col in result.columns


# ─── NSE: fetch_fo_bhavcopy STF filter ──────────────────────────────────────


class TestFetchFoBhavcopy:
    """fetch_fo_bhavcopy should keep only FinInstrmTp=='STF' rows (stock futures)."""

    RAW_CSV = (
        "TckrSymb,FinInstrmTp,XpryDt,OpnIntrst,SttlmPric\n"
        "RELIANCE,STF,2024-03-28,50000,2950.00\n"
        "RELIANCE,STF,2024-04-25,20000,2955.00\n"
        "RELIANCE,OPT,2024-03-28,30000,2900.00\n"
        "TCS,STF,2024-03-28,40000,3550.00\n"
        "NIFTY,IDF,2024-03-28,100000,22000.00\n"
    )

    def _patch_fo(self, tmp_path):
        """Write the raw CSV to tmp_path so _fo_cache_path finds it."""
        d = date(2024, 3, 1)
        day_dir = tmp_path / d.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        fname = f"BhavCopy_NSE_FO_{d.strftime('%Y%m%d')}.csv"
        (day_dir / fname).write_text(self.RAW_CSV)
        return d, day_dir

    def test_only_stf_rows_returned(self, tmp_path):
        from screener.operator import fetch as fetch_mod

        d, _ = self._patch_fo(tmp_path)
        orig_cache_root = fetch_mod.CACHE_ROOT
        fetch_mod.CACHE_ROOT = tmp_path
        try:
            result = fetch_mod.fetch_fo_bhavcopy(d)
        finally:
            fetch_mod.CACHE_ROOT = orig_cache_root

        assert set(result["SYMBOL"].unique()) == {"RELIANCE", "TCS"}
        assert len(result) == 3  # 2 RELIANCE expiries + 1 TCS

    def test_columns_renamed(self, tmp_path):
        from screener.operator import fetch as fetch_mod

        d, _ = self._patch_fo(tmp_path)
        orig_cache_root = fetch_mod.CACHE_ROOT
        fetch_mod.CACHE_ROOT = tmp_path
        try:
            result = fetch_mod.fetch_fo_bhavcopy(d)
        finally:
            fetch_mod.CACHE_ROOT = orig_cache_root

        assert "SYMBOL" in result.columns
        assert "EXPIRY" in result.columns
        assert "OI" in result.columns

    def test_expiry_is_datetime(self, tmp_path):
        from screener.operator import fetch as fetch_mod

        d, _ = self._patch_fo(tmp_path)
        orig_cache_root = fetch_mod.CACHE_ROOT
        fetch_mod.CACHE_ROOT = tmp_path
        try:
            result = fetch_mod.fetch_fo_bhavcopy(d)
        finally:
            fetch_mod.CACHE_ROOT = orig_cache_root

        assert pd.api.types.is_datetime64_any_dtype(result["EXPIRY"])


# ─── near_month_oi aggregation ───────────────────────────────────────────────


class TestNearMonthOi:
    """near_month_oi: first two expiries per symbol → Current_OI, Next_OI,
    Cumulative_OI."""

    def _make_fo_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["EXPIRY"] = pd.to_datetime(df["EXPIRY"])
        return df

    def test_two_expiries_populated(self):
        fo = self._make_fo_df(
            [
                {"SYMBOL": "RELIANCE", "EXPIRY": "2024-03-28", "OI": 50000},
                {"SYMBOL": "RELIANCE", "EXPIRY": "2024-04-25", "OI": 20000},
            ]
        )
        result = near_month_oi(fo)
        row = result[result["SYMBOL"] == "RELIANCE"].iloc[0]
        assert row["Current_OI"] == 50000
        assert row["Next_OI"] == 20000
        assert row["Cumulative_OI"] == 70000

    def test_one_expiry_next_oi_is_nan(self):
        fo = self._make_fo_df(
            [
                {"SYMBOL": "TCS", "EXPIRY": "2024-03-28", "OI": 40000},
            ]
        )
        result = near_month_oi(fo)
        row = result[result["SYMBOL"] == "TCS"].iloc[0]
        assert row["Current_OI"] == 40000
        assert pd.isna(row["Next_OI"])
        assert row["Cumulative_OI"] == 40000  # NaN treated as 0

    def test_three_expiries_only_first_two_used(self):
        """Third expiry is ignored; only current and next."""
        fo = self._make_fo_df(
            [
                {"SYMBOL": "INFY", "EXPIRY": "2024-03-28", "OI": 10000},
                {"SYMBOL": "INFY", "EXPIRY": "2024-04-25", "OI": 8000},
                {"SYMBOL": "INFY", "EXPIRY": "2024-05-30", "OI": 5000},
            ]
        )
        result = near_month_oi(fo)
        row = result[result["SYMBOL"] == "INFY"].iloc[0]
        assert row["Current_OI"] == 10000
        assert row["Next_OI"] == 8000
        assert row["Cumulative_OI"] == 18000

    def test_multiple_symbols(self):
        fo = self._make_fo_df(
            [
                {"SYMBOL": "A", "EXPIRY": "2024-03-28", "OI": 100},
                {"SYMBOL": "A", "EXPIRY": "2024-04-25", "OI": 50},
                {"SYMBOL": "B", "EXPIRY": "2024-03-28", "OI": 200},
            ]
        )
        result = near_month_oi(fo)
        assert len(result) == 2
        a = result[result["SYMBOL"] == "A"].iloc[0]
        b = result[result["SYMBOL"] == "B"].iloc[0]
        assert a["Cumulative_OI"] == 150
        assert b["Cumulative_OI"] == 200

    def test_sorted_by_expiry_not_insertion_order(self):
        """Expiries are sorted ascending, so near-month is always smallest."""
        fo = self._make_fo_df(
            [
                {"SYMBOL": "X", "EXPIRY": "2024-04-25", "OI": 999},  # far month first
                {"SYMBOL": "X", "EXPIRY": "2024-03-28", "OI": 100},  # near month second
            ]
        )
        result = near_month_oi(fo)
        row = result[result["SYMBOL"] == "X"].iloc[0]
        assert row["Current_OI"] == 100  # near month
        assert row["Next_OI"] == 999  # far month
