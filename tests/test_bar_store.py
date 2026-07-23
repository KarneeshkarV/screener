"""Interval-partitioned bar store and local 1m→coarser resampling.

Covers the Phase 1 minute-bar storage seams:

* ``bar_store`` path layout (``{root}/{market}/{interval}/{symbol}.parquet``),
  atomic save/load with naive-UTC normalization, and idempotent append-merge;
* ``price_frames.resample_intraday_bars`` session-anchored bucketing (US 1h
  lands on 09:30 ET, India 30m on 09:15 IST, half-days produce fewer buckets);
* ``YFinancePriceFetcher`` serving 5m/15m/1h from the stored 1m series without
  a download, and persisting native-interval downloads into the store;
* the ``bars record`` command appending a trailing 1m window for a universe.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.bar_store import (
    append_bars,
    bar_path,
    load_bars,
    save_bars,
)
from screener.backtester.data import YFinancePriceFetcher
from screener.backtester.price_frames import resample_intraday_bars
from screener.cli import cli
from tests.conftest import StubPriceFetcher


# --------------------------------------------------------------------------- #
# Synthetic 1m bars
# --------------------------------------------------------------------------- #
def _us_1m_index(sessions: int = 1, bars_per: int = 390) -> pd.DatetimeIndex:
    """Naive-UTC 1m stamps for US regular sessions (14:30 UTC = 09:30 ET)."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-07-20 14:30:00")  # a Monday
    for _ in range(sessions):
        stamps.extend(day + pd.Timedelta(minutes=b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _india_1m_index(sessions: int = 1, bars_per: int = 375) -> pd.DatetimeIndex:
    """Naive-UTC 1m stamps for India sessions (03:45 UTC = 09:15 IST)."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-07-20 03:45:00")
    for _ in range(sessions):
        stamps.extend(day + pd.Timedelta(minutes=b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _ohlcv(index: pd.DatetimeIndex, start_px: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = pd.Series(start_px + np.arange(n, dtype=float) * 0.01, index=index)
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": np.full(n, 1_000.0),
        },
        index=index,
    )


# --------------------------------------------------------------------------- #
# bar_store primitives
# --------------------------------------------------------------------------- #
def test_bar_path_layout_and_sanitization(tmp_path):
    path = bar_path("AAPL", market="us", interval="1m", root=tmp_path)
    assert path == tmp_path / "us" / "1m" / "AAPL.parquet"

    raw = bar_path("AAPL", market="us", interval="1m", raw=True, root=tmp_path)
    assert raw.name == "AAPL__raw.parquet"

    colon = bar_path("NSE:RELIANCE", market="india", interval="5m", root=tmp_path)
    assert colon == tmp_path / "india" / "5m" / "NSE_RELIANCE.parquet"


def test_save_load_round_trip_preserves_naive_utc(tmp_path):
    index = _us_1m_index(sessions=1, bars_per=10)
    frame = _ohlcv(index)
    save_bars("AAPL", frame, market="us", interval="1m", root=tmp_path)

    loaded = load_bars("AAPL", market="us", interval="1m", root=tmp_path)
    assert loaded is not None
    assert loaded.index.tz is None
    assert loaded.index.tolist() == index.tolist()
    # No partial parquet or temp files are left behind by the atomic write.
    leftovers = [p.name for p in (tmp_path / "us" / "1m").iterdir()]
    assert leftovers == ["AAPL.parquet"]


def test_load_missing_and_corrupt_return_none(tmp_path):
    assert load_bars("NOPE", market="us", interval="1m", root=tmp_path) is None

    path = bar_path("BAD", market="us", interval="1m", root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")
    assert load_bars("BAD", market="us", interval="1m", root=tmp_path) is None


def test_append_bars_merges_dedupes_and_is_idempotent(tmp_path):
    first = _ohlcv(_us_1m_index(sessions=1, bars_per=5), start_px=100.0)
    merged = append_bars("AAPL", first, market="us", interval="1m", root=tmp_path)
    assert len(merged) == 5

    # Overlapping window (the last 2 bars) with revised prices.
    revised = _ohlcv(first.index[3:], start_px=200.0)
    merged = append_bars("AAPL", revised, market="us", interval="1m", root=tmp_path)
    assert len(merged) == 5  # no duplicates
    # Keep-last semantics: overlapping bars take the revised values.
    assert merged.loc[revised.index[-1], "close"] == pytest.approx(
        revised["close"].iloc[-1]
    )

    # Re-appending identical content is a no-op (file untouched).
    path = bar_path("AAPL", market="us", interval="1m", root=tmp_path)
    before = path.stat().st_mtime
    append_bars("AAPL", revised, market="us", interval="1m", root=tmp_path)
    assert path.stat().st_mtime == before


def test_append_bars_empty_frame_returns_existing(tmp_path):
    frame = _ohlcv(_us_1m_index(sessions=1, bars_per=3))
    append_bars("AAPL", frame, market="us", interval="1m", root=tmp_path)
    merged = append_bars(
        "AAPL", pd.DataFrame(), market="us", interval="1m", root=tmp_path
    )
    assert len(merged) == 3


def test_append_bars_empty_to_empty_store_stays_empty(tmp_path):
    merged = append_bars(
        "AAPL", pd.DataFrame(), market="us", interval="1m", root=tmp_path
    )
    assert merged.empty
    assert not bar_path("AAPL", market="us", interval="1m", root=tmp_path).exists()


# --------------------------------------------------------------------------- #
# resample_intraday_bars
# --------------------------------------------------------------------------- #
def test_resample_us_5m_and_1h_anchor_at_session_open():
    frame = _ohlcv(_us_1m_index(sessions=2, bars_per=390))

    five = resample_intraday_bars(frame, "5m", "America/New_York")
    assert len(five) == 2 * 78
    assert five.index[0] == pd.Timestamp("2026-07-20 14:30")
    assert five.index[78] == pd.Timestamp("2026-07-21 14:30")  # session 2 open

    hourly = resample_intraday_bars(frame, "1h", "America/New_York")
    assert len(hourly) == 2 * 7
    # US 1h bars land on 09:30/10:30/... ET = 14:30/15:30/... UTC, not :00.
    assert [ts.hour for ts in hourly.index[:7]] == [14, 15, 16, 17, 18, 19, 20]
    assert all(ts.minute == 30 for ts in hourly.index[:7])


def test_resample_ohlcv_aggregation():
    index = _us_1m_index(sessions=1, bars_per=10)
    frame = _ohlcv(index)
    out = resample_intraday_bars(frame, "5m", "America/New_York")
    assert len(out) == 2
    first_bucket = frame.iloc[:5]
    assert out.iloc[0]["open"] == first_bucket["open"].iloc[0]
    assert out.iloc[0]["high"] == first_bucket["high"].max()
    assert out.iloc[0]["low"] == first_bucket["low"].min()
    assert out.iloc[0]["close"] == first_bucket["close"].iloc[-1]
    assert out.iloc[0]["volume"] == first_bucket["volume"].sum()


def test_resample_india_30m_anchors_at_0915_ist():
    frame = _ohlcv(_india_1m_index(sessions=1, bars_per=375))
    out = resample_intraday_bars(frame, "30m", "Asia/Kolkata")
    # 09:15 IST open → 13 buckets (375 1m bars / 30), first at 03:45 UTC.
    assert len(out) == 13
    assert out.index[0] == pd.Timestamp("2026-07-20 03:45")
    assert out.index[1] == pd.Timestamp("2026-07-20 04:15")


def test_resample_half_day_session_produces_fewer_buckets():
    full = _us_1m_index(sessions=1, bars_per=390)
    half = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-21 14:30") + pd.Timedelta(minutes=b) for b in range(60)]
    )
    frame = _ohlcv(full.append(half))
    out = resample_intraday_bars(frame, "5m", "America/New_York")
    # Full session: 78 buckets; half-day: 12 buckets; sessions never mix.
    assert len(out) == 78 + 12
    assert out.index[78] == pd.Timestamp("2026-07-21 14:30")


def test_resample_rejects_unsupported_interval():
    frame = _ohlcv(_us_1m_index(sessions=1, bars_per=5))
    with pytest.raises(ValueError, match="cannot resample"):
        resample_intraday_bars(frame, "1m", "America/New_York")


def test_resample_empty_frame_stays_empty():
    out = resample_intraday_bars(pd.DataFrame(), "15m", "America/New_York")
    assert out.empty


# --------------------------------------------------------------------------- #
# Fetcher: one stored 1m series serves coarser intervals
# --------------------------------------------------------------------------- #
def test_fetcher_serves_15m_from_stored_1m_without_download(tmp_path, monkeypatch):
    import yfinance as yf

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("download should not be called")

    monkeypatch.setattr(yf, "download", _boom)
    index = _us_1m_index(sessions=2, bars_per=390)
    save_bars("AAPL", _ohlcv(index), market="us", interval="1m", root=tmp_path)

    fetcher = YFinancePriceFetcher(interval="15m", bars_root=tmp_path, market="us")
    out = fetcher.fetch(["AAPL"], date(2026, 7, 20), date(2026, 7, 21))["AAPL"]

    expected = resample_intraday_bars(_ohlcv(index), "15m", "America/New_York")
    assert len(out) == 2 * 26
    assert out.index.tolist() == expected.index.tolist()
    assert out["close"].tolist() == expected["close"].tolist()


def test_fetcher_1m_request_does_not_resample_and_uses_store(tmp_path, monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(
        yf,
        "download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")),
    )
    index = _us_1m_index(sessions=2, bars_per=390)
    save_bars("AAPL", _ohlcv(index), market="us", interval="1m", root=tmp_path)

    fetcher = YFinancePriceFetcher(interval="1m", bars_root=tmp_path, market="us")
    out = fetcher.fetch(["AAPL"], date(2026, 7, 20), date(2026, 7, 21))["AAPL"]
    assert out.index.tolist() == index.tolist()


def test_fetcher_downloads_native_interval_when_1m_does_not_cover(
    tmp_path, monkeypatch
):
    """No 1m coverage → native-interval download, persisted into the store."""
    import yfinance as yf

    native_index = pd.DatetimeIndex(["2026-07-20 14:30", "2026-07-20 14:45"])

    def fake_download(tickers, **kwargs):
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Volume": [100.0, 200.0],
            },
            index=native_index,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    fetcher = YFinancePriceFetcher(
        interval="15m", bars_root=tmp_path, market="us", max_workers=1
    )
    out = fetcher.fetch(["AAPL"], date(2026, 7, 20), date(2026, 7, 20))["AAPL"]

    assert out.index.tolist() == native_index.tolist()
    stored = load_bars("AAPL", market="us", interval="15m", root=tmp_path)
    assert stored is not None
    assert stored.index.tolist() == native_index.tolist()
    # The legacy flat cache is not written for intraday intervals.
    assert not (tmp_path / "AAPL__15m.parquet").exists()


# --------------------------------------------------------------------------- #
# bars record command
# --------------------------------------------------------------------------- #
def _recorded_1m_frames() -> dict[str, pd.DataFrame]:
    # Yesterday's 09:30-ET session so the bars fall inside the --days 2 window.
    base = (pd.Timestamp.now() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(
        hours=14, minutes=30
    )
    index = pd.DatetimeIndex([base + pd.Timedelta(minutes=b) for b in range(30)])
    return {
        "AAPL": _ohlcv(index, start_px=100.0),
        "MSFT": _ohlcv(index, start_px=200.0),
    }


def test_bars_record_appends_trailing_window(tmp_path, monkeypatch):
    from screener.backtester import bar_store

    monkeypatch.setattr(bar_store, "BARS_ROOT", tmp_path)
    monkeypatch.setattr(
        "screener.backtester.data.build_price_fetcher",
        lambda **kwargs: StubPriceFetcher(_recorded_1m_frames()),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["bars", "record", "-m", "us", "--tickers", "AAPL,MSFT", "--days", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "2/2 symbols" in result.output

    for symbol in ("AAPL", "MSFT"):
        stored = load_bars(symbol, market="us", interval="1m", root=tmp_path)
        assert stored is not None
        assert len(stored) == 30

    # A second run is idempotent: same bars, no duplicates.
    result = runner.invoke(
        cli, ["bars", "record", "-m", "us", "--tickers", "AAPL,MSFT", "--days", "2"]
    )
    assert result.exit_code == 0, result.output
    assert len(load_bars("AAPL", market="us", interval="1m", root=tmp_path)) == 30


def test_bars_record_rejects_empty_ticker_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["bars", "record", "-m", "us", "--tickers", " , "])
    assert result.exit_code != 0
    assert "no symbols" in result.output


def test_bars_record_uses_default_universe_and_max_symbols(tmp_path, monkeypatch):
    from pathlib import Path

    from screener.backtester import bar_store
    from screener.universes import Universe

    monkeypatch.setattr(bar_store, "BARS_ROOT", tmp_path)
    monkeypatch.setattr(
        "screener.universes.load_current_universe",
        lambda name, **kwargs: Universe(
            name=name,
            symbols=("AAPL", "MSFT", "NVDA"),
            source="test",
            cached_path=Path("unused"),
        ),
    )
    monkeypatch.setattr(
        "screener.backtester.data.build_price_fetcher",
        lambda **kwargs: StubPriceFetcher(_recorded_1m_frames()),
    )

    result = CliRunner().invoke(
        cli, ["bars", "record", "-m", "us", "--max-symbols", "2", "--days", "2"]
    )
    assert result.exit_code == 0, result.output
    # The market's default universe (sp500) feeds the recorder when neither
    # --universe nor --tickers is given; --max-symbols caps it.
    assert "(sp500)" in result.output
    assert "2 us symbols" in result.output
    assert load_bars("NVDA", market="us", interval="1m", root=tmp_path) is None


def test_bars_record_warns_on_symbols_without_bars(tmp_path, monkeypatch):
    from screener.backtester import bar_store

    monkeypatch.setattr(bar_store, "BARS_ROOT", tmp_path)
    frames = _recorded_1m_frames()
    frames.pop("MSFT")  # AAPL has bars; MSFT comes back empty from the stub.
    monkeypatch.setattr(
        "screener.backtester.data.build_price_fetcher",
        lambda **kwargs: StubPriceFetcher(frames),
    )

    result = CliRunner().invoke(
        cli, ["bars", "record", "-m", "us", "--tickers", "AAPL,MSFT", "--days", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "1/2 symbols" in result.output
    assert "no bars returned for 1 symbol(s): MSFT" in result.output
