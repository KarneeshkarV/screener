"""A split re-bases the vendor's history; a cache entry must not outlive it.

With ``auto_adjust=True`` yfinance back-propagates a split into every bar it
serves, so a cache entry written before one prices the same sessions on a
scale the vendor has abandoned. The date-keyed merge would stitch the two
together into a series with a fabricated jump at the seam, and every return
computed across that seam is wrong - a 1:125 reverse split turned a -96%
loser into a top-ranked momentum winner on a live screen.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.data import YFinancePriceFetcher, _save_cache
from screener.backtester.price_cache import has_scale_verified, scale_verified_path
from screener.backtester.price_frames import has_price_seam, overlap_rescaled


def _full_key(ticker: str) -> str:
    return YFinancePriceFetcher()._cache_key(ticker)


def _frame(start: str, periods: int, price: float) -> pd.DataFrame:
    idx = pd.bdate_range(pd.Timestamp(start), periods=periods)
    return pd.DataFrame(
        {
            "open": [price] * periods,
            "high": [price] * periods,
            "low": [price] * periods,
            "close": [price] * periods,
            "volume": [1_000] * periods,
        },
        index=idx,
    )


def _download_frame(price: float, start, end) -> pd.DataFrame:
    idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1))
    return pd.DataFrame(
        {
            "Open": [price] * len(idx),
            "High": [price] * len(idx),
            "Low": [price] * len(idx),
            "Close": [price] * len(idx),
            "Volume": [1_000] * len(idx),
        },
        index=idx,
    )


def test_overlap_rescaled_spots_a_reverse_split() -> None:
    cached = _frame("2026-01-05", 10, 2.0)
    fresh = _frame("2026-01-12", 10, 250.0)
    assert overlap_rescaled(cached, fresh)


def test_overlap_rescaled_spots_a_forward_split() -> None:
    cached = _frame("2026-01-05", 10, 250.0)
    fresh = _frame("2026-01-12", 10, 2.0)
    assert overlap_rescaled(cached, fresh)


def test_overlap_rescaled_accepts_an_unchanged_scale() -> None:
    cached = _frame("2026-01-05", 10, 10.0)
    fresh = _frame("2026-01-12", 10, 10.0)
    assert not overlap_rescaled(cached, fresh)


def test_overlap_rescaled_needs_an_overlap_to_judge() -> None:
    cached = _frame("2026-01-05", 5, 2.0)
    fresh = _frame("2026-02-02", 5, 250.0)
    # Disjoint windows say nothing about scale; a plain merge is correct.
    assert not overlap_rescaled(cached, fresh)


def test_overlap_rescaled_tolerates_vendor_rounding() -> None:
    cached = _frame("2026-01-05", 10, 10.0)
    fresh = _frame("2026-01-12", 10, 10.001)
    assert not overlap_rescaled(cached, fresh)


@pytest.mark.parametrize("ratio", [10.0, 0.1])
def test_has_price_seam_finds_a_stitched_jump(ratio: float) -> None:
    before = _frame("2026-01-05", 5, 2.0)
    after = _frame("2026-01-12", 5, 2.0 * ratio)
    assert has_price_seam(pd.concat([before, after]))


def test_has_price_seam_ignores_an_ordinary_series() -> None:
    assert not has_price_seam(_frame("2026-01-05", 20, 10.0))


def test_rescaled_cache_is_discarded_not_merged(tmp_path, monkeypatch) -> None:
    """The stale scale loses, and the ticker is re-downloaded whole."""
    import yfinance as yf

    ticker = "SPLITCO"
    # Cached on the pre-split scale, and already scale-verified so the legacy
    # seam repair is not what this test exercises.
    _save_cache(_full_key(ticker), _frame("2026-01-05", 40, 2.0), tmp_path)
    scale_verified_path(_full_key(ticker), tmp_path).write_text("{}")

    windows: list[tuple] = []

    def fake_download(tickers, **kwargs):
        windows.append((kwargs["start"], kwargs["end"]))
        # The vendor now serves everything on the post-split scale.
        return _download_frame(250.0, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(
        "screener.backtester.price_cache.needs_tail_refresh",
        lambda path, end: True,
    )
    monkeypatch.setattr(
        "screener.backtester.data._needs_tail_refresh", lambda path, end: True
    )

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    frames = fetcher.fetch([ticker], date(2026, 1, 5), date(2026, 3, 2))
    close = frames[ticker]["close"]

    # No bar is left on the old scale, so no seam survives into the result.
    assert not close.empty
    assert (close == 250.0).all()
    assert not has_price_seam(frames[ticker])
    # The retry pass asked for the whole window, not just the tail.
    assert len(windows) >= 2


def test_unchanged_scale_keeps_the_ordinary_merge(tmp_path, monkeypatch) -> None:
    """A tail refresh that agrees with the cache must not trigger a re-download."""
    import yfinance as yf

    ticker = "CALMCO"
    _save_cache(_full_key(ticker), _frame("2026-01-05", 40, 10.0), tmp_path)
    scale_verified_path(_full_key(ticker), tmp_path).write_text("{}")

    calls: list[tuple] = []

    def fake_download(tickers, **kwargs):
        calls.append((kwargs["start"], kwargs["end"]))
        return _download_frame(10.0, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(
        "screener.backtester.data._needs_tail_refresh", lambda path, end: True
    )

    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)
    frames = fetcher.fetch([ticker], date(2026, 1, 5), date(2026, 3, 2))
    assert (frames[ticker]["close"] == 10.0).all()
    # One tail request only: nothing was re-downloaded.
    assert len(calls) == 1


def test_legacy_seam_entry_is_repaired_once(tmp_path, monkeypatch) -> None:
    """An entry stitched before detection existed re-downloads exactly once."""
    import yfinance as yf

    ticker = "LEGACY"
    stitched = pd.concat(
        [_frame("2026-01-05", 20, 2.0), _frame("2026-02-02", 20, 250.0)]
    )
    _save_cache(_full_key(ticker), stitched, tmp_path)
    assert not has_scale_verified(_full_key(ticker), tmp_path)

    calls: list[tuple] = []

    def fake_download(tickers, **kwargs):
        calls.append((kwargs["start"], kwargs["end"]))
        return _download_frame(250.0, kwargs["start"], kwargs["end"])

    monkeypatch.setattr(yf, "download", fake_download)
    fetcher = YFinancePriceFetcher(cache_dir=tmp_path)

    frames = fetcher.fetch([ticker], date(2026, 1, 5), date(2026, 3, 2))
    assert (frames[ticker]["close"] == 250.0).all()
    assert has_scale_verified(_full_key(ticker), tmp_path)
    first_calls = len(calls)
    assert first_calls >= 1

    # Second run: the marker is set, so a clean cache serves with no download.
    fetcher.fetch([ticker], date(2026, 1, 5), date(2026, 3, 2))
    assert len(calls) == first_calls
