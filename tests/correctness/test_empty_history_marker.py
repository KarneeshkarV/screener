"""An empty price reply is only "no history" when the cache does not contradict it.

yfinance and FMP sometimes answer a good request with no bars and no error.
Recorded as a "no history" marker, that answer hid a real ticker from every run
for the marker's whole TTL. A reply with none of the bars the ticker's own cache
already holds inside the window is a failed request, not a fact about the
ticker. The edge probes that reach past the cache and come back empty are still
evidence about where the vendor's history stops, and keep being recorded.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.data import FMPPriceFetcher, YFinancePriceFetcher, _save_cache
from screener.backtester.price_cache import (
    empty_history_path,
    has_empty_history,
    load_coverage,
)
from tests.test_fmp_fetcher import DummySession, _fmp_full_key
from tests.test_yfinance_cache import _full_key, _plain_bars


def _empty_yfinance(monkeypatch) -> list[object]:
    import yfinance as yf

    calls: list[object] = []

    def empty_download(tickers, **kwargs):
        calls.append(tickers)
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", empty_download)
    return calls


def test_an_empty_yfinance_reply_over_cached_bars_writes_no_marker(
    tmp_path, monkeypatch
):
    """A refresh served nothing for a window the cache holds bars in.

    The vendor served those bars before, so the next ordinary run must still
    ask for the missing sessions rather than skip the name for a day.
    """
    calls = _empty_yfinance(monkeypatch)
    cached = _plain_bars(date(2024, 1, 2), date(2024, 1, 20)).rename(columns=str.lower)
    _save_cache(_full_key("AAA"), cached, tmp_path)

    refreshing = YFinancePriceFetcher(cache_dir=tmp_path, refresh=True)
    out = refreshing.fetch(["AAA"], date(2024, 1, 2), date(2024, 1, 25))

    assert out["AAA"].index.max() == cached.index.max()
    assert not empty_history_path(_full_key("AAA"), tmp_path).exists()

    before = len(calls)
    YFinancePriceFetcher(cache_dir=tmp_path).fetch(
        ["AAA"], date(2024, 1, 2), date(2024, 1, 25)
    )
    assert len(calls) > before, "the next run must still ask the vendor"


@pytest.mark.parametrize(
    ("cached_from", "cached_to", "marker", "coverage"),
    [
        # "backfill": the cache starts late; the probe reaches back to the
        # window's start and ends on the cache's fifth bar.
        (
            date(2024, 1, 15),
            date(2024, 2, 1),
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-19")),
            (pd.Timestamp("2024-01-15"), None),
        ),
        # "extend": the cache stops early; the probe starts on its fifth-newest
        # bar and reaches forward to the window's end.
        (
            date(2024, 1, 2),
            date(2024, 1, 13),
            (pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-31")),
            (None, pd.Timestamp("2024-01-12")),
        ),
    ],
    ids=["backfill", "extend"],
)
def test_an_empty_probe_past_the_cache_edge_is_still_recorded(
    tmp_path, monkeypatch, cached_from, cached_to, marker, coverage
):
    """Both the marker and the coverage bound, exactly as before the fix."""
    _empty_yfinance(monkeypatch)
    monkeypatch.setenv("SCREENER_PRICE_TAIL_TTL_SECONDS", "999999999")
    _save_cache(
        _full_key("AAA"),
        _plain_bars(cached_from, cached_to).rename(columns=str.lower),
        tmp_path,
    )

    YFinancePriceFetcher(cache_dir=tmp_path).fetch(
        ["AAA"], date(2024, 1, 1), date(2024, 1, 31)
    )

    assert has_empty_history(_full_key("AAA"), *marker, tmp_path)
    assert load_coverage(_full_key("AAA"), tmp_path) == coverage


def test_an_empty_fmp_reply_over_cached_bars_writes_no_marker(tmp_path):
    """The FMP leg: same contradiction, same answer."""
    cached = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        },
        index=pd.bdate_range("2024-01-02", "2024-01-19"),
    )
    _save_cache(_fmp_full_key("AAA"), cached, tmp_path)
    session = DummySession({"symbol": "AAA", "historical": []})
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    out = fetcher.fetch(["AAA"], date(2024, 1, 2), date(2024, 1, 25))

    assert out["AAA"].index.max() == cached.index.max()
    assert not empty_history_path(_fmp_full_key("AAA"), tmp_path).exists()

    before = len(session.calls)
    fetcher.fetch(["AAA"], date(2024, 1, 2), date(2024, 1, 25))
    assert len(session.calls) > before, "the next run must still ask the vendor"
