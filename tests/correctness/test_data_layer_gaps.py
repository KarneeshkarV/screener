"""Two data-layer gaps: a dividend re-base on the FMP leg, and volume-less bars."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.data import FMPPriceFetcher, _load_cached, _save_cache
from screener.backtester.price_frames import normalize_price_frame
from tests.test_fmp_fetcher import DummySession, _fmp_full_key


def test_fmp_dividend_rebase_is_not_stitched_into_the_cache(
    tmp_path, monkeypatch
) -> None:
    # A 0.5% dividend makes FMP scale every earlier adjusted bar by 0.995. That
    # is under the split tolerance, so the tail refresh used to merge the new
    # bars onto the old scale and the seam read as a one-day 0.5% return.
    cached = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000},
        index=pd.bdate_range("2023-01-02", "2023-06-30"),
    )
    _save_cache(_fmp_full_key("AAA"), cached, tmp_path)
    payload = {
        "historical": [
            {
                "date": day.date().isoformat(),
                "open": 99.5,
                "high": 99.5,
                "low": 99.5,
                "close": 99.5,
                "adjClose": 99.5,
                "volume": 1000,
            }
            for day in pd.bdate_range("2023-06-12", "2023-07-14")
        ]
    }
    monkeypatch.setattr(
        "screener.backtester.data._needs_tail_refresh", lambda path, end: True
    )
    fetcher = FMPPriceFetcher(
        api_key="test-key",
        cache_dir=tmp_path,
        session=DummySession(payload),  # type: ignore[arg-type]
    )

    out = fetcher.fetch(["AAA"], date(2023, 1, 2), date(2023, 7, 14))["AAA"]

    assert out["close"].nunique() == 1
    stored = _load_cached(_fmp_full_key("AAA"), tmp_path)
    assert stored["close"].nunique() == 1


def test_a_bar_dropped_only_for_missing_volume_is_logged(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    class _Log:
        def warning(self, event: str, **fields) -> None:
            events.append((event, fields))

    monkeypatch.setattr("screener.logging_config.get_logger", lambda _name: _Log())
    days = pd.bdate_range("2024-01-01", periods=4)
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, np.nan, 1.0],
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Volume": [10.0, np.nan, 10.0, 10.0],
        },
        index=days,
    )

    out = normalize_price_frame(raw)

    # The bar with no open is not a session; the bar with no volume is.
    assert list(out.index) == [days[0], days[3]]
    assert events == [
        (
            "price_bars_dropped_for_missing_volume",
            {"count": 1, "first": str(days[1]), "last": str(days[1])},
        )
    ]
