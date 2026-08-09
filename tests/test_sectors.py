"""Unit tests for sector mapping and sector-neutral rank scores."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from screener import sectors
from screener.backtester.rolling_candidates import (
    _build_rolling_candidate_matrices,
    _sector_neutralize_scores,
)
from screener.sectors import UNKNOWN_SECTOR, sector_by_ticker


def test_sector_by_ticker_uses_cache_and_fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr("screener.sectors.CACHE_DIR", tmp_path)

    calls: list[str] = []

    def fake_info(symbol: str) -> dict:
        calls.append(symbol)
        if symbol == "AAPL":
            return {"sector": "Technology"}
        if symbol == "JPM":
            return {"sector": "Financial Services"}
        return {}

    first = sector_by_ticker(
        ["AAPL", "JPM", "ZZZ"],
        "us",
        use_cache=True,
        info_fetcher=fake_info,
    )
    assert first == {
        "AAPL": "Technology",
        "JPM": "Financial Services",
        "ZZZ": UNKNOWN_SECTOR,
    }
    assert set(calls) == {"AAPL", "JPM", "ZZZ"}

    # Second call hits cache — fetcher not invoked again.
    calls.clear()
    second = sector_by_ticker(
        ["AAPL", "JPM", "ZZZ"],
        "us",
        use_cache=True,
        info_fetcher=fake_info,
    )
    assert second == first
    assert calls == []


def test_sector_by_ticker_india_symbol_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr("screener.sectors.CACHE_DIR", tmp_path)

    seen: list[str] = []

    def fake_info(symbol: str) -> dict:
        seen.append(symbol)
        return {"sector": "Energy"}

    out = sector_by_ticker(
        ["NSE:RELIANCE", "RELIANCE"],
        "india",
        use_cache=False,
        info_fetcher=fake_info,
    )
    assert out["NSE:RELIANCE"] == "Energy"
    assert out["RELIANCE"] == "Energy"
    # Both map to RELIANCE.NS — only one yfinance lookup.
    assert seen == ["RELIANCE.NS"]


def test_sector_neutralize_zscores_within_sector() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    # Day 0: Tech has A=3, B=1 (mean 2, pop std 1) -> z = 1, -1
    #         Fin has C=10 alone -> z = 0
    scores = pd.DataFrame(
        {
            "A": [3.0, 5.0],
            "B": [1.0, 1.0],
            "C": [10.0, 10.0],
        },
        index=idx,
    )
    sectors = {"A": "Tech", "B": "Tech", "C": "Fin"}
    out = _sector_neutralize_scores(scores, sectors)
    assert out.loc[idx[0], "A"] == pytest.approx(1.0)
    assert out.loc[idx[0], "B"] == pytest.approx(-1.0)
    assert out.loc[idx[0], "C"] == pytest.approx(0.0)


def test_sector_neutralize_empty_matrix() -> None:
    assert _sector_neutralize_scores(pd.DataFrame(), {}).empty


def test_build_matrices_sector_neutral_warns_unknown() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    close = pd.Series(np.linspace(100, 104, 5), index=idx)
    vol = pd.Series(1_000.0, index=idx)

    def bars(score: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": vol,
                "rank_score": pd.Series(score, index=idx, dtype=float),
            }
        )

    bars_by_tv = {"AAA": bars(2.0), "BBB": bars(4.0)}
    entry = {
        "AAA": pd.Series(True, index=idx),
        "BBB": pd.Series(True, index=idx),
    }
    warnings: list[str] = []
    mats = _build_rolling_candidate_matrices(
        bars_by_tv,
        entry,
        {},
        list(idx),
        lookback_required=0,
        warnings=warnings,
        sector_neutral=True,
        sector_by_tv={"AAA": "Tech"},  # BBB missing -> UNKNOWN
    )
    assert mats.rank_score_mat is not None
    assert warnings == [
        "sector neutralization: 1 ticker(s) mapped to UNKNOWN sector: BBB"
    ]


def test_sector_neutral_noop_without_rank_score() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    close = pd.Series(np.linspace(100, 104, 5), index=idx)
    vol = pd.Series(1_000.0, index=idx)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": vol,
        }
    )
    mats = _build_rolling_candidate_matrices(
        {"AAA": frame},
        {"AAA": pd.Series(True, index=idx)},
        {},
        list(idx),
        lookback_required=0,
        sector_neutral=True,
        sector_by_tv={"AAA": "Tech"},
    )
    assert mats.rank_score_mat is None


def test_default_fetcher_and_invalid_cache(monkeypatch) -> None:
    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.info = {"sector": "Technology"} if symbol == "GOOD" else []

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=Ticker))
    assert sectors._default_info_fetcher("GOOD") == {"sector": "Technology"}
    assert sectors._default_info_fetcher("BAD") == {}

    monkeypatch.setattr(sectors, "is_fresh", lambda *args: True)
    monkeypatch.setattr(sectors, "read_json", lambda *args, **kwargs: [])
    assert sectors._load_cached_sector("BAD") is None
    monkeypatch.setattr(sectors, "read_json", lambda *args, **kwargs: {"sector": " "})
    assert sectors._load_cached_sector("BAD") is None


def test_sector_fetch_failure_blank_and_empty_input(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sectors, "CACHE_DIR", tmp_path)

    def fail(symbol: str) -> dict:
        raise RuntimeError(symbol)

    assert sector_by_ticker(["", "FAIL"], "us", use_cache=False, info_fetcher=fail) == {
        "FAIL": UNKNOWN_SECTOR
    }
    assert sector_by_ticker(
        ["BLANK"], "us", use_cache=False, info_fetcher=lambda symbol: {"sector": " "}
    ) == {"BLANK": UNKNOWN_SECTOR}


def test_sector_neutralize_single_cell_frame():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    scores = pd.DataFrame({"AAA": [1.5]}, index=idx)
    out = _sector_neutralize_scores(scores, {"AAA": "Tech"})
    # A single name in its (day, sector) group neutralizes to 0.
    assert out.shape == scores.shape
    assert float(out.iloc[0, 0]) == 0.0


def test_a_stale_unknown_is_refetched_but_a_real_sector_is_not(tmp_path, monkeypatch):
    """A provider outage must not pin a universe to one bucket for a month.

    Every failure path - rate limit, network error, missing field - lands on
    UNKNOWN, so caching it for the full 30 days turns sector-neutral ranking
    into a silent no-op long after the provider recovers. Real sectors keep the
    long TTL; UNKNOWN expires within a day.
    """
    import time

    monkeypatch.setattr(sectors, "CACHE_DIR", tmp_path)
    calls: list[str] = []

    def fetcher(symbol: str) -> dict[str, str]:
        calls.append(symbol)
        return {"sector": "Technology"}

    # Both cached two days ago: past the negative TTL, inside the positive one.
    two_days = time.time() - 2 * 24 * 60 * 60
    for symbol, sector in (("GOOD", "Technology"), ("RATELIMITED", "UNKNOWN")):
        path = sectors._cache_path(symbol)
        sectors.write_json(path, {"symbol": symbol, "sector": sector})
        os.utime(path, (two_days, two_days))

    assert sectors._load_cached_sector("GOOD") == "Technology"
    assert sectors._load_cached_sector("RATELIMITED") is None

    resolved = sectors.sector_by_ticker(["RATELIMITED"], "us", info_fetcher=fetcher)
    assert resolved == {"RATELIMITED": "Technology"}
    assert calls == ["RATELIMITED"]


def test_a_fresh_unknown_is_not_refetched(tmp_path, monkeypatch):
    """Within the negative TTL a sweep must not re-request the same dud symbol."""
    monkeypatch.setattr(sectors, "CACHE_DIR", tmp_path)
    sectors.write_json(
        sectors._cache_path("NOSECTOR"), {"symbol": "NOSECTOR", "sector": "UNKNOWN"}
    )
    assert sectors._load_cached_sector("NOSECTOR") == "UNKNOWN"


def test_fmp_batches_and_reports_only_what_it_answered(monkeypatch):
    """An absent symbol is unresolved, not UNKNOWN - the caller must retry it."""
    from screener import fmp

    seen: list[str] = []

    class FakeClient:
        def __init__(self, *a, **k) -> None:
            pass

        def get(self, path: str):
            seen.append(path)
            symbols = path.removeprefix("profile/").split(",")
            return [
                {"symbol": s, "sector": "Technology"} for s in symbols if s != "GHOST"
            ]

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(fmp, "FmpClient", FakeClient)
    monkeypatch.setattr(sectors, "FMP_PROFILE_BATCH", 2)

    out = sectors.fetch_fmp_sectors(["AAPL", "MSFT", "GHOST", "AAPL"])
    assert out == {"AAPL": "Technology", "MSFT": "Technology"}
    # Deduped to three symbols, batched two at a time.
    assert seen == ["profile/AAPL,MSFT", "profile/GHOST"]


def test_fmp_is_skipped_without_a_key(monkeypatch):
    from screener import fmp

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: None)
    assert sectors.fetch_fmp_sectors(["AAPL"]) == {}


def test_a_failed_fmp_batch_falls_through_to_the_per_symbol_source(monkeypatch):
    from screener import fmp

    class Boom:
        def __init__(self, *a, **k) -> None:
            pass

        def get(self, path: str):
            raise RuntimeError("502")

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(fmp, "FmpClient", Boom)
    assert sectors.fetch_fmp_sectors(["AAPL"]) == {}


def test_sector_by_ticker_prefers_fmp_over_the_per_symbol_fetcher(
    tmp_path, monkeypatch
):
    """FMP answers in one batch; yfinance only sees what FMP could not."""
    monkeypatch.setattr(sectors, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        sectors, "fetch_fmp_sectors", lambda syms: {"AAPL": "Technology"}
    )
    fallback_calls: list[str] = []

    def fallback(symbol: str) -> dict[str, str]:
        fallback_calls.append(symbol)
        return {"sector": "Energy"}

    monkeypatch.setattr(sectors, "_default_info_fetcher", fallback)
    out = sector_by_ticker(["AAPL", "XOM"], "us")
    assert out == {"AAPL": "Technology", "XOM": "Energy"}
    assert fallback_calls == ["XOM"]
    # The FMP answer is cached like any other, so the next call skips both.
    assert sectors._load_cached_sector("AAPL") == "Technology"
