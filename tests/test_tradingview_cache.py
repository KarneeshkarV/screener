from __future__ import annotations

import logging

import pandas as pd

from screener import cache
from screener import scanner as scanner_module
from screener.scanner import get_scanner_data_cached


class FakeQuery:
    def __init__(self) -> None:
        self.calls = 0

    def get_scanner_data(self):
        self.calls += 1
        return 2, pd.DataFrame({"name": ["AAA", "BBB"], "volume": [10, 20]})


def test_scanner_data_cache_reuses_same_query(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = FakeQuery()

    first_count, first_df = get_scanner_data_cached(
        query,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )
    second_count, second_df = get_scanner_data_cached(
        query,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    assert query.calls == 1
    assert first_count == second_count == 2
    assert first_df.equals(second_df)


def test_scanner_data_cache_refresh_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = FakeQuery()

    get_scanner_data_cached(
        query,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )
    get_scanner_data_cached(
        query,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=True,
    )

    assert query.calls == 2


def test_scanner_cache_refetches_when_one_partner_file_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = FakeQuery()
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    get_scanner_data_cached(query, **kwargs)
    meta_files = list((tmp_path / "tradingview_scanner").glob("*.json"))
    assert meta_files
    meta_files[0].unlink()
    count, _ = get_scanner_data_cached(query, **kwargs)
    assert query.calls == 2
    assert count == 2

    frame_files = list((tmp_path / "tradingview_scanner").glob("*.parquet"))
    assert frame_files
    frame_files[0].unlink()
    count, df = get_scanner_data_cached(query, **kwargs)
    assert query.calls == 3
    assert count == 2
    assert list(df["name"]) == ["AAA", "BBB"]


class RaisingQuery:
    def __init__(self) -> None:
        self.calls = 0

    def get_scanner_data(self):
        self.calls += 1
        raise RuntimeError("tradingview is down")


class EmptySuccessQuery:
    def __init__(self) -> None:
        self.calls = 0

    def get_scanner_data(self):
        self.calls += 1
        return 0, pd.DataFrame(columns=["name", "volume"])


class FlakyThenHealthyQuery:
    """Fails the whole first scan (all retries), returns data on the next scan.

    ``call_with_resilience`` retries up to ``RetryConfig.attempts`` (default 3)
    times within one scan, so a single transient raise would be retried away.
    To model a real provider outage the first scan must exhaust every attempt.
    """

    def __init__(self, fail_attempts: int = 3) -> None:
        self.calls = 0
        self._fail_attempts = fail_attempts

    def get_scanner_data(self):
        self.calls += 1
        if self.calls <= self._fail_attempts:
            raise RuntimeError("tradingview is down")
        return 2, pd.DataFrame({"name": ["AAA", "BBB"], "volume": [10, 20]})


def _scanner_files(tmp_path):
    namespace = tmp_path / "tradingview_scanner"
    frames = list(namespace.glob("*.parquet")) if namespace.exists() else []
    metas = list(namespace.glob("*.json")) if namespace.exists() else []
    return frames, metas


def test_failed_scan_is_not_cached(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = RaisingQuery()

    with caplog.at_level(logging.WARNING, logger=scanner_module.LOG.name):
        count, df = get_scanner_data_cached(
            query,
            key_parts=("market", "filters", 100),
            columns=["name", "volume"],
            cache_ttl=60,
            refresh=False,
        )

    assert count == 0
    assert df.empty
    frames, metas = _scanner_files(tmp_path)
    assert not frames
    assert not metas
    assert any("not cached" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_successful_empty_scan_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = EmptySuccessQuery()
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    count, df = get_scanner_data_cached(query, **kwargs)
    assert count == 0
    assert df.empty
    frames, metas = _scanner_files(tmp_path)
    assert frames
    assert metas

    get_scanner_data_cached(query, **kwargs)
    assert query.calls == 1


def test_failed_scan_does_not_shadow_later_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = FlakyThenHealthyQuery()
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    _, first_df = get_scanner_data_cached(query, **kwargs)
    assert first_df.empty
    # The failed scan must not have written a stale empty cache entry.
    frames, metas = _scanner_files(tmp_path)
    assert not frames
    assert not metas

    count, second_df = get_scanner_data_cached(query, **kwargs)
    assert count == 2
    assert list(second_df["name"]) == ["AAA", "BBB"]


class FakeScanQuery:
    calls = 0

    def set_markets(self, *args):
        return self

    def select(self, *args):
        return self

    def where(self, *args):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args):
        return self

    def get_scanner_data(self):
        FakeScanQuery.calls += 1
        return 1, pd.DataFrame(
            {
                "name": ["AAA"],
                "description": ["Acme"],
                "close": [10.0],
                "change": [1.0],
                "volume": [1_000],
                "market_cap_basic": [1e9],
            }
        )


def test_scan_cache_key_ignores_filter_order(tmp_path, monkeypatch):
    # Query.where() ANDs its filters, so reordering them must hit the same
    # cache entry instead of refetching.
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(scanner_module, "Query", FakeScanQuery)
    monkeypatch.setattr(FakeScanQuery, "calls", 0)

    scanner_module.scan("us", ["filter_a", "filter_b"], cache_ttl=60)
    scanner_module.scan("us", ["filter_b", "filter_a"], cache_ttl=60)

    assert FakeScanQuery.calls == 1


def _shape_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "LOWDUP",
                "description": "Acme Ltd",
                "close": 20.0,
                "change": -4.0,
                "volume": 1_000.0,
                "market_cap_basic": 1_000_000.0,
                "EMA5": 19.0,
                "EMA20": 20.0,
                "EMA100": 21.0,
                "EMA200": 22.0,
                "RSI": 20.0,
            },
            {
                "name": "KEEPDUP",
                "description": "Acme Ltd",
                "close": 100.0,
                "change": 8.0,
                "volume": 10_000_000.0,
                "market_cap_basic": 10_000_000_000.0,
                "EMA5": 110.0,
                "EMA20": 100.0,
                "EMA100": 90.0,
                "EMA200": 80.0,
                "RSI": 60.0,
            },
            {
                "name": "BETA",
                "description": "Beta Ltd",
                "close": 80.0,
                "change": 5.0,
                "volume": 5_000_000.0,
                "market_cap_basic": 5_000_000_000.0,
                "EMA5": 85.0,
                "EMA20": 80.0,
                "EMA100": 70.0,
                "EMA200": 65.0,
                "RSI": 58.0,
            },
        ]
    )


def test_shape_scan_results_setup_score_sorts_then_dedupes():
    out = scanner_module.shape_scan_results(
        _shape_frame(), limit=5, order_by="setup_score", detail=False
    )

    assert "setup_score" in out.columns
    assert out["setup_score"].is_monotonic_decreasing
    assert "KEEPDUP" in out["name"].tolist()
    assert "LOWDUP" not in out["name"].tolist()
    assert out["description"].tolist().count("Acme Ltd") == 1


def test_shape_scan_results_hides_helper_score_columns_by_detail_mode():
    hidden = scanner_module.shape_scan_results(
        _shape_frame(), limit=5, order_by="setup_score", detail=False
    )
    detailed = scanner_module.shape_scan_results(
        _shape_frame(), limit=5, order_by="setup_score", detail=True
    )

    for col in scanner_module.SETUP_SCORE_COLUMNS:
        assert col not in hidden.columns
    assert "RSI" in detailed.columns
    assert "EMA5" not in detailed.columns
    assert "EMA20" not in detailed.columns


def test_shape_scan_results_non_setup_order_dedupes_and_limits_without_score():
    out = scanner_module.shape_scan_results(
        _shape_frame(), limit=1, order_by="volume", detail=False
    )

    assert len(out) == 1
    assert "setup_score" not in out.columns
    assert out.iloc[0]["name"] == "LOWDUP"
