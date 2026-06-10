from __future__ import annotations

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
