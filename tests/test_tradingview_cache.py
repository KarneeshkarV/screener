from __future__ import annotations

import pandas as pd

from screener import cache
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
