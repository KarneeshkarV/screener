from __future__ import annotations

import os
import time

import pandas as pd

from screener import cache


def test_stable_key_is_deterministic_for_dict_order():
    a = cache.stable_key({"b": 2, "a": 1})
    b = cache.stable_key({"a": 1, "b": 2})
    assert a == b


def test_parse_ttl_suffixes():
    assert cache.parse_ttl("30s") == 30
    assert cache.parse_ttl("15m") == 900
    assert cache.parse_ttl("2h") == 7200
    assert cache.parse_ttl("1d") == 86400
    assert cache.parse_ttl("off") is None


def test_cached_frame_call_reuses_fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    calls = {"count": 0}

    def fetch() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame({"symbol": ["AAA"], "value": [1.0]})

    first = cache.cached_frame_call(
        "frames",
        ("same",),
        ttl_seconds=60,
        refresh=False,
        fetch=fetch,
    )
    second = cache.cached_frame_call(
        "frames",
        ("same",),
        ttl_seconds=60,
        refresh=False,
        fetch=fetch,
    )

    assert calls["count"] == 1
    assert first.equals(second)


def test_is_fresh_honors_ttl(tmp_path):
    path = tmp_path / "item.json"
    path.write_text("{}")
    now = time.time()
    assert cache.is_fresh(path, 60, now=now)
    os.utime(path, (now - 120, now - 120))
    assert not cache.is_fresh(path, 60, now=now)


def test_cache_area_registry_configures_paths(tmp_path):
    cache.reset_cache_area_paths()
    try:
        scanner = tmp_path / "scanner"
        panels = tmp_path / "panels"
        cache.set_cache_area_path("scanner", scanner)
        cache.set_cache_area_path("panels", panels)

        assert cache.known_cache_dirs()["scanner"] == scanner
        assert cache.cache_path("ns", "key", "json") == scanner / "ns" / "key.json"
        assert cache.panel_path("fii_dii") == panels / "fii_dii.parquet"
    finally:
        cache.reset_cache_area_paths()
