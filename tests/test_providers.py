"""Unit tests for the CachedProvider seam — offline, no network."""

from __future__ import annotations

import pandas as pd

from screener import cache
from screener import resilience
from screener.providers import CachedProvider, FakeProvider, ProviderSpec


def _json_provider(ttl: float | None = 60) -> CachedProvider:
    return CachedProvider(
        ProviderSpec(provider="test", namespace="provider_json", ttl_seconds=ttl)
    )


def _frame_provider(ttl: float | None = 60) -> CachedProvider:
    return CachedProvider(
        ProviderSpec(
            provider="test", namespace="provider_frame", ttl_seconds=ttl, kind="frame"
        )
    )


def test_fetch_caches_json_on_miss_then_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider()
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"value": calls["n"]}

    first = provider.fetch(("k",), fetch)
    second = provider.fetch(("k",), fetch)

    assert first == {"value": 1}
    assert second == {"value": 1}  # served from cache, fetch not re-run
    assert calls["n"] == 1


def test_refresh_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider()
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"value": calls["n"]}

    provider.fetch(("k",), fetch)
    refreshed = provider.fetch(("k",), fetch, refresh=True)

    assert refreshed == {"value": 2}
    assert calls["n"] == 2


def test_resilience_fallback_is_returned_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    # One attempt, no sleep: a raising fetch trips straight to the fallback.
    provider = _json_provider()
    retry = resilience.RetryConfig(attempts=1, base_delay=0.0, jitter=0.0)

    def boom() -> dict:
        raise RuntimeError("provider down")

    out = provider.fetch(("k",), boom, fallback={"fallback": True}, retry=retry)
    assert out == {"fallback": True}

    # The fallback was cached, so a follow-up does not re-invoke the breaker.
    again = provider.fetch(("k",), boom, fallback={"fallback": "ignored"}, retry=retry)
    assert again == {"fallback": True}


def test_ttl_none_disables_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider(ttl=None)
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"value": calls["n"]}

    provider.fetch(("k",), fetch)
    provider.fetch(("k",), fetch)
    assert calls["n"] == 2  # ttl=None means never fresh -> always refetch


def test_per_call_ttl_override(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider(ttl=60)  # default would cache
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"value": calls["n"]}

    provider.fetch(("k",), fetch, ttl_seconds=None)  # override: disable cache
    provider.fetch(("k",), fetch, ttl_seconds=None)
    assert calls["n"] == 2


def test_frame_kind_round_trips_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _frame_provider()
    calls = {"n": 0}

    def fetch() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    first = provider.fetch(("k",), fetch)
    second = provider.fetch(("k",), fetch)

    assert calls["n"] == 1  # second served from parquet cache
    assert first.equals(second)
    assert list(first.columns) == ["a", "b"]


def test_fake_provider_runs_fetch_without_cache():
    fake = FakeProvider()
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"value": calls["n"]}

    assert fake.fetch(("k",), fetch) == {"value": 1}
    assert fake.fetch(("k",), fetch) == {"value": 2}  # no caching
    assert fake.calls == [(("k",), False), (("k",), False)]


def test_fake_provider_returns_fallback_on_error():
    fake = FakeProvider()

    def boom() -> dict:
        raise RuntimeError("down")

    assert fake.fetch(("k",), boom, fallback={"x": 1}) == {"x": 1}
