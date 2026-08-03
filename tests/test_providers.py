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


def test_resilience_failure_is_not_cached_and_second_call_refetches(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    # One attempt, no sleep: a raising fetch trips straight to the fallback.
    provider = _json_provider()
    retry = resilience.RetryConfig(attempts=1, base_delay=0.0, jitter=0.0)

    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider down")
        return {"live": True}

    out = provider.fetch(("k",), fetch, fallback={"fallback": True}, retry=retry)
    assert out == {"fallback": True}

    again = provider.fetch(("k",), fetch, fallback={"fallback": "ignored"}, retry=retry)
    assert again == {"live": True}
    assert calls["n"] == 2


def test_stale_json_cache_is_served_on_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider(ttl=None)
    retry = resilience.RetryConfig(attempts=1, base_delay=0.0, jitter=0.0)

    assert provider.fetch(("k",), lambda: {"old": True}) == {"old": True}

    def boom() -> dict:
        raise RuntimeError("provider down")

    with caplog.at_level("WARNING", logger="screener.providers"):
        result = provider.fetch(("k",), boom, fallback={"fallback": True}, retry=retry)

    assert result == {"old": True}
    assert "Serving stale provider_json cache data" in caplog.text


def test_stale_json_null_is_distinguished_from_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _json_provider(ttl=None)
    retry = resilience.RetryConfig(attempts=1, base_delay=0.0, jitter=0.0)

    assert provider.fetch(("k",), lambda: None) is None

    def boom():
        raise RuntimeError("provider down")

    assert (
        provider.fetch(("k",), boom, fallback={"fallback": True}, retry=retry) is None
    )


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


def _frame_meta_provider(ttl: float | None = 60) -> CachedProvider:
    return CachedProvider(
        ProviderSpec(
            provider="test",
            namespace="provider_frame_meta",
            ttl_seconds=ttl,
            kind="frame_meta",
        )
    )


def test_frame_meta_kind_round_trips_frame_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _frame_meta_provider()
    calls = {"n": 0}

    def fetch() -> tuple[pd.DataFrame, dict]:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2]}), {"count": 7}

    first_frame, first_meta = provider.fetch(("k",), fetch)
    second_frame, second_meta = provider.fetch(("k",), fetch)

    assert calls["n"] == 1  # second served from the parquet + sidecar pair
    assert first_frame.equals(second_frame)
    assert first_meta == second_meta == {"count": 7}


def test_frame_meta_missing_sidecar_forces_a_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _frame_meta_provider()
    calls = {"n": 0}

    def fetch() -> tuple[pd.DataFrame, dict]:
        calls["n"] += 1
        return pd.DataFrame({"a": [1]}), {"count": calls["n"]}

    provider.fetch(("k",), fetch)
    for meta_file in (tmp_path / "provider_frame_meta").glob("*.json"):
        meta_file.unlink()

    _, meta = provider.fetch(("k",), fetch)
    assert calls["n"] == 2
    assert meta == {"count": 2}


def test_stale_frame_meta_cache_is_served_on_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    provider = _frame_meta_provider(ttl=None)
    retry = resilience.RetryConfig(attempts=1, base_delay=0.0, jitter=0.0)

    provider.fetch(("k",), lambda: (pd.DataFrame({"a": [1]}), {"count": 3}))

    def boom() -> tuple[pd.DataFrame, dict]:
        raise RuntimeError("provider down")

    with caplog.at_level("WARNING", logger="screener.providers"):
        frame, meta = provider.fetch(("k",), boom, fallback=None, retry=retry)

    assert meta == {"count": 3}
    assert frame["a"].tolist() == [1]
    assert "Serving stale provider_frame_meta cache data" in caplog.text


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
