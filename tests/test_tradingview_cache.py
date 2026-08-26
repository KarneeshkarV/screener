from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from screener import cache
from screener import scanner as scanner_module
from screener.providers import StaleDataError
from screener.resilience import RetryConfig
from screener.scanner import (
    FETCHED_AT_META_KEY,
    get_scanner_data_cached,
)


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


def test_as_of_round_trips_through_the_cache(tmp_path, monkeypatch):
    """A cached entry reports its original fetch time, not the read time."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    query = FakeQuery()
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=3600,
        refresh=False,
    )

    _, first_df, first_as_of = scanner_module._scanner_entry(query, **kwargs)
    _, second_df, second_as_of = scanner_module._scanner_entry(query, **kwargs)

    assert query.calls == 1  # second call served from cache
    assert first_as_of == second_as_of
    assert first_df.equals(second_df)
    # The sidecar really carries the timestamp, so it survives the parquet
    # + JSON round-trip on disk.
    sidecar = json.loads(
        list((tmp_path / "tradingview_scanner").glob("*.json"))[0].read_text()
    )
    # Full precision, not truncated to whole seconds: a truncated stamp reports
    # the fetch as up to a second older than it was.
    assert sidecar[FETCHED_AT_META_KEY] == first_as_of.isoformat()


def test_as_of_is_a_fresh_timestamp_on_live_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    before = datetime.now(UTC) - timedelta(seconds=1)

    _, _, as_of = scanner_module._scanner_entry(
        FakeQuery(),
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    assert datetime.now(UTC) + timedelta(seconds=1) > as_of > before
    assert as_of.tzinfo is not None


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


def test_stale_scan_cache_is_served_when_tradingview_is_down(
    tmp_path, monkeypatch, caplog
):
    """An expired entry beats an empty result when the provider is unreachable."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        refresh=False,
    )

    healthy = FakeQuery()
    get_scanner_data_cached(healthy, cache_ttl=60, **kwargs)

    # ttl_seconds=None makes every entry stale, so the next scan must refetch.
    down = RaisingQuery()
    with caplog.at_level(logging.WARNING, logger="screener.providers"):
        count, df = get_scanner_data_cached(down, cache_ttl=None, **kwargs)

    assert down.calls > 0  # the live fetch really was attempted
    assert count == 2
    assert list(df["name"]) == ["AAA", "BBB"]
    assert "Serving stale tradingview_scanner cache data" in caplog.text


def test_strict_scan_raises_instead_of_serving_stale(tmp_path, monkeypatch):
    """strict=True refuses the arbitrarily old entry a default scan would use."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        refresh=False,
    )

    get_scanner_data_cached(FakeQuery(), cache_ttl=60, **kwargs)

    down = RaisingQuery()
    with pytest.raises(StaleDataError):
        # One attempt keeps the outage test free of retry backoff sleeps.
        get_scanner_data_cached(down, cache_ttl=None, retries=1, strict=True, **kwargs)
    assert down.calls == 1  # strict still attempts the live fetch first


def test_strict_scan_false_keeps_the_stale_fallback(tmp_path, monkeypatch):
    """The default path still serves the old frame when TradingView is down."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    kwargs = dict(
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        refresh=False,
    )

    get_scanner_data_cached(FakeQuery(), cache_ttl=60, **kwargs)

    count, df = get_scanner_data_cached(
        RaisingQuery(), cache_ttl=None, retries=1, strict=False, **kwargs
    )
    assert count == 2
    assert list(df["name"]) == ["AAA", "BBB"]


class TimeoutRecordingQuery:
    """Records exactly which kwargs reached the request layer."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def get_scanner_data(self, **kwargs):
        self.kwargs = kwargs
        return 1, pd.DataFrame({"name": ["AAA"], "volume": [10]})


def test_timeout_reaches_the_request_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)

    query = TimeoutRecordingQuery()
    get_scanner_data_cached(
        query,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=None,  # always refetch so the query runs every call
        refresh=True,
        timeout=7.5,
    )
    assert query.kwargs == {"timeout": 7.5}

    unset = TimeoutRecordingQuery()
    get_scanner_data_cached(
        unset,
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=None,
        refresh=True,
        timeout=None,
    )
    assert unset.kwargs == {}  # None must not change the library's default


class RetryRecordingProvider:
    """Captures the retry config the scan hands the provider seam.

    Returns a canned payload without running ``fetch_fn``: the point is what
    the scan passes *to* the seam, and running it would hit the real Query.
    """

    def __init__(self) -> None:
        self.retries: list[RetryConfig | None] = []

    def fetch(self, key_parts, fetch_fn, **kwargs):
        self.retries.append(kwargs.get("retry"))
        return pd.DataFrame({"name": ["AAA"], "volume": [10]}), {"count": 1}


def test_retry_override_reaches_the_provider_seam(monkeypatch):
    provider = RetryRecordingProvider()
    monkeypatch.setattr(scanner_module, "SCANNER_PROVIDER", provider)

    total, df, as_of = scanner_module.scan(
        "us",
        [],
        limit=5,
        order_by="volume",
        retries=4,
    )

    assert provider.retries == [RetryConfig(attempts=4)]
    assert total == 1
    assert list(df["name"]) == ["AAA"]
    assert as_of is not None


def test_scan_without_cache_entry_still_returns_empty_on_failure(tmp_path, monkeypatch):
    """Stale-serve must not mask a cold outage: no entry still means empty."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)

    count, df = get_scanner_data_cached(
        RaisingQuery(),
        key_parts=("market", "filters", 100),
        columns=["name", "volume"],
        cache_ttl=60,
        refresh=False,
    )

    assert count == 0
    assert df.empty
    assert list(df.columns) == ["name", "volume"]


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


def test_scan_cache_key_ignores_filter_order(monkeypatch, fake_provider):
    # Query.where() ANDs its filters, so reordering them must derive the same
    # cache key. Now that the scan goes through the provider seam the key parts
    # can be asserted directly off ``FakeProvider``, instead of inferring them
    # from a cache hit against a monkeypatched ``CACHE_ROOT``.
    provider = fake_provider()
    monkeypatch.setattr(scanner_module, "SCANNER_PROVIDER", provider)
    monkeypatch.setattr(scanner_module, "Query", FakeScanQuery)

    scanner_module.scan("us", ["filter_a", "filter_b"], cache_ttl=60)
    scanner_module.scan("us", ["filter_b", "filter_a"], cache_ttl=60)

    assert len(provider.calls) == 2
    assert provider.calls[0][0] == provider.calls[1][0]


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


def test_legacy_entry_without_timestamp_reports_its_mtime_not_now(
    tmp_path, monkeypatch
) -> None:
    """A pre-``fetched_at`` cache entry must not claim to be freshly fetched.

    The TTL-gated path would make "now" harmless, because a served entry really
    is within the TTL. ``_read_stale`` is the dangerous one: it ignores the TTL
    by design, so an arbitrarily old legacy frame reporting ``as_of=now`` would
    look freshly fetched to a caller sizing real orders. The sidecar is written
    with the payload, so its mtime is the honest fetch time.
    """
    sidecar = tmp_path / "legacy.json"
    sidecar.write_text(json.dumps({"count": 1}))
    old = datetime.now(UTC) - timedelta(days=9)
    import os

    os.utime(sidecar, (old.timestamp(), old.timestamp()))

    as_of = scanner_module._fetched_at_from_meta({"count": 1}, sidecar)

    assert as_of.tzinfo is not None
    assert abs((as_of - old).total_seconds()) < 2
    assert (datetime.now(UTC) - as_of).days >= 8


def test_naive_stored_timestamp_is_refused_and_falls_back_to_mtime(
    tmp_path,
) -> None:
    """A naive timestamp cannot be compared against an aware "now" safely."""
    sidecar = tmp_path / "naive.json"
    sidecar.write_text("{}")
    old = datetime.now(UTC) - timedelta(days=3)
    import os

    os.utime(sidecar, (old.timestamp(), old.timestamp()))

    as_of = scanner_module._fetched_at_from_meta(
        {FETCHED_AT_META_KEY: "2020-01-01T00:00:00"}, sidecar
    )

    assert as_of.tzinfo is not None
    assert abs((as_of - old).total_seconds()) < 2
