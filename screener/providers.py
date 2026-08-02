"""One provider seam: TTL caching + retry/circuit-breaker behind one call.

Data-fetch sites historically hand-wired three concerns at every call:
TTL caching (``screener.cache``), retry/circuit-breaker
(``screener.resilience``) and session handling. This module composes the
first two behind a single ``CachedProvider.fetch(...)`` so a call site
declares a :class:`ProviderSpec` once at module top and then calls
``PROVIDER.fetch(key_parts, fetch_fn, fallback=...)``.

``cache.py`` and ``resilience.py`` remain the implementation underneath; this
module only orchestrates them. Cache namespaces and TTLs are preserved exactly
so on-disk caches stay valid.

The seam is injectable for tests: a module-level :class:`CachedProvider` can be
swapped for :class:`FakeProvider` (no cache, no network) by reassigning the
module attribute. See ``tests/conftest.py`` for the fake adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, cast

from screener.cache import (
    FrameWithMeta as FrameWithMeta,  # re-exported: the "frame_meta" payload type
    cache_path,
    cached_frame_call,
    cached_frame_meta_call,
    cached_json_call,
    read_frame,
    read_frame_meta,
    read_json,
    stable_key,
)
from screener.resilience import RetryConfig, call_with_resilience

if TYPE_CHECKING:
    import pandas as pd


T = TypeVar("T")
LOG = logging.getLogger(__name__)

Kind = Literal["json", "frame", "frame_meta"]


class _Unset:
    """Sentinel so ``ttl_seconds=None`` (TTL off) differs from "use default"."""


_UNSET = _Unset()
_FETCH_FAILED = object()
# Distinguishes "no cache entry" from a cached ``null``/empty payload.
_MISSING = object()


class _ProviderFetchFailed(RuntimeError):
    """Prevent a resilience fallback from being persisted by cache helpers."""


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative description of one cached, resilience-wrapped data source.

    ``provider`` is the resilience circuit-breaker name ("fmp", "yfinance",
    "nse", "tradingview", "openscreener", ...). ``namespace`` is the on-disk
    cache namespace. ``ttl_seconds`` is the default cache TTL (overridable
    per-call). ``kind`` selects the cache backend: ``"json"``, ``"frame"``
    (parquet), or ``"frame_meta"`` (parquet plus a JSON sidecar, for payloads
    that are a frame *and* a scalar the frame cannot carry).
    """

    provider: str
    namespace: str
    ttl_seconds: float | None
    kind: Kind = "json"


class CachedProvider:
    """fetch(key_parts, fetch_fn, *, refresh, fallback) -> data | fallback.

    One call = TTL cache lookup -> on miss, resilience-wrapped fetch -> cache
    store. Provider failures are never cached; a stale entry is preferred over
    the caller's fallback when retries are exhausted or the circuit is open.
    """

    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec

    def fetch(
        self,
        key_parts: Any,
        fetch_fn: Callable[[], T],
        *,
        refresh: bool = False,
        fallback: T = None,  # type: ignore[assignment]
        ttl_seconds: float | None | _Unset = _UNSET,
        operation: str | None = None,
        retry: RetryConfig | None = None,
    ) -> T:
        ttl = self.spec.ttl_seconds if isinstance(ttl_seconds, _Unset) else ttl_seconds
        op = operation or self.spec.namespace

        def resilient() -> T:
            result = call_with_resilience(
                self.spec.provider,
                op,
                fetch_fn,
                fallback=_FETCH_FAILED,
                retry=retry,
            )
            if result is _FETCH_FAILED:
                raise _ProviderFetchFailed
            return cast(T, result)

        try:
            if self.spec.kind == "frame_meta":
                # kind == "frame_meta" callers bind T to (DataFrame, dict);
                # cached_frame_meta_call works in that pair, so cast its types.
                return cast(
                    T,
                    cached_frame_meta_call(
                        self.spec.namespace,
                        key_parts,
                        ttl_seconds=ttl,
                        refresh=refresh,
                        fetch=cast("Callable[[], FrameWithMeta]", resilient),
                    ),
                )
            if self.spec.kind == "frame":
                # kind == "frame" callers bind T to pd.DataFrame;
                # cached_frame_call works in DataFrames, so cast its types.
                return cast(
                    T,
                    cached_frame_call(
                        self.spec.namespace,
                        key_parts,
                        ttl_seconds=ttl,
                        refresh=refresh,
                        fetch=cast("Callable[[], pd.DataFrame]", resilient),
                    ),
                )
            return cached_json_call(
                self.spec.namespace,
                key_parts,
                ttl_seconds=ttl,
                refresh=refresh,
                fetch=resilient,
            )
        except _ProviderFetchFailed:
            stale = self._read_stale(key_parts)
            if stale is not _MISSING:
                LOG.warning(
                    "Serving stale %s cache data due to %s provider failure",
                    self.spec.namespace,
                    self.spec.provider,
                )
                return cast(T, stale)
            return fallback

    def _read_stale(self, key_parts: Any) -> Any:
        """Return the cached value ignoring TTL, or ``_MISSING`` if there is none."""
        if self.spec.kind == "frame_meta":
            entry = read_frame_meta(self.spec.namespace, key_parts)
            return _MISSING if entry is None else entry
        path = cache_path(
            self.spec.namespace,
            stable_key(key_parts),
            "parquet" if self.spec.kind == "frame" else "json",
        )
        if self.spec.kind == "frame":
            frame = read_frame(path)
            return _MISSING if frame is None else frame
        return read_json(path, default=_MISSING)


class FakeProvider:
    """Test double for :class:`CachedProvider`: no cache, no resilience.

    ``fetch`` calls ``fetch_fn`` directly and returns its result (or
    ``fallback`` when ``fetch_fn`` raises). Records ``(key_parts, refresh)``
    for assertions.
    """

    def __init__(self, spec: ProviderSpec | None = None) -> None:
        self.spec = spec
        self.calls: list[tuple[Any, bool]] = []

    def fetch(
        self,
        key_parts: Any,
        fetch_fn: Callable[[], T],
        *,
        refresh: bool = False,
        fallback: T = None,  # type: ignore[assignment]
        ttl_seconds: Any = None,
        operation: str | None = None,
        retry: RetryConfig | None = None,
    ) -> T:
        self.calls.append((key_parts, refresh))
        try:
            return fetch_fn()
        except Exception:
            return fallback


__all__ = ["ProviderSpec", "CachedProvider", "FakeProvider", "FrameWithMeta"]
