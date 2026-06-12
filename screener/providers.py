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

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

from screener.cache import cached_frame_call, cached_json_call
from screener.resilience import RetryConfig, call_with_resilience


T = TypeVar("T")

Kind = Literal["json", "frame"]


class _Unset:
    """Sentinel so ``ttl_seconds=None`` (TTL off) differs from "use default"."""


_UNSET = _Unset()


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative description of one cached, resilience-wrapped data source.

    ``provider`` is the resilience circuit-breaker name ("fmp", "yfinance",
    "nse", "tradingview", "openscreener", ...). ``namespace`` is the on-disk
    cache namespace. ``ttl_seconds`` is the default cache TTL (overridable
    per-call). ``kind`` selects the JSON vs parquet cache backend.
    """

    provider: str
    namespace: str
    ttl_seconds: float | None
    kind: Kind = "json"


class CachedProvider:
    """fetch(key_parts, fetch_fn, *, refresh, fallback) -> data | fallback.

    One call = TTL cache lookup -> on miss, resilience-wrapped fetch -> cache
    store. The resilience wrapper retries ``fetch_fn`` and trips the provider's
    circuit breaker; on exhausted retries / open circuit it returns
    ``fallback`` (which is then cached, mirroring the legacy hand-wired sites).
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
            return call_with_resilience(
                self.spec.provider,
                op,
                fetch_fn,
                fallback=fallback,
                retry=retry,
            )

        if self.spec.kind == "frame":
            return cached_frame_call(
                self.spec.namespace,
                key_parts,
                ttl_seconds=ttl,
                refresh=refresh,
                fetch=resilient,
            )
        return cached_json_call(
            self.spec.namespace,
            key_parts,
            ttl_seconds=ttl,
            refresh=refresh,
            fetch=resilient,
        )


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


__all__ = ["ProviderSpec", "CachedProvider", "FakeProvider"]
