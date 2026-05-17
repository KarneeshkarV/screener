"""Shared cookie-primed accessor for ``www.nseindia.com/api/*`` JSON endpoints.

NSE's JSON API rejects non-browser User-Agents and requires the cookies set by
a prior visit to the homepage. We reuse the already-primed ``requests.Session``
from ``jugaad_data.nse.NSEArchives`` (per project decision — no ``nsepython``
dependency), layer browser headers + a homepage warm-up on top, and route every
call through ``call_with_resilience`` so a flaky/blocking NSE degrades to
``None`` rather than raising. On a 401/403 (cookie expiry / soft block) we
re-prime once and retry.

``requests.Session`` is not thread-safe, and the option-chain / pledge overlays
fan out across ``ThreadPoolExecutor`` workers. Each worker therefore gets its
own homepage-primed session via ``threading.local()``; a soft-block reprime
rebuilds only the calling thread's session.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import requests

from screener.cache import cached_json_call
from screener.resilience import call_with_resilience

_NSE_HOME = "https://www.nseindia.com"
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{_NSE_HOME}/",
}

_tls = threading.local()


class _SoftBlock:
    """Sentinel for an NSE 401/403 (cookie expiry) so we re-prime once."""


_SOFT_BLOCK = _SoftBlock()


def _new_session() -> requests.Session:
    from jugaad_data.nse import NSEArchives

    sess = NSEArchives().s
    sess.headers.update(_BROWSER_HEADERS)
    return sess


def get_primed_session() -> requests.Session:
    """Return the calling thread's session with NSE cookies seeded (once).

    ``requests.Session`` is not thread-safe, so each worker thread keeps its
    own homepage-primed session in thread-local storage.
    """
    session: requests.Session | None = getattr(_tls, "session", None)
    if session is None:
        session = _new_session()
        _tls.session = session
        _tls.primed = False
    if not getattr(_tls, "primed", False):
        call_with_resilience(
            "nse",
            "nse homepage warmup",
            lambda: session.get(f"{_NSE_HOME}/", timeout=10),
            fallback=None,
        )
        _tls.primed = True
    return session


def _reprime() -> requests.Session:
    """Rebuild *this thread's* session + homepage warm-up (cookie expiry /
    soft block). Other threads keep their own sessions untouched."""
    _tls.session = None
    _tls.primed = False
    return get_primed_session()


def fetch_nse_json(
    url: str,
    operation: str,
    *,
    timeout: float = 10.0,
    after_reprime: Callable[[requests.Session], None] | None = None,
) -> Any | None:
    """GET ``url`` and return parsed JSON, or ``None`` on any failure.

    Never raises — overlays must degrade gracefully (mirrors the contract of
    ``delivery._load_one_day``).
    """

    def _do(session: requests.Session) -> Any | None:
        def _request() -> Any | None:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in (401, 403):
                return _SOFT_BLOCK
            resp.raise_for_status()
            return resp.json()

        return call_with_resilience("nse", operation, _request, fallback=None)

    result = _do(get_primed_session())
    if result is _SOFT_BLOCK:
        session = _reprime()
        if after_reprime is not None:
            after_reprime(session)
        result = _do(session)
    return None if result is _SOFT_BLOCK else result


def nse_cached_json(
    namespace: str,
    key_parts: Any,
    url: str,
    operation: str,
    *,
    refresh: bool = False,
    ttl_seconds: float | None = 900.0,
    after_reprime: Callable[[requests.Session], None] | None = None,
) -> Any | None:
    """TTL-cached ``fetch_nse_json`` (default 15 min, intraday-safe)."""
    return cached_json_call(
        namespace,
        key_parts,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        fetch=lambda: fetch_nse_json(url, operation, after_reprime=after_reprime),
    )
