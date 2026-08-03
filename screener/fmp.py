"""One Financial Modeling Prep (FMP) HTTP client for every call site.

FMP access was reimplemented in ``garp``, ``insiders``, ``institutional``,
``backtester.fundamentals`` and ``backtester.data``'s ``FMPPriceFetcher`` with
divergent libraries (urllib vs. requests), timeouts, headers and error
handling. This module owns the single transport: base URLs, API-key
resolution, the one HTTP mechanism, a per-call timeout, the error mode and
JSON decoding. Call sites route their FMP requests through :class:`FmpClient`
/ :func:`request_json` instead of hand-wiring HTTP.

``resolve_api_key`` reads the project ``.env`` through the neutral
:mod:`screener.config`; this module must not import a feature package (see
``tests/test_import_boundaries.py``).

Caching (``screener.cache``) and retry/circuit-breaker (``screener.resilience``)
stay in ``screener.providers``; this module is only the transport used inside a
provider ``fetch_fn``. Non-2xx responses raise ``urllib.error.HTTPError`` (the
error mode every prior urllib site already had, and equivalent to the
requests-based site's ``raise_for_status``), so the resilience wrapper retries
and falls back exactly as before.

Optional session transport: callers that already own a connection-pooling
``requests``-style session (``backtester.fundamentals`` reuses one per worker)
may pass ``session=`` to :func:`request_json` / :class:`FmpClient`. The session
path issues ``session.get(url, headers=, timeout=)``, calls ``raise_for_status``
and returns ``.json()`` -- i.e. non-2xx raises ``requests.HTTPError`` for that
site instead of ``urllib.error.HTTPError``. Both are ``Exception`` subclasses,
so every ``except Exception`` retry/fallback path treats them identically; the
distinction is invisible to callers. ``session`` defaults to ``None`` (the
urllib mechanism), so existing call sites are unaffected.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol

from screener.config import load_env_file


class HttpResponse(Protocol):
    """Minimal response shape used by the optional session transport."""

    def raise_for_status(self) -> Any: ...

    def json(self) -> Any: ...


class HttpSession(Protocol):
    """A ``requests``-style session (only ``get`` is used).

    ``requests.Session`` satisfies this structurally; it is typed as a Protocol
    so :mod:`screener.fmp` need not import ``requests``.
    """

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse: ...


FMP_V3_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_V4_BASE_URL = "https://financialmodelingprep.com/api/v4"

# The prior sites all sent this User-Agent (garp ``_FMP_HEADERS`` and insiders
# ``_SCREENER_HEADERS`` were identical); some FMP endpoints reject a bare
# library default UA, so it is the unified default here.
DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; screener-cli/1.0)"
}

DEFAULT_TIMEOUT = 20.0


def resolve_api_key() -> str | None:
    """Resolve ``FMP_API_KEY``, loading the project ``.env`` once if unset."""
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    load_env_file()
    return os.environ.get("FMP_API_KEY")


def request_json(
    url: str,
    *,
    headers: Mapping[str, str] = DEFAULT_HEADERS,
    timeout: float = DEFAULT_TIMEOUT,
    session: HttpSession | None = None,
) -> Any:
    """GET ``url`` and decode the JSON body (the one HTTP mechanism).

    ``urllib.request.urlopen`` is used by attribute (not a bound import) so a
    call site's ``monkeypatch.setattr(<module>.urllib.request, "urlopen", ...)``
    still intercepts the request. Bytes are decoded with ``errors="ignore"`` to
    match the legacy urllib sites.

    When ``session`` is given, the request is delegated to
    ``session.get(url, headers=, timeout=)`` followed by ``raise_for_status``
    and ``.json()`` -- preserving the connection pooling and ``requests``-style
    error mode of the ``backtester.fundamentals`` call site verbatim.
    """
    if session is not None:
        response = session.get(url, headers=dict(headers), timeout=timeout)
        response.raise_for_status()
        return response.json()
    req = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


class FmpClient:
    """A single FMP endpoint (base URL + API key + header/timeout policy).

    ``get(path, params)`` appends ``apikey`` to ``params`` and requests
    ``{base_url}/{path}?{query}``, preserving each legacy site's URL shape and
    parameter order (``apikey`` last).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = FMP_V3_BASE_URL,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session: HttpSession | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.headers: dict[str, str] = (
            dict(headers) if headers is not None else dict(DEFAULT_HEADERS)
        )
        self.timeout = timeout
        self.session = session

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode({**dict(params or {}), "apikey": self.api_key})
        url = f"{self.base_url}/{path}?{query}"
        if self.session is None:
            # Byte-identical to the pre-session call so existing urllib callers
            # (and their ``request_json`` fakes that take no ``session`` kwarg)
            # are unaffected; the session kwarg is added only when one is set.
            return request_json(url, headers=self.headers, timeout=self.timeout)
        return request_json(
            url, headers=self.headers, timeout=self.timeout, session=self.session
        )


__all__ = [
    "FMP_V3_BASE_URL",
    "FMP_V4_BASE_URL",
    "DEFAULT_HEADERS",
    "DEFAULT_TIMEOUT",
    "FmpClient",
    "HttpResponse",
    "HttpSession",
    "request_json",
    "resolve_api_key",
]
