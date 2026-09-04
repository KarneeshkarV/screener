"""The repo's one connection-pooling ``requests`` session shape.

Both FMP fetchers hit the same host from a ``ThreadPoolExecutor`` and both
need HTTP keep-alive to survive across tickers: a fresh connection per request
pays a TCP connect and a full TLS handshake, which on a cold 850-ticker
fundamentals fetch measured 24% of wall time (py-spy self-time: 16.1%
``do_handshake``, 8.0% ``create_connection``).

They used to answer that differently - ``FMPPriceFetcher`` shared one session
with a sized adapter, ``FMPFundamentalFetcher`` kept one session per worker
thread - so this module holds the single answer and both call it.

**Share one session; size its pool to the worker count.** ``requests.Session``
is documented as not thread-safe, and the caveat is real but narrow: it covers
mutating session state concurrently. These callers mount their adapters before
any worker starts and then issue plain GETs, the cookie jar underneath is
already lock-guarded, and urllib3's connection pool is thread-safe by design.
Sharing is what lets one pool serve every ticker instead of fragmenting
keep-alive across N pools.

Sizing is the part that is easy to get wrong and silent when you do. An
unsized adapter pools 10 connections; a caller running more workers than that
gets ``Connection pool is full, discarding connection`` and re-handshakes on
every overflow request, which is the same cost the pooling was meant to
remove, only harder to see.

The rule this encodes: **share a pooled session when the requests are
stateless, and keep per-thread sessions only when each thread needs its own
cookie state.** ``unusual_volume/nse_client.py`` is the one place that fails
that test - NSE gates its APIs behind per-session cookie priming, and a soft
block re-primes one thread's session without disturbing the others - so it
keeps ``threading.local`` on purpose and is not a candidate for this helper.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter


def pooled_session(
    max_workers: int, *, session: requests.Session | None = None
) -> requests.Session:
    """Return a session whose connection pool is sized for ``max_workers``.

    ``session`` is the caller's own transport when it has one - a test double,
    a proxy, a mounted retry adapter - and is returned configured rather than
    replaced. A double that does not implement ``mount`` is passed through
    untouched, since only a real session has a pool to size.
    """
    resolved = session or requests.Session()
    if hasattr(resolved, "mount"):
        workers = max(1, int(max_workers))
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        resolved.mount("http://", adapter)
        resolved.mount("https://", adapter)
    return resolved
