"""Single seam for every ``nseindia.com`` quirk: priming, soft-block reprime,
the F&O ban-list feed, and the NSE trading calendar.

NSE's JSON API rejects non-browser User-Agents and requires the cookies set by
a prior visit to the homepage. We reuse the already-primed ``requests.Session``
from ``jugaad_data.nse.NSEArchives`` (per project decision — no ``nsepython``
dependency), layer browser headers + a homepage warm-up on top, and route every
call through ``call_with_resilience`` so a flaky/blocking NSE degrades to
``None`` rather than raising. On a 401/403 (cookie expiry / soft block) we
re-prime once and retry.

Some endpoints (the equity option chain) are gated behind a *second* page
visit. ``get_json(..., extra_prime_page=...)`` handles that inside this module:
"which extra pages this thread has primed on which session" is tracked here, so
call sites never reach for thread-locals of their own.

``requests.Session`` is not thread-safe, and the option-chain / pledge overlays
fan out across ``ThreadPoolExecutor`` workers. Each worker therefore gets its
own homepage-primed session via ``threading.local()``; a soft-block reprime
rebuilds only the calling thread's session.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, timedelta
from typing import Any

import requests

from screener.cache import cached_json_call
from screener.resilience import call_with_resilience

LOG = logging.getLogger(__name__)

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

    # NSEArchives is untyped, so .s is Any; annotate to the documented Session.
    sess: requests.Session = NSEArchives().s
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
        _tls.primed_pages = {}
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
    _tls.primed_pages = {}
    return get_primed_session()


def _prime_page(session: requests.Session, page_url: str) -> None:
    """Visit ``page_url`` once per (thread, session) to seed its cookies.

    NSE gates some APIs (the equity option chain) behind a prior visit to a
    specific page; without it the API returns ``{}`` (also the documented
    off-hours/market-closed response). Only mark primed on a real success so a
    failed warm-up retries on a later call rather than being cached as done.
    """
    primed_pages: dict[int, set[str]] = getattr(_tls, "primed_pages", None) or {}
    session_id = id(session)
    if page_url in primed_pages.get(session_id, set()):
        return
    try:
        resp = session.get(page_url, timeout=10)
        if resp.status_code < 400:
            primed_pages.setdefault(session_id, set()).add(page_url)
            _tls.primed_pages = primed_pages
    except Exception:
        LOG.debug("NSE page priming failed for %s; will retry on next call", page_url)


def fetch_nse_json(
    url: str,
    operation: str,
    *,
    timeout: float = 10.0,
    extra_prime_page: str | None = None,
) -> Any | None:
    """GET ``url`` and return parsed JSON, or ``None`` on any failure.

    ``extra_prime_page`` (e.g. the option-chain page) is visited once per
    thread/session before the API call, and re-visited after a soft-block
    reprime. Never raises — overlays must degrade gracefully (mirrors the
    contract of ``delivery._load_one_day``).
    """

    def _do(session: requests.Session) -> Any | None:
        if extra_prime_page is not None:
            _prime_page(session, extra_prime_page)

        def _request() -> Any | None:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in (401, 403):
                return _SOFT_BLOCK
            resp.raise_for_status()
            return resp.json()

        return call_with_resilience("nse", operation, _request, fallback=None)

    result = _do(get_primed_session())
    if result is _SOFT_BLOCK:
        result = _do(_reprime())
    return None if result is _SOFT_BLOCK else result


def nse_cached_json(
    namespace: str,
    key_parts: Any,
    url: str,
    operation: str,
    *,
    refresh: bool = False,
    ttl_seconds: float | None = 900.0,
    extra_prime_page: str | None = None,
) -> Any | None:
    """TTL-cached ``fetch_nse_json`` (default 15 min, intraday-safe)."""
    return cached_json_call(
        namespace,
        key_parts,
        ttl_seconds=ttl_seconds,
        refresh=refresh,
        fetch=lambda: fetch_nse_json(url, operation, extra_prime_page=extra_prime_page),
    )


def fetch_nse_text(url: str, operation: str, *, timeout: float = 8.0) -> str | None:
    """GET ``url`` through the primed/repriming session and return the body text.

    Used for NSE archive CSV feeds (e.g. the F&O ban list). Returns ``None`` on
    a non-200, a soft block that survives one reprime, or any network failure.
    """

    def _do(session: requests.Session) -> Any | None:
        def _request() -> Any | None:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in (401, 403):
                return _SOFT_BLOCK
            if resp.status_code != 200:
                return None
            return resp.text

        return call_with_resilience("nse", operation, _request, fallback=None)

    result = _do(get_primed_session())
    if result is _SOFT_BLOCK:
        result = _do(_reprime())
    return None if (result is _SOFT_BLOCK or not isinstance(result, str)) else result


# ── trading calendar ───────────────────────────────────────────────────────

_HOLIDAYS_URL = "https://www.nseindia.com/api/holiday-master?type=trading"


def _parse_holiday_payload(raw: Any) -> set[date]:
    """Extract trading-holiday dates from NSE's ``holiday-master`` payload.

    The endpoint returns ``{"<segment>": [{"tradingDate": "DD-Mon-YYYY", ...}]}``.
    We union the dates across whatever segment lists are present.
    """
    out: set[date] = set()
    if not isinstance(raw, dict):
        return out
    for rows in raw.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("tradingDate") or row.get("date")
            if not value:
                continue
            try:
                import pandas as pd

                parsed = pd.to_datetime(str(value), dayfirst=True).date()
            except (ValueError, TypeError):
                continue
            out.add(parsed)
    return out


class TradingCalendar:
    """NSE trading-day arithmetic: weekday check + a lazily-loaded holiday set.

    Holidays are fetched once (cached 24h through the disk cache) the first
    time a holiday-sensitive query runs. If the fetch fails the calendar
    degrades to weekday-only behaviour — exactly today's logic — so nothing
    breaks when NSE is unreachable or in offline tests.
    """

    def __init__(self) -> None:
        self._holidays: set[date] | None = None
        self._lock = threading.Lock()

    def _holiday_set(self) -> set[date]:
        if self._holidays is None:
            with self._lock:
                if self._holidays is None:
                    self._holidays = self._load_holidays()
        return self._holidays

    def _load_holidays(self) -> set[date]:
        raw = nse_cached_json(
            "nse_holidays",
            ("holidays", str(date.today().year)),
            _HOLIDAYS_URL,
            "nse holiday master",
            ttl_seconds=24 * 3600,
        )
        return _parse_holiday_payload(raw)

    def is_trading_day(self, d: date) -> bool:
        """True if ``d`` is a weekday and not a known NSE holiday.

        When the holiday set is unavailable this is a pure weekday check,
        preserving the legacy weekend-only behaviour.
        """
        if d.weekday() >= 5:
            return False
        return d not in self._holiday_set()

    def last_trading_day_on_or_before(self, d: date, *, lookback: int = 7) -> date:
        """Walk back from ``d`` to the nearest trading day, bounded by lookback.

        Falls back to returning ``d`` if no trading day is found within the
        window (matching the pre-existing weekday-only walk-back, which also
        could only walk a bounded number of days).
        """
        for delta in range(lookback + 1):
            candidate = d - timedelta(days=delta)
            if self.is_trading_day(candidate):
                return candidate
        return d


_CALENDAR = TradingCalendar()


def is_trading_day(d: date) -> bool:
    """Module-level shortcut for :meth:`TradingCalendar.is_trading_day`."""
    return _CALENDAR.is_trading_day(d)


def last_trading_day_on_or_before(d: date, *, lookback: int = 7) -> date:
    """Module-level shortcut for :meth:`TradingCalendar.last_trading_day_on_or_before`."""
    return _CALENDAR.last_trading_day_on_or_before(d, lookback=lookback)
