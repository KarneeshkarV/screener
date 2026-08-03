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
import os
import threading
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
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


# ── archive files ──────────────────────────────────────────────────────────


def save_delivery_bhavcopy(
    dt: date,
    cache_dir: Path,
    *,
    resilience_call=call_with_resilience,
) -> Path | None:
    """Save NSE delivery bhavcopy through the NSE Adapter Seam."""
    from jugaad_data.nse import full_bhavcopy_save

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = resilience_call(
        "nse",
        f"delivery bhavcopy {dt}",
        lambda: full_bhavcopy_save(dt, str(cache_dir)),
        fallback=None,
    )
    if path is None or not path or not os.path.isfile(path):
        return None
    return Path(path)


def load_delivery_bhavcopy_csv(
    dt: date,
    cache_dir: Path,
    *,
    resilience_call=call_with_resilience,
) -> pd.DataFrame | None:
    """Load one raw delivery bhavcopy CSV, returning ``None`` on failures."""
    path = save_delivery_bhavcopy(
        dt,
        cache_dir,
        resilience_call=resilience_call,
    )
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def cash_bhavcopy_cache_path(d: date, cache_root: Path) -> Path:
    """Cache path for NSE cash bhavcopy keyed by requested URL date."""
    day_dir = cache_root / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"sec_bhavdata_full_{d.strftime('%d%b%Y')}bhav.csv"


def read_cash_bhavcopy_raw(
    d: date,
    *,
    cache_root: Path,
    resilience_call=call_with_resilience,
) -> pd.DataFrame:
    """Download or load raw cash bhavcopy CSV with no domain filtering."""
    from jugaad_data.nse import NSEArchives

    path = cash_bhavcopy_cache_path(d, cache_root)
    if not path.exists():
        n = NSEArchives()
        resilience_call(
            "nse",
            f"cash bhavcopy {d}",
            lambda: n.full_bhavcopy_save(d, str(path.parent)),
            fallback=None,
        )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        # pandas >= 3 infers a dedicated string dtype for text columns, so an
        # ``== object`` check alone would silently skip stripping there.
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def fo_bhavcopy_cache_path(d: date, cache_root: Path) -> Path:
    """Cache path for decoded NSE F&O UDiff bhavcopy CSV."""
    day_dir = cache_root / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"BhavCopy_NSE_FO_{d.strftime('%Y%m%d')}.csv"


# NSE switched the F&O bhavcopy to the UDiff format on this date; earlier dates
# only exist in the legacy ``fo<DD><MMM><YYYY>bhav.csv.zip`` archive below.
FO_UDIFF_START = date(2024, 7, 8)

# Legacy (pre-UDiff) F&O bhavcopy archive, e.g.
# .../content/historical/DERIVATIVES/2023/JAN/fo02JAN2023bhav.csv.zip
LEGACY_FO_ARCHIVE_URL = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{yyyy}/{mmm}/fo{dd}{mmm}{yyyy}bhav.csv.zip"
)
_MONTHS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

# Legacy FO bhavcopy columns -> UDiff schema names. Legacy header:
#   INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW,
#   CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP
# Legacy has no UndrlygPric and no NewBrdLotQty.
_LEGACY_COLUMN_MAP = {
    "SYMBOL": "TckrSymb",
    "EXPIRY_DT": "XpryDt",
    "STRIKE_PR": "StrkPric",
    "OPTION_TYP": "OptnTp",
    "OPEN": "OpnPric",
    "HIGH": "HghPric",
    "LOW": "LwPric",
    "CLOSE": "ClsPric",
    "SETTLE_PR": "SttlmPric",
    "OPEN_INT": "OpnIntrst",
    "CHG_IN_OI": "ChngInOpnIntrst",
    "CONTRACTS": "TtlTradgVol",
    "TIMESTAMP": "TradDt",
}
# Legacy INSTRUMENT -> UDiff FinInstrmTp.
_LEGACY_INSTRUMENT_MAP = {
    "OPTSTK": "STO",
    "OPTIDX": "IDO",
    "FUTSTK": "STF",
    "FUTIDX": "IDF",
}


def _legacy_fo_url(d: date) -> str:
    mmm = _MONTHS[d.month - 1]
    return LEGACY_FO_ARCHIVE_URL.format(yyyy=d.year, mmm=mmm, dd=f"{d.day:02d}")


def normalize_legacy_fo_bhavcopy(frame: pd.DataFrame) -> pd.DataFrame:
    """Map legacy FO bhavcopy columns/values onto the UDiff schema.

    Detected by the presence of the legacy ``INSTRUMENT`` column, so cached
    legacy CSVs normalize on re-read too. Dates become ``datetime64`` (legacy
    ``TIMESTAMP`` is ``02-JAN-2023``; ``EXPIRY_DT`` is ``25-Jan-2023``), which
    the downstream ``pd.to_datetime`` consumers accept unchanged. The trailing
    ``Unnamed`` junk column is dropped.
    """
    df = frame.rename(columns=_LEGACY_COLUMN_MAP)
    junk = [c for c in df.columns if str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)
    if "INSTRUMENT" in df.columns:
        df["FinInstrmTp"] = (
            df["INSTRUMENT"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(_LEGACY_INSTRUMENT_MAP)
            .fillna(df["INSTRUMENT"])
        )
        df = df.drop(columns=["INSTRUMENT"])
    for col in ("TradDt", "XpryDt"):
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip().str.title(),
                format="%d-%b-%Y",
                errors="coerce",
            )
    return df


def read_fo_bhavcopy_raw(
    d: date,
    *,
    cache_root: Path,
    archive_url_template: str,
    resilience_call=call_with_resilience,
) -> pd.DataFrame:
    """Download or load decoded raw F&O bhavcopy CSV.

    Dates on or after :data:`FO_UDIFF_START` use the UDiff archive (via
    ``archive_url_template``); earlier dates fall back to the legacy archive
    and are normalized to the UDiff column names so callers see one schema.
    """
    from jugaad_data.nse import NSEArchives

    is_legacy = d < FO_UDIFF_START
    path = fo_bhavcopy_cache_path(d, cache_root)
    if not path.exists():
        n = NSEArchives()
        url = (
            _legacy_fo_url(d)
            if is_legacy
            else archive_url_template.format(yyyymmdd=d.strftime("%Y%m%d"))
        )
        response = resilience_call(
            "nse",
            f"fo bhavcopy {d}",
            lambda: n.s.get(url, timeout=10),
            fallback=None,
        )
        if response is None:
            raise RuntimeError(f"FO bhavcopy fetch failed for {d}: NSE unavailable")
        status_code = int(getattr(response, "status_code", 0))
        content = bytes(getattr(response, "content", b""))
        if status_code != 200 or content[:2] != b"PK":
            headers = getattr(response, "headers", {})
            content_type = (
                headers.get("content-type") if isinstance(headers, dict) else None
            )
            raise RuntimeError(
                f"FO bhavcopy fetch failed for {d}: HTTP {status_code}, "
                f"content-type={content_type!r}"
            )
        with zipfile.ZipFile(BytesIO(content)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fp:
                path.write_bytes(fp.read())
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # Detect a legacy-shaped frame (fresh or cached) and normalize to UDiff.
    if "INSTRUMENT" in df.columns:
        df = normalize_legacy_fo_bhavcopy(df)
    return df


# ── trading calendar ───────────────────────────────────────────────────────

_HOLIDAYS_URL = "https://www.nseindia.com/api/holiday-master?type=trading"

# NSE occasionally trades on a weekend (Union Budget day, Diwali Muhurat, or a
# disaster-recovery drill). Each date below was confirmed to have a real FO
# bhavcopy archive (HTTP-200 application/zip) via a live probe:
#   2024-01-20 (Sat) special live-trading session
#   2024-03-02 (Sat) special session
#   2024-05-18 (Sat) special / DR session
#   2025-02-01 (Sat) Union Budget session
#   2026-02-01 (Sun) Union Budget session
#   2023-11-12 (Sun) Diwali Muhurat trading
SPECIAL_TRADING_SESSIONS: frozenset[date] = frozenset(
    {
        date(2023, 11, 12),
        date(2024, 1, 20),
        date(2024, 3, 2),
        date(2024, 5, 18),
        date(2025, 2, 1),
        date(2026, 2, 1),
    }
)


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
        preserving the legacy weekend-only behaviour. Known NSE weekend
        special sessions are always trading days.
        """
        if d in SPECIAL_TRADING_SESSIONS:
            return True
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
