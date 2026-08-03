"""US SEC filings reader backed by Financial Modeling Prep (FMP).

Two FMP endpoints power this module, both behind the shared
``screener.fmp`` transport and the ``screener.providers`` cache +
resilience seam (mirroring ``screener.institutional`` / ``screener.insiders``):

* ``{v3}/sec_filings/{symbol}?type=&page=`` — the recent SEC filings index
  (10-K, 10-Q, 8-K, ...). Each row carries FMP's misspelled ``fillingDate``
  plus ``acceptedDate``, ``type``, ``link`` (the EDGAR index page) and
  ``finalLink`` (the primary document). Rows are paginated newest-first.
* ``{v4}/financial-reports-json?symbol=&year=&period=`` — a filed 10-K/10-Q
  rendered as XBRL R-file sections. The JSON object carries ``symbol``,
  ``period`` and ``year`` keys, then one key per report section (e.g.
  "CONSOLIDATED BALANCE SHEETS", "Revenue"). Section names are R-file titles:
  they may be truncated to ~30 chars, carry ``_2`` de-duplication suffixes and
  "(Tables)" variants. Each section value is a list of single-key dicts mapping
  a row label to its list of cell values (the columns).

Parsing/matching is kept pure and separate from I/O so tests can stub the
fetchers. Only ``us`` is supported (these are SEC filings); the command layer
rejects other markets.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from screener import fmp
from screener.providers import CachedProvider, ProviderSpec

logger = logging.getLogger(__name__)

# FMP filings index: 24h cache; filed reports are immutable once accepted, so
# their section JSON is cached for a week. Both share the "fmp" circuit breaker.
_FMP_FILINGS_PROVIDER = CachedProvider(
    ProviderSpec(provider="fmp", namespace="fmp_filings", ttl_seconds=86400)
)
_FMP_REPORT_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="fmp", namespace="fmp_financial_reports", ttl_seconds=7 * 86400
    )
)

# Safety cap on the filings-index pagination so bad data can't loop forever.
_MAX_PAGES = 10

VALID_PERIODS = ("FY", "Q1", "Q2", "Q3")


# ── models ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Filing:
    """One SEC filing index row (dates parsed for display)."""

    symbol: str
    type: str
    filing_date: date | None
    accepted_date: date | None
    link: str
    final_link: str


@dataclass(frozen=True)
class SectionRow:
    """A single report row: a label and its list of cell values (columns)."""

    label: str
    values: list[str]


@dataclass(frozen=True)
class ReportSection:
    """A named report section (an XBRL R-file) with its rows."""

    name: str
    rows: list[SectionRow]

    def raw(self) -> list[dict[str, list[str]]]:
        """Reconstruct the section's raw ``[{label: [values]}, ...]`` shape."""
        return [{row.label: row.values} for row in self.rows]


@dataclass(frozen=True)
class FinancialReport:
    """A parsed 10-K/10-Q: identity plus its ordered sections."""

    symbol: str
    period: str
    year: int | None
    sections: list[ReportSection]

    def section_names(self) -> list[str]:
        return [section.name for section in self.sections]


# ── pure parsing ─────────────────────────────────────────────────────────────


def parse_date(value: Any) -> date | None:
    """Parse FMP's ``"2025-10-31 00:00:00"`` / ``"2025-10-31"`` into a ``date``.

    Returns ``None`` for missing, blank or unparseable values rather than
    raising, so a malformed row degrades to a blank cell instead of a crash.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_filings(payload: Any) -> list[Filing]:
    """Parse a raw ``sec_filings`` payload into :class:`Filing` records.

    Non-list payloads (e.g. FMP's ``{"Error Message": ...}``) yield an empty
    list. FMP's ``fillingDate`` misspelling is read as the filing date.
    """
    if not isinstance(payload, list):
        return []
    filings: list[Filing] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        filings.append(
            Filing(
                symbol=str(row.get("symbol") or ""),
                type=str(row.get("type") or ""),
                filing_date=parse_date(row.get("fillingDate")),
                accepted_date=parse_date(row.get("acceptedDate")),
                link=str(row.get("link") or ""),
                final_link=str(row.get("finalLink") or ""),
            )
        )
    return filings


def _parse_section(name: str, value: Any) -> ReportSection:
    """Parse one section value (a list of single-key ``{label: [cells]}`` dicts).

    Robust to the messy real shapes: non-list values, non-dict entries, dicts
    with multiple keys, and scalar (non-list) cell values are all coerced into
    ``SectionRow``s rather than raising.
    """
    rows: list[SectionRow] = []
    entries = value if isinstance(value, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for label, cells in entry.items():
            if isinstance(cells, list):
                values = [str(cell) for cell in cells]
            else:
                values = [str(cells)]
            rows.append(SectionRow(label=str(label), values=values))
    return ReportSection(name=str(name), rows=rows)


def parse_report(payload: Any) -> FinancialReport | None:
    """Parse a ``financial-reports-json`` object into a :class:`FinancialReport`.

    Returns ``None`` when the payload is not a dict or carries no report
    sections (FMP returns ``{}`` / an error object for missing reports). The
    ``symbol``/``period``/``year`` metadata keys are stripped from the section
    list; every remaining key becomes a section in payload order.
    """
    if not isinstance(payload, dict):
        return None
    meta = {"symbol", "period", "year"}
    sections = [
        _parse_section(key, value) for key, value in payload.items() if key not in meta
    ]
    if not sections:
        return None
    year_raw = payload.get("year")
    try:
        year = int(year_raw) if year_raw is not None else None
    except (TypeError, ValueError):
        year = None
    return FinancialReport(
        symbol=str(payload.get("symbol") or ""),
        period=str(payload.get("period") or ""),
        year=year,
        sections=sections,
    )


def match_sections(report: FinancialReport, pattern: str) -> list[ReportSection]:
    """Case-insensitive substring match of ``pattern`` against section names.

    Truncated names (``"CONSOLIDATED STATEMENTS OF OPER"``), ``_2`` suffixes and
    "(Tables)" variants are matched as-is, so a short substring like ``"balance"``
    finds ``"CONSOLIDATED BALANCE SHEETS"`` and every ``"... (Parenthetical)"``
    sibling. A blank pattern matches nothing.
    """
    needle = pattern.strip().lower()
    if not needle:
        return []
    return [s for s in report.sections if needle in s.name.lower()]


# ── I/O behind the provider seam ─────────────────────────────────────────────


def _fetch_filings_raw(
    symbol: str,
    *,
    api_key: str,
    filing_type: str | None,
    limit: int,
    cache_ttl: float | None,
    refresh: bool,
) -> list[dict]:
    """Fetch (paginated) raw filing rows for ``symbol`` through the FMP seam."""

    def _fetch() -> list[dict]:
        client = fmp.FmpClient(api_key, timeout=20)
        path = f"sec_filings/{urllib.parse.quote(symbol)}"
        collected: list[dict] = []
        for page in range(_MAX_PAGES):
            params: dict[str, Any] = {"page": page}
            if filing_type:
                params["type"] = filing_type
            payload = client.get(path, params)
            if not isinstance(payload, list) or not payload:
                break
            collected.extend(row for row in payload if isinstance(row, dict))
            if len(collected) >= limit:
                break
        else:
            logger.warning(
                "FMP filings for %s may be truncated at %d pages",
                symbol,
                _MAX_PAGES,
            )
        return collected

    return _FMP_FILINGS_PROVIDER.fetch(
        ("sec_filings", symbol, filing_type or "", int(limit)),
        _fetch,
        refresh=refresh,
        fallback=[],
        ttl_seconds=cache_ttl,
        operation=f"sec filings {symbol}",
    )


def load_filings(
    symbol: str,
    *,
    api_key: str,
    filing_type: str | None = None,
    limit: int = 20,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> list[Filing]:
    """Load up to ``limit`` recent SEC filings for one US ``symbol``."""
    raw = _fetch_filings_raw(
        symbol,
        api_key=api_key,
        filing_type=filing_type,
        limit=limit,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return parse_filings(raw)[:limit]


def _fetch_report_raw(
    symbol: str,
    *,
    api_key: str,
    year: int,
    period: str,
    cache_ttl: float | None,
    refresh: bool,
) -> Any:
    """Fetch the raw ``financial-reports-json`` object through the FMP seam."""

    def _fetch() -> Any:
        client = fmp.FmpClient(api_key, base_url=fmp.FMP_V4_BASE_URL, timeout=20)
        return client.get(
            "financial-reports-json",
            {"symbol": symbol, "year": year, "period": period},
        )

    return _FMP_REPORT_PROVIDER.fetch(
        ("financial_reports_json", symbol, int(year), period),
        _fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"financial report {symbol} {year} {period}",
    )


def load_report(
    symbol: str,
    *,
    api_key: str,
    year: int,
    period: str = "FY",
    cache_ttl: float | None = 7 * 86400,
    refresh: bool = False,
) -> FinancialReport | None:
    """Load and parse a 10-K/10-Q section report for one US ``symbol``."""
    payload = _fetch_report_raw(
        symbol,
        api_key=api_key,
        year=year,
        period=period,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return parse_report(payload)


__all__ = [
    "VALID_PERIODS",
    "Filing",
    "FinancialReport",
    "ReportSection",
    "SectionRow",
    "load_filings",
    "load_report",
    "match_sections",
    "parse_date",
    "parse_filings",
    "parse_report",
]
