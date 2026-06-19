"""Current index constituent loaders used by rolling backtests."""

from __future__ import annotations

from datetime import date
import json
import logging
from pathlib import Path
from typing import Literal
import warnings

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
import requests

from screener.cache import is_fresh
from screener.resilience import call_with_resilience


LOG = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "universes"
_SP500_CHANGES_CACHE_TTL_SECONDS = 24 * 60 * 60
UniverseName = Literal["sp500", "nifty50"]


class Universe(BaseModel):
    name: UniverseName
    symbols: tuple[str, ...]
    source: str
    cached_path: Path

    model_config = ConfigDict(frozen=True)

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(symbol.strip() for symbol in value if symbol.strip())
        if not normalized:
            raise ValueError("symbols must not be empty")
        return normalized

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source must not be empty")
        return normalized


def _cache_path(name: UniverseName, as_of: date) -> Path:
    # ``.v2`` namespaces the cache format: it carries a ``point_in_time`` flag
    # and (for sp500) reflects the PIT reconstruction. Pre-fix ``.txt`` caches
    # held survivorship-biased "today's membership" under a past date and must
    # NOT be trusted by the PIT-aware loader, so they live at a different path.
    return CACHE_DIR / f"{name}_{as_of.isoformat()}.v2.txt"


def _write_cache(
    name: UniverseName,
    as_of: date,
    symbols: list[str],
    source: str,
    *,
    point_in_time: bool,
    metadata: dict[str, str] | None = None,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name, as_of)
    lines = [
        f"# universe={name}",
        f"# as_of={as_of.isoformat()}",
        f"# source={source}",
        f"# point_in_time={'true' if point_in_time else 'false'}",
        *(f"# {key}={value}" for key, value in (metadata or {}).items()),
        *symbols,
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_cache(
    name: UniverseName, as_of: date
) -> tuple[Universe, bool, dict[str, str]] | None:
    """Return ``(universe, point_in_time, metadata)`` or ``None`` to refetch.

    A cache file missing the ``point_in_time`` header was not written by the
    PIT-aware writer (or is corrupt) and is treated as a miss so we never serve
    a stale survivorship-biased list as if it were point-in-time.
    """
    path = _cache_path(name, as_of)
    if not path.exists():
        return None
    source = "cache"
    point_in_time: bool | None = None
    metadata: dict[str, str] = {}
    symbols: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# source="):
            source = line.split("=", 1)[1]
            continue
        if line.startswith("# point_in_time="):
            point_in_time = line.split("=", 1)[1].strip().lower() == "true"
            continue
        if line.startswith("#"):
            key, sep, value = line[1:].partition("=")
            if sep:
                metadata[key.strip()] = value.strip()
            continue
        symbols.append(line)
    if not symbols or point_in_time is None:
        return None
    universe = Universe(
        name=name, symbols=tuple(symbols), source=source, cached_path=path
    )
    return universe, point_in_time, metadata


def load_current_universe(
    name: UniverseName,
    *,
    as_of: date | None = None,
    use_cache: bool = True,
) -> Universe:
    """Return index membership as it stood on ``as_of``.

    Contract:

    * For indices that publish a machine-readable change log (currently only
      ``sp500`` via the Wikipedia "changes" table), the returned ``symbols``
      are *point-in-time*: the constituents that were actually in the index on
      ``as_of``. Reconstruction starts from today's members, ADDS BACK names
      removed after ``as_of`` (so delisted/removed tickers are included) and
      REMOVES names added after ``as_of`` (so post-``as_of`` IPOs are excluded).
    * For indices without reconstruction data (e.g. ``nifty50``), a past
      ``as_of`` cannot be honoured: the loader returns *today's* membership and
      emits a loud warning that the result is survivorship-biased and NOT
      point-in-time, so callers are not silently misled.
    """
    as_of = as_of or date.today()
    is_past = as_of < date.today()
    if use_cache:
        cached = _read_cache(name, as_of)
        if cached is not None:
            universe, point_in_time, metadata = cached
            if (
                name == "sp500"
                and is_past
                and not _sp500_pit_cache_matches_change_log(metadata)
            ):
                LOG.debug(
                    "%s cache for as_of=%s is stale relative to S&P change log",
                    name,
                    as_of.isoformat(),
                )
            else:
                # Warn on cache hits too — otherwise a second load of a past-date
                # universe written earlier this run (or a prior run) would silently
                # return the survivorship-biased set with no warning.
                if is_past and not point_in_time:
                    _warn_not_point_in_time(name, as_of)
                return universe
    if name == "sp500":
        symbols, source, point_in_time = _fetch_sp500_pit(as_of, use_cache=use_cache)
    elif name == "nifty50":
        symbols, source = _fetch_nifty50()
        point_in_time = not is_past
    else:
        raise ValueError(f"unknown universe: {name}")
    if is_past and not point_in_time:
        _warn_not_point_in_time(name, as_of)
    write_metadata = (
        _sp500_changes_cache_metadata() if name == "sp500" and is_past else None
    )
    path = _write_cache(
        name,
        as_of,
        symbols,
        source,
        point_in_time=point_in_time,
        metadata=write_metadata,
    )
    return Universe(name=name, symbols=tuple(symbols), source=source, cached_path=path)


def _warn_not_point_in_time(name: UniverseName, as_of: date) -> None:
    message = (
        f"{name} membership for as_of={as_of.isoformat()} could not be fully "
        f"reconstructed and is NOT point-in-time: it is survivorship-biased "
        f"(removed/delisted names may be missing and post-as_of additions may "
        f"be present). Treat historical results from this universe with caution."
    )
    LOG.warning(message)
    warnings.warn(message, stacklevel=3)


_SP500_SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _read_sp500_html() -> list[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = call_with_resilience(
        "wikipedia",
        "sp500 constituents",
        lambda: requests.get(_SP500_SOURCE, headers=headers, timeout=30),
        fallback=None,
    )
    if resp is None:
        raise RuntimeError("S&P 500 constituents unavailable")
    resp.raise_for_status()
    from io import StringIO

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("S&P 500 constituents table not found")
    return tables


def _fetch_sp500_table() -> pd.DataFrame:
    df = _read_sp500_html()[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("S&P 500 constituents table missing Symbol column")
    return df


def _normalize_sp500_symbols(raw: pd.Series) -> pd.Series:
    return raw.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)


def _fetch_sp500() -> tuple[list[str], str]:
    df = _fetch_sp500_table()
    symbols = _normalize_sp500_symbols(df["Symbol"].dropna()).tolist()
    return _dedupe(symbols), _SP500_SOURCE


def _changes_cache_path() -> Path:
    return CACHE_DIR / "sp500_changes.json"


def _sp500_changes_cache_metadata() -> dict[str, str] | None:
    path = _changes_cache_path()
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"sp500_changes_mtime_ns": str(stat.st_mtime_ns)}


def _sp500_pit_cache_matches_change_log(metadata: dict[str, str]) -> bool:
    path = _changes_cache_path()
    if not is_fresh(path, _SP500_CHANGES_CACHE_TTL_SECONDS):
        return False
    expected = _sp500_changes_cache_metadata()
    if expected is None:
        return False
    return metadata.get("sp500_changes_mtime_ns") == expected["sp500_changes_mtime_ns"]


def _fetch_sp500_changes() -> list[tuple[date, str, str]]:
    """Return the S&P 500 change log as ``(date, added_symbol, removed_symbol)``.

    Mirrors the ``pandas.read_html`` parsing used by :func:`_fetch_sp500_table`.
    The Wikipedia page exposes a second "Selected changes" table whose columns
    are a (Date, Added{Ticker,Security}, Removed{Ticker,Security}, Reason)
    MultiIndex. Either the added or removed ticker may be blank for a given row.
    """
    tables = _read_sp500_html()
    changes_df: pd.DataFrame | None = None
    for table in tables[1:]:
        cols = _flatten_columns(table.columns)
        if any("date" in c for c in cols) and any("added" in c for c in cols):
            changes_df = table
            break
    if changes_df is None:
        return []

    flat = [_flatten_columns([col])[0] for col in changes_df.columns]
    changes_df = changes_df.copy()
    changes_df.columns = flat

    date_col = next((c for c in flat if "date" in c), None)
    added_col = next((c for c in flat if "added" in c and "ticker" in c), None)
    removed_col = next((c for c in flat if "removed" in c and "ticker" in c), None)
    if date_col is None or (added_col is None and removed_col is None):
        return []

    parsed_dates = pd.to_datetime(changes_df[date_col], errors="coerce")
    rows: list[tuple[date, str, str]] = []
    for idx in range(len(changes_df)):
        ts = parsed_dates.iloc[idx]
        if pd.isna(ts):
            continue
        added = _clean_change_symbol(
            changes_df[added_col].iloc[idx] if added_col else ""
        )
        removed = _clean_change_symbol(
            changes_df[removed_col].iloc[idx] if removed_col else ""
        )
        if not added and not removed:
            continue
        rows.append((ts.date(), added, removed))
    return rows


def _flatten_columns(columns) -> list[str]:
    flat: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            text = " ".join(str(part) for part in col)
        else:
            text = str(col)
        flat.append(text.strip().lower())
    return flat


def _clean_change_symbol(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.upper().replace(".", "-")


def _load_sp500_changes(*, use_cache: bool = True) -> list[tuple[date, str, str]]:
    path = _changes_cache_path()
    if use_cache and is_fresh(path, _SP500_CHANGES_CACHE_TTL_SECONDS):
        try:
            payload = json.loads(path.read_text())
            return [
                (date.fromisoformat(d), added, removed) for d, added, removed in payload
            ]
        except (ValueError, OSError):
            LOG.debug("sp500 changes cache at %s unreadable; refetching", path)
    elif use_cache and path.exists():
        LOG.debug("sp500 changes cache at %s is stale; refetching", path)

    changes = _fetch_sp500_changes()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [[d.isoformat(), added, removed] for d, added, removed in changes],
            indent=0,
        )
    )
    return changes


def _fetch_sp500_pit(
    as_of: date, *, use_cache: bool = True
) -> tuple[list[str], str, bool]:
    """Reconstruct point-in-time S&P 500 membership as of ``as_of``.

    Start from current members, then walk the change log backwards: for every
    change dated strictly after ``as_of`` undo it — ADD BACK the removed name
    and REMOVE the added name. This yields the constituents that were actually
    in the index on ``as_of``, including names removed/delisted since.

    Returns ``(symbols, source, point_in_time)``. ``point_in_time`` is ``False``
    when the reconstruction cannot be trusted for a past ``as_of`` — either the
    change log is empty/unparseable (we fall back to today's members) or it does
    not reach back as far as ``as_of`` (earlier adds/removes are unknown). In
    both cases the caller warns rather than silently returning a biased list.
    """
    current, _ = _fetch_sp500()
    members = dict.fromkeys(current)
    changes = _load_sp500_changes(use_cache=use_cache)
    for changed_on, added, removed in changes:
        if changed_on <= as_of:
            continue
        if added:
            members.pop(added, None)
        if removed:
            members[removed] = None

    point_in_time = True
    if as_of < date.today():
        if not changes:
            # No change log at all: members is just today's list -> biased.
            point_in_time = False
        elif as_of < min(changed_on for changed_on, _, _ in changes):
            # The log does not extend back to as_of, so adds/removes before the
            # earliest logged change are missing and the set is incomplete.
            point_in_time = False
    return list(members), _SP500_SOURCE, point_in_time


def _fetch_nifty50() -> tuple[list[str], str]:
    source = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = call_with_resilience(
        "nse",
        "nifty50 constituents",
        lambda: requests.get(source, headers=headers, timeout=30),
        fallback=None,
    )
    if resp is None:
        raise RuntimeError("Nifty 50 constituents unavailable")
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    symbol_col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    if symbol_col not in df.columns:
        raise RuntimeError("Nifty 50 constituents CSV missing Symbol column")
    symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
    return _dedupe(symbols), source


def _dedupe(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(s for s in symbols if s))


def _membership_cache_path(name: UniverseName, as_of: date) -> Path:
    return CACHE_DIR / f"{name}_membership_{as_of.isoformat()}.json"


def load_sp500_membership(
    *,
    as_of: date | None = None,
    use_cache: bool = True,
) -> dict[str, date | None]:
    """Return symbol → index "Date added" for current S&P 500 members.

    A ``None`` value means the source table has no parseable date for that
    symbol; callers should treat such symbols as always eligible. Only
    current members are covered — companies removed from the index are not
    reconstructed (their delisted price history is unavailable upstream), so
    this reduces survivorship bias rather than eliminating it.
    """
    as_of = as_of or date.today()
    path = _membership_cache_path("sp500", as_of)
    if use_cache and path.exists():
        try:
            payload = json.loads(path.read_text())
            return {
                symbol: date.fromisoformat(added) if added else None
                for symbol, added in payload.items()
            }
        except (ValueError, OSError):
            LOG.debug("sp500 membership cache at %s unreadable; refetching", path)

    df = _fetch_sp500_table()
    if "Date added" not in df.columns:
        raise RuntimeError("S&P 500 constituents table missing 'Date added' column")
    symbols = _normalize_sp500_symbols(df["Symbol"].astype(str))
    added = pd.to_datetime(df["Date added"], errors="coerce")
    membership: dict[str, date | None] = {}
    for symbol, added_ts in zip(symbols, added):
        if not symbol or symbol in membership:
            continue
        membership[symbol] = added_ts.date() if pd.notna(added_ts) else None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                symbol: added.isoformat() if added else None
                for symbol, added in membership.items()
            },
            indent=0,
        )
    )
    return membership
