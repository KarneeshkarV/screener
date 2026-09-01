"""Current index constituent loaders used by rolling backtests."""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, field_validator

from screener.cache import is_fresh
from screener.resilience import call_with_resilience

LOG = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "universes"
_SP500_CHANGES_CACHE_TTL_SECONDS = 24 * 60 * 60

# Constituent and membership caches are keyed by ``as_of``, so freshness
# depends on which day the entry describes rather than on one flat TTL:
#
# * A past ``as_of`` describes a day that is over. Nothing upstream can make
#   that entry more correct -- for the non-reconstructable indices a refetch
#   would just return *today's* list, which is exactly the survivorship bias
#   the loader already warns about. Such entries are pinned. (S&P 500
#   point-in-time entries have a second, sharper check against the change log
#   in ``_sp500_pit_cache_matches_change_log``.)
# * A today-dated entry is a live snapshot, so it expires and is refetched -
#   an index add/remove takes effect intraday, and a long-running process must
#   not serve this morning's list all day.
#
# ``is_fresh`` reads a negative TTL as "always fresh, if the file exists", which
# is what pins the past-dated entries. This replaces a bare ``path.exists()``:
# the policy is now stated once, in the cache layer's own vocabulary.
_UNIVERSE_CACHE_TTL_SECONDS = 12 * 60 * 60
_UNIVERSE_CACHE_PINNED_TTL_SECONDS = -1.0

UniverseName = str


def _universe_cache_ttl(as_of: date) -> float:
    """TTL for a universe cache entry describing ``as_of``."""
    if as_of < date.today():
        return _UNIVERSE_CACHE_PINNED_TTL_SECONDS
    return _UNIVERSE_CACHE_TTL_SECONDS


@dataclass(frozen=True)
class UniverseDefinition:
    """Metadata and loader for a named current-constituent universe."""

    name: str
    market: str
    benchmark: str
    loader: Callable[[], tuple[list[str], str]]


@dataclass(frozen=True)
class UniverseSelection:
    """Resolved universe plus optional point-in-time selection policy.

    ``membership_windows`` contains half-open ``[start, end)`` eligibility
    intervals. ``dynamic_*`` fields request a lagged ADV-ranked universe that
    is recomputed by the rolling engine on each configured rebalance date.
    """

    name: str
    market: str
    benchmark: str | None
    symbols: tuple[str, ...]
    source: str
    membership_windows: tuple[tuple[str, date, date | None], ...] = ()
    dynamic_size: int | None = None
    dynamic_lookback: int = 60
    dynamic_rebalance: str = "monthly"


_UNIVERSE_REGISTRY: dict[str, UniverseDefinition] = {}


def register_universe(definition: UniverseDefinition) -> None:
    """Register a named universe, rejecting accidental duplicate providers."""
    key = definition.name.strip().lower()
    if not key:
        raise ValueError("universe name must not be empty")
    if key in _UNIVERSE_REGISTRY:
        raise ValueError(f"universe already registered: {key}")
    _UNIVERSE_REGISTRY[key] = definition


def available_universes() -> tuple[str, ...]:
    """Return built-in universe names in deterministic order."""
    return tuple(sorted(_UNIVERSE_REGISTRY))


def get_universe_definition(name: str) -> UniverseDefinition:
    try:
        return _UNIVERSE_REGISTRY[name.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(available_universes())
        raise ValueError(f"unknown universe: {name}; built-ins: {choices}") from exc


class UniverseSource(str, Enum):
    """Where a universe's symbols come from.

    These are deliberately NOT interchangeable — current TradingView liquidity,
    current index membership, and point-in-time index membership carry different
    survivorship characteristics, and user-supplied tickers/files bypass all of
    them. Each call site names its source explicitly so the choice is auditable.
    """

    TICKERS = "tickers"  # explicit user-supplied symbols
    FILE = "file"  # user newline-delimited file
    TV_LIQUIDITY = "tv_liquidity"  # current TradingView volume/price scan
    INDEX_CURRENT = "index_current"  # today's index membership
    INDEX_PIT = "index_pit"  # point-in-time index membership as of a date


class UniverseRequest(BaseModel):
    """A single, explicit request for a backtest/scan universe.

    ``tickers`` and ``file`` are honoured first (in that order) when present,
    regardless of ``source``; ``source`` selects the fetch used when neither is
    supplied. ``comment_prefixes`` controls which file lines are dropped as
    comments, ``limit`` caps a fetched list, and ``as_of``/``index_name`` drive
    the index sources. Symbol normalization is intentionally left to each fetch
    (TradingView names, ``.NS`` suffixes, etc.) so no source is silently coerced
    into another's convention.
    """

    model_config = ConfigDict(frozen=True)

    source: UniverseSource
    market: str = ""
    tickers: tuple[str, ...] | None = None
    file: str | None = None
    comment_prefixes: tuple[str, ...] = ()
    limit: int = 0
    as_of: date | None = None
    index_name: UniverseName | None = None


def parse_ticker_csv(raw: str | None) -> list[str]:
    """Split a comma-separated ticker string, trimming blanks."""
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def read_universe_file(
    path: str | Path, *, comment_prefixes: tuple[str, ...] = ()
) -> list[str]:
    """Read newline-separated symbols from ``path``.

    Blank lines are dropped; a line whose stripped form starts with any of
    ``comment_prefixes`` is treated as a comment and skipped. Raises the usual
    ``FileNotFoundError`` when ``path`` does not exist so callers can map it to
    their own error surface (e.g. a Click usage error).
    """
    lines = Path(path).read_text().splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if comment_prefixes and stripped.startswith(comment_prefixes):
            continue
        out.append(stripped)
    return out


def resolve_universe(
    request: UniverseRequest,
    *,
    tv_loader: Callable[[], list[str]] | None = None,
) -> list[str]:
    """Resolve ``request`` to a concrete symbol list.

    Explicit ``tickers`` then ``file`` win over ``source``. TradingView liquidity
    is delegated to ``tv_loader`` (which stays in the owning module so its scan
    remains individually testable/patchable); index membership is served by
    :func:`load_current_universe`, keeping the current-vs-point-in-time
    distinction explicit.
    """
    if request.tickers is not None:
        return list(request.tickers)
    if request.file is not None:
        return read_universe_file(
            request.file, comment_prefixes=request.comment_prefixes
        )
    if request.source is UniverseSource.TV_LIQUIDITY:
        if tv_loader is None:
            raise ValueError("TV liquidity source requires a tv_loader")
        return list(tv_loader())
    if request.source in (UniverseSource.INDEX_CURRENT, UniverseSource.INDEX_PIT):
        if request.index_name is None:
            raise ValueError(f"{request.source.value} source requires index_name")
        if request.source is UniverseSource.INDEX_CURRENT:
            return list(load_current_universe(request.index_name).symbols)
        return list(
            load_current_universe(request.index_name, as_of=request.as_of).symbols
        )
    raise ValueError(
        "no universe resolved: pass tickers or file for a "
        f"{request.source.value} request"
    )


def load_tv_liquidity_universe(
    market: str,
    universe_limit: int,
    *,
    cache_ttl: float | None = 900,
    refresh: bool = False,
) -> list[str]:
    """Fetch the most liquid tradable names for ``market`` from TradingView.

    This is the fetch behind :attr:`UniverseSource.TV_LIQUIDITY`: common stocks
    above the market's minimum close, ordered by volume. ``universe_limit`` of
    ``0`` means "broad market" and fetches the scanner's 5000-row ceiling.

    Lives here rather than in a screen's Click adapter so that domain modules
    can ask for a universe without importing a command module.
    """
    # Lazy: TradingView and the scanner's provider seam are only needed on the
    # fetch path, and importing them at module scope would pull the scan stack
    # into every consumer of this registry.
    from tradingview_screener import col

    from screener.markets import get_market
    from screener.scanner import scan

    price_floor = get_market(market).rs_breakout_min_close
    requested_limit = 5000 if universe_limit == 0 else universe_limit
    filters = [col("type") == "stock", col("close") >= price_floor]
    _total, df, _as_of = scan(
        market=market,
        filters=filters,
        limit=requested_limit,
        order_by="volume",
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return [str(t) for t in df["name"].dropna().tolist()]


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


def _universe_cache_is_fresh(name: UniverseName, as_of: date) -> bool:
    """True when the constituent cache for ``as_of`` is still within policy."""
    return is_fresh(_cache_path(name, as_of), _universe_cache_ttl(as_of))


def _read_cache(
    name: UniverseName, as_of: date
) -> tuple[Universe, bool, dict[str, str]] | None:
    """Return ``(universe, point_in_time, metadata)``, or ``None`` when absent.

    Freshness is *not* checked here: ``_universe_cache_is_fresh`` decides
    whether the entry may be served directly, while this reader also backs the
    stale-serve path taken when the upstream fetch fails.

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
    name = name.strip().lower()
    as_of = as_of or date.today()
    is_past = as_of < date.today()
    # Read the entry even when ``use_cache`` is off: a forced refresh still
    # prefers a stale list over no list at all if the provider is down, which
    # is how ``CachedProvider`` treats ``refresh=True``.
    cached = _read_cache(name, as_of)
    if use_cache and cached is not None and _universe_cache_is_fresh(name, as_of):
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
            # Warn on cache hits too - otherwise a second load of a past-date
            # universe written earlier this run (or a prior run) would silently
            # return the survivorship-biased set with no warning.
            if is_past and not point_in_time:
                _warn_not_point_in_time(name, as_of)
            return universe
    normalized_name = name.strip().lower()
    definition = (
        None if normalized_name == "sp500" else get_universe_definition(normalized_name)
    )
    try:
        if definition is None:
            symbols, source, point_in_time = _fetch_sp500_pit(
                as_of, use_cache=use_cache
            )
        else:
            symbols, source = definition.loader()
            point_in_time = not is_past
    except Exception as exc:
        if cached is None:
            raise
        universe, cached_point_in_time, _ = cached
        LOG.warning(
            "Serving stale %s universe cache from %s due to constituent "
            "fetch failure: %r",
            name,
            universe.cached_path,
            exc,
        )
        if is_past and not cached_point_in_time:
            _warn_not_point_in_time(name, as_of)
        return universe
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


NSE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


def parse_nse_index_csv(text: str, *, label: str, suffix: str = "") -> list[str]:
    """Return deduped symbols from an NSE index constituent CSV.

    Shared by the live constituent fetchers and by the Wayback backfill in
    :mod:`screener.universe_backfill`, so an archived snapshot is normalized
    into exactly the vocabulary the live loader would produce. Letting the two
    drift would make a point-in-time membership window name a symbol that never
    matches a price column, silently dropping the name from the backtest.
    """
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    symbol_col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    if symbol_col not in df.columns:
        raise RuntimeError(f"{label} CSV missing Symbol column")
    symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
    return _dedupe([f"{symbol}{suffix}" for symbol in symbols])


def _fetch_nifty50() -> tuple[list[str], str]:
    source = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    resp = call_with_resilience(
        "nse",
        "nifty50 constituents",
        lambda: requests.get(source, headers=NSE_REQUEST_HEADERS, timeout=30),
        fallback=None,
    )
    if resp is None:
        raise RuntimeError("Nifty 50 constituents unavailable")
    resp.raise_for_status()
    return parse_nse_index_csv(resp.text, label="Nifty 50 constituents"), source


_NIFTY500_SOURCE = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


# The earnings backtest consumes NSE tickers in yfinance form (``.NS``).
NSE_SYMBOL_SUFFIX = ".NS"


def _fetch_nifty500() -> tuple[list[str], str]:
    resp = call_with_resilience(
        "nse",
        "nifty500 constituents",
        lambda: requests.get(_NIFTY500_SOURCE, headers=NSE_REQUEST_HEADERS, timeout=30),
        fallback=None,
    )
    if resp is None:
        raise RuntimeError("Nifty 500 constituents unavailable")
    resp.raise_for_status()
    symbols = parse_nse_index_csv(
        resp.text,
        label="Nifty 500 constituents",
        suffix=NSE_SYMBOL_SUFFIX,
    )
    return symbols, _NIFTY500_SOURCE


_SENSEX_SOURCE = "https://en.wikipedia.org/wiki/List_of_BSE_SENSEX_companies"


def _fetch_sensex() -> tuple[list[str], str]:
    """Load current Sensex constituents in yfinance's ``.BO`` vocabulary.

    BSE's public constituent page is bot-protected and does not expose a stable
    documented API. This free adapter therefore uses the auditable Wikipedia
    table and records that source explicitly; custom/provider snapshots can be
    used when licensed BSE data is available.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = call_with_resilience(
        "wikipedia",
        "sensex constituents",
        lambda: requests.get(_SENSEX_SOURCE, headers=headers, timeout=30),
        fallback=None,
    )
    if resp is None:
        raise RuntimeError("Sensex constituents unavailable")
    resp.raise_for_status()
    from io import StringIO

    tables = pd.read_html(StringIO(resp.text))
    table = next((df for df in tables if "Symbol" in df.columns), None)
    if table is None:
        raise RuntimeError("Sensex constituents table missing Symbol column")
    if "Entry date" in table.columns:
        # The public table may stage an announced replacement before its
        # effective date, temporarily showing both the incoming and outgoing
        # name. Do not admit the future constituent early.
        entry_dates = pd.to_datetime(
            table["Entry date"].astype(str).str.replace(r"\[.*", "", regex=True),
            errors="coerce",
            dayfirst=True,
        )
        table = table[entry_dates.isna() | (entry_dates.dt.date <= date.today())]
    symbols = table["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
    pre_filter_count = len(symbols)
    LOG.debug("sensex table yielded %d rows before .BO filter", pre_filter_count)
    symbols = [symbol for symbol in symbols if symbol.endswith(".BO")]
    if not 25 <= len(set(symbols)) <= 35:
        raise RuntimeError(
            f"Sensex constituent count failed validation: {len(set(symbols))} "
            f"of .BO-suffixed rows (from {pre_filter_count} table rows); "
            "upstream table format may have changed"
        )
    return _dedupe(symbols), _SENSEX_SOURCE


def _dedupe(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(s for s in symbols if s))


def _membership_cache_path(name: UniverseName, as_of: date) -> Path:
    return CACHE_DIR / f"{name}_membership_{as_of.isoformat()}.json"


def _read_membership_cache(path: Path) -> dict[str, date | None] | None:
    """Read a membership cache entry ignoring freshness, or ``None``."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return {
            symbol: date.fromisoformat(added) if added else None
            for symbol, added in payload.items()
        }
    except (ValueError, OSError, AttributeError):
        LOG.debug("sp500 membership cache at %s unreadable; refetching", path)
        return None


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
    # Same policy as the constituent cache: a past-date entry is pinned, a
    # today-dated one expires, and a stale entry still beats no data when the
    # upstream table is unreachable.
    cached = _read_membership_cache(path)
    if use_cache and cached is not None and is_fresh(path, _universe_cache_ttl(as_of)):
        return cached

    try:
        df = _fetch_sp500_table()
    except Exception as exc:
        if cached is None:
            raise
        LOG.warning(
            "Serving stale sp500 membership cache from %s due to fetch failure: %r",
            path,
            exc,
        )
        return cached
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


def _load_definition_file(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read universe config {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()[:12]
    try:
        if path.suffix.lower() == ".toml":
            payload = tomllib.loads(raw.decode())
        elif path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw.decode()) or {}
        elif path.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            raise ValueError("universe config must be TOML, YAML, or JSON")
    except (
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"could not parse universe config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("universe config must contain a top-level mapping")
    return payload, digest


def _config_entry(name: str, config_path: Path) -> tuple[Mapping[str, Any], str]:
    payload, digest = _load_definition_file(config_path)
    entries = payload.get("universes", payload)
    if not isinstance(entries, Mapping) or name not in entries:
        raise ValueError(f"universe {name!r} not found in {config_path}")
    entry = entries[name]
    if not isinstance(entry, Mapping):
        raise ValueError(f"universe {name!r} must be a mapping")
    return entry, digest


def configured_universe_names(config_path: str | Path) -> tuple[str, ...]:
    """Return custom universe names declared by a config file."""
    payload, _ = _load_definition_file(Path(config_path))
    entries = payload.get("universes", payload)
    if not isinstance(entries, Mapping):
        raise ValueError("universes must be a mapping")
    return tuple(sorted(str(name) for name in entries))


def sync_universe_snapshot(
    name: str,
    *,
    output: str | Path,
    as_of: date | None = None,
    use_cache: bool = False,
) -> tuple[Path, bool, int]:
    """Append a complete dated snapshot when named membership has changed.

    Returns ``(path, changed, symbol_count)``. Repeated runs with unchanged
    membership are idempotent, making the function suitable for cron or CI.
    """
    effective = as_of or date.today()
    loaded = load_current_universe(name, as_of=effective, use_cache=use_cache)
    path = Path(output).expanduser()
    existing = pd.DataFrame(columns=["effective_date", "symbol"])
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        required = {"effective_date", "symbol"}
        if not required.issubset(existing.columns):
            raise ValueError(
                f"snapshot CSV {path} requires columns: effective_date, symbol"
            )
    normalized = tuple(dict.fromkeys(symbol.strip() for symbol in loaded.symbols))
    if not existing.empty:
        latest_date = str(existing["effective_date"].max())
        latest = tuple(
            existing.loc[existing["effective_date"] == latest_date, "symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if set(latest) == set(normalized):
            return path, False, len(normalized)
        if effective < date.fromisoformat(latest_date):
            raise ValueError(
                f"cannot append {effective.isoformat()} before latest snapshot {latest_date}"
            )
    addition = pd.DataFrame(
        {
            "effective_date": [effective.isoformat()] * len(normalized),
            "symbol": normalized,
        }
    )
    if not existing.empty:
        existing = existing[existing["effective_date"] != effective.isoformat()]
    combined = pd.concat([existing, addition], ignore_index=True)
    combined = combined.sort_values(["effective_date", "symbol"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    return path, True, len(normalized)


def _normalized_symbols(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("symbols must be a list")
    symbols = tuple(
        dict.fromkeys(
            str(value).strip().upper() for value in values if str(value).strip()
        )
    )
    if not symbols:
        raise ValueError("universe symbols must not be empty")
    return symbols


def _read_snapshot_source(source: str, *, base_dir: Path) -> pd.DataFrame:
    if source.startswith(("https://", "http://")):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        from io import StringIO

        return pd.read_csv(StringIO(resp.text))
    path = Path(source)
    if not path.is_absolute():
        path = base_dir / path
    return pd.read_csv(path)


def _snapshot_rows(
    entry: Mapping[str, Any], *, base_dir: Path, as_of: date
) -> list[tuple[date, tuple[str, ...]]]:
    raw_snapshots = entry.get("snapshots")
    rows: list[tuple[date, tuple[str, ...]]] = []
    if isinstance(raw_snapshots, Mapping):
        for raw_date, raw_symbols in raw_snapshots.items():
            snapshot_date = date.fromisoformat(str(raw_date))
            if snapshot_date <= as_of:
                rows.append((snapshot_date, _normalized_symbols(raw_symbols)))
    else:
        source = entry.get("path") or entry.get("url")
        if not source:
            raise ValueError("snapshot universe requires snapshots, path, or url")
        frame = _read_snapshot_source(str(source), base_dir=base_dir)
        date_col = next(
            (
                column
                for column in ("effective_date", "snapshot_date", "date")
                if column in frame.columns
            ),
            None,
        )
        if date_col is None or "symbol" not in frame.columns:
            raise ValueError(
                "snapshot CSV requires symbol and effective_date/snapshot_date columns"
            )
        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame[frame[date_col].notna()]
        for timestamp, group in frame.groupby(date_col, sort=True):
            snapshot_date = pd.Timestamp(cast(Any, timestamp)).date()
            if snapshot_date <= as_of:
                rows.append(
                    (snapshot_date, _normalized_symbols(group["symbol"].tolist()))
                )
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise ValueError("snapshot universe has no snapshots on or before --end")
    return rows


def _windows_from_snapshots(
    snapshots: list[tuple[date, tuple[str, ...]]],
) -> tuple[tuple[str, date, date | None], ...]:
    windows: list[tuple[str, date, date | None]] = []
    for index, (effective, symbols) in enumerate(snapshots):
        until = snapshots[index + 1][0] if index + 1 < len(snapshots) else None
        windows.extend((symbol, effective, until) for symbol in symbols)
    return tuple(windows)


def load_universe_selection(
    name: str,
    *,
    market: str,
    as_of: date,
    config_path: str | Path | None = None,
    use_cache: bool = True,
    dynamic_base: str | None = None,
    dynamic_size: int = 100,
    dynamic_lookback: int = 60,
    dynamic_rebalance: str = "monthly",
) -> UniverseSelection:
    """Resolve a built-in, custom, or lagged rule-based universe."""
    key = name.strip().lower()
    if key == "dynamic":
        base = dynamic_base or ("nifty500" if market == "india" else "sp500")
        loaded = load_current_universe(base, as_of=as_of, use_cache=use_cache)
        definition = get_universe_definition(base)
        return UniverseSelection(
            name="dynamic",
            market=market,
            benchmark=definition.benchmark,
            symbols=loaded.symbols,
            source=f"dynamic base={base}; {loaded.source}",
            dynamic_size=_validate_dynamic_size(dynamic_size),
            dynamic_lookback=_validate_dynamic_lookback(dynamic_lookback),
            dynamic_rebalance=_validate_rebalance(dynamic_rebalance),
        )
    if key in _UNIVERSE_REGISTRY:
        definition = get_universe_definition(key)
        if definition.market != market:
            raise ValueError(
                f"universe {key!r} belongs to market {definition.market!r}, not {market!r}"
            )
        loaded = load_current_universe(key, as_of=as_of, use_cache=use_cache)
        return UniverseSelection(
            name=key,
            market=definition.market,
            benchmark=definition.benchmark,
            symbols=loaded.symbols,
            source=f"{loaded.source}; cache={loaded.cached_path}",
        )
    if config_path is None:
        choices = ", ".join((*available_universes(), "dynamic"))
        raise ValueError(
            f"custom universe {key!r} requires --universe-config; built-ins: {choices}"
        )
    path = Path(config_path)
    entry, digest = _config_entry(key, path)
    configured_market = str(entry.get("market", market)).strip().lower()
    if configured_market != market:
        raise ValueError(
            f"universe {key!r} belongs to market {configured_market!r}, not {market!r}"
        )
    kind = str(entry.get("type", "static")).strip().lower()
    benchmark = str(entry["benchmark"]).strip() if entry.get("benchmark") else None
    source = f"config:{path}#{key}@sha256:{digest}"
    if kind == "static":
        symbols = _normalized_symbols(entry.get("symbols"))
        return UniverseSelection(key, market, benchmark, symbols, source)
    if kind == "snapshots":
        snapshots = _snapshot_rows(entry, base_dir=path.parent, as_of=as_of)
        symbols = tuple(
            dict.fromkeys(symbol for _, members in snapshots for symbol in members)
        )
        return UniverseSelection(
            key,
            market,
            benchmark,
            symbols,
            source,
            membership_windows=_windows_from_snapshots(snapshots),
        )
    if kind == "dynamic":
        if entry.get("symbols") is not None:
            symbols = _normalized_symbols(entry.get("symbols"))
        elif entry.get("path"):
            raw_path = Path(str(entry["path"]))
            if not raw_path.is_absolute():
                raw_path = path.parent / raw_path
            symbols = tuple(read_universe_file(raw_path, comment_prefixes=("#",)))
        else:
            base = str(entry.get("base", "")).strip().lower()
            if not base:
                raise ValueError("dynamic universe requires symbols, path, or base")
            symbols = load_current_universe(
                base, as_of=as_of, use_cache=use_cache
            ).symbols
        return UniverseSelection(
            key,
            market,
            benchmark,
            symbols,
            source,
            dynamic_size=_validate_dynamic_size(int(entry.get("size", dynamic_size))),
            dynamic_lookback=_validate_dynamic_lookback(
                int(entry.get("lookback", dynamic_lookback))
            ),
            dynamic_rebalance=_validate_rebalance(
                str(entry.get("rebalance", dynamic_rebalance))
            ),
        )
    raise ValueError(f"unsupported universe type {kind!r}")


def _validate_dynamic_size(value: int) -> int:
    if value <= 0:
        raise ValueError("dynamic universe size must be > 0")
    return value


def _validate_dynamic_lookback(value: int) -> int:
    if value < 2:
        raise ValueError("dynamic universe lookback must be >= 2")
    return value


def _validate_rebalance(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ValueError(
            "dynamic universe rebalance must be daily, weekly, monthly, or quarterly"
        )
    return normalized


register_universe(UniverseDefinition("sp500", "us", "SPY", _fetch_sp500))
register_universe(UniverseDefinition("nifty50", "india", "^NSEI", _fetch_nifty50))
register_universe(UniverseDefinition("nifty500", "india", "^NSEI", _fetch_nifty500))
register_universe(UniverseDefinition("sensex", "india", "^BSESN", _fetch_sensex))
