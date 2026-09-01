"""Reconstruct point-in-time index membership from web-archive crawls.

The built-in NSE universes (:mod:`screener.universes`) can only fetch *today's*
constituent list - NSE publishes no machine-readable membership history - so a
backtest run over them is survivorship-biased and ``load_current_universe``
says so out loud for any past ``as_of``.

This module closes that gap without inventing data. NSE's constituent CSV lives
at a stable URL that the Internet Archive has crawled for years, so every crawl
is a dated, verbatim copy of the membership NSE published that day. Collecting
the distinct crawls yields the ``symbol,effective_date`` snapshot CSV that a
``type: snapshots`` custom universe consumes, and the rolling engine turns those
into half-open eligibility windows.

Two properties of this reconstruction matter for interpreting a backtest:

* **It is lookahead-free.** A snapshot is dated at the crawl that observed it,
  which is on or after the day NSE published that membership, never before. No
  name becomes eligible earlier than it was really in the index.
* **It is coarse, and errs toward late.** Crawls are irregular, so a membership
  change is attributed to the first crawl that saw it rather than to its true
  effective date. An addition therefore enters late (conservative), and a
  deletion leaves late - the deleted name stays eligible until the next crawl.
  It was still a real, tradeable listing over that stretch, so this is a
  resolution limit rather than a bias toward names that turned out well. Read
  the dates :func:`list_archived_snapshots` returns and check for gaps before
  trusting a window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from screener.universes import (
    NSE_REQUEST_HEADERS,
    NSE_SYMBOL_SUFFIX,
    parse_nse_index_csv,
)

LOG = logging.getLogger(__name__)

_CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
# ``id_`` asks the replay server for the stored bytes rather than a rewritten
# page, so the CSV comes back exactly as NSE served it on the crawl date.
_REPLAY_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{url}"


@dataclass(frozen=True)
class BackfillSource:
    """How to find and parse the archived constituent CSVs for one universe.

    ``urls`` lists every host the index has published the same list under over
    the years. NSE moved between ``www1``, ``archives`` and ``nsearchives``,
    and niftyindices.com serves its own copy; the Internet Archive crawled each
    separately, so all of them have to be swept to recover the full history.

    ``min_symbols`` is the smallest member count that is plausible for this
    index. A truncated capture that still parses would otherwise be written as
    a real snapshot and erase most of the index for its whole window, so the
    floor has to be per-index rather than a global 1.
    """

    label: str
    urls: tuple[str, ...]
    suffix: str = ""
    min_symbols: int = 1


BACKFILL_SOURCES: dict[str, BackfillSource] = {
    "nifty50": BackfillSource(
        label="Nifty 50 constituents",
        urls=(
            "archives.nseindia.com/content/indices/ind_nifty50list.csv",
            "nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
            "www1.nseindia.com/content/indices/ind_nifty50list.csv",
            "www.nseindia.com/content/indices/ind_nifty50list.csv",
            "niftyindices.com/IndexConstituent/ind_nifty50list.csv",
            "www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        ),
        min_symbols=40,
    ),
    "nifty500": BackfillSource(
        label="Nifty 500 constituents",
        urls=(
            "archives.nseindia.com/content/indices/ind_nifty500list.csv",
            "nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "www1.nseindia.com/content/indices/ind_nifty500list.csv",
            "www.nseindia.com/content/indices/ind_nifty500list.csv",
            "niftyindices.com/IndexConstituent/ind_nifty500list.csv",
            "www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        ),
        suffix=NSE_SYMBOL_SUFFIX,
        min_symbols=400,
    ),
}


def backfillable_universes() -> tuple[str, ...]:
    """Return the universe names this module can reconstruct."""
    return tuple(sorted(BACKFILL_SOURCES))


def get_backfill_source(name: str) -> BackfillSource:
    """Return the archived-CSV source for ``name``."""
    try:
        return BACKFILL_SOURCES[name.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(backfillable_universes())
        raise ValueError(
            f"no archived constituent source for universe {name!r}; "
            f"available: {choices}"
        ) from exc


@dataclass(frozen=True)
class ArchivedSnapshot:
    """One distinct archived copy of a constituent CSV."""

    observed: date
    timestamp: str
    url: str
    digest: str

    @property
    def replay_url(self) -> str:
        return _REPLAY_TEMPLATE.format(timestamp=self.timestamp, url=self.url)


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of a backfill run."""

    path: Path
    snapshots: tuple[ArchivedSnapshot, ...]
    symbols: tuple[str, ...]
    rows: int
    warnings: tuple[str, ...]


def _parse_timestamp(timestamp: str) -> date | None:
    try:
        return datetime.strptime(timestamp[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _query_cdx(
    url: str, session: requests.Session, *, timeout: float
) -> list[list[str]]:
    response = session.get(
        _CDX_ENDPOINT,
        params={
            "url": url,
            "output": "json",
            "fl": "timestamp,original,digest,statuscode",
            "filter": "statuscode:200",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    # The first row is the column header; an archive with no captures at all
    # returns an empty body rather than a header-only table.
    return list(payload[1:]) if payload else []


def list_archived_snapshots(
    name: str,
    *,
    since: date | None = None,
    until: date | None = None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> tuple[tuple[ArchivedSnapshot, ...], tuple[str, ...]]:
    """Return ``(snapshots, warnings)`` for one universe, oldest first.

    Crawls are deduped by observation date and by consecutive content digest:
    a date can carry only one membership, and re-crawling unchanged content
    says nothing new. Both rules are deliberately local - the day keeps its
    last crawl rather than its first, and a digest is compared only against the
    crawl before it - so neither a same-day change nor a membership that
    reverts to an earlier state is collapsed away.

    A mirror whose archive index cannot be read yields a warning rather than an
    error - the history is assembled from whichever mirrors do answer, and a
    caller that silently lost one should be able to see that it did.
    """
    source = get_backfill_source(name)
    owned = session is None
    session = session or requests.Session()
    warnings: list[str] = []
    candidates: list[ArchivedSnapshot] = []
    try:
        for url in source.urls:
            try:
                rows = _query_cdx(url, session, timeout=timeout)
            except (requests.RequestException, ValueError) as exc:
                warnings.append(f"archive index unavailable for {url}: {exc!r}")
                continue
            for row in rows:
                if len(row) < 4:
                    continue
                timestamp, original, digest, _status = row[:4]
                observed = _parse_timestamp(str(timestamp))
                if observed is None:
                    continue
                if since is not None and observed < since:
                    continue
                if until is not None and observed > until:
                    continue
                candidates.append(
                    ArchivedSnapshot(
                        observed=observed,
                        timestamp=str(timestamp),
                        url=str(original),
                        digest=str(digest),
                    )
                )
    finally:
        if owned:
            session.close()
    candidates.sort(key=lambda snapshot: snapshot.timestamp)
    # One snapshot per calendar day, because a date can carry only one
    # membership. Keep the *last* crawl of the day: when two mirrors disagree
    # because NSE published a change partway through, the later crawl is the
    # one that saw the change, and keeping the earlier one would hold removed
    # names eligible and new names out until the next crawl.
    by_date: dict[date, ArchivedSnapshot] = {}
    for snapshot in candidates:
        by_date[snapshot.observed] = snapshot
    kept: list[ArchivedSnapshot] = []
    for _observed, snapshot in sorted(by_date.items()):
        # Drop a crawl only when it repeats the digest immediately before it.
        # Deduping against every digest ever seen would swallow a genuine
        # A -> B -> A revert, and the window would then claim B forever.
        if kept and snapshot.digest == kept[-1].digest:
            continue
        kept.append(snapshot)
    return tuple(kept), tuple(warnings)


def fetch_snapshot_members(
    snapshot: ArchivedSnapshot,
    source: BackfillSource,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> tuple[str, ...]:
    """Download one archived CSV and normalize it to universe symbols."""
    owned = session is None
    session = session or requests.Session()
    try:
        response = session.get(
            snapshot.replay_url, headers=NSE_REQUEST_HEADERS, timeout=timeout
        )
        response.raise_for_status()
        symbols = parse_nse_index_csv(
            response.text, label=source.label, suffix=source.suffix
        )
    finally:
        if owned:
            session.close()
    return tuple(symbols)


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["effective_date", "symbol"])
    existing = pd.read_csv(path, dtype=str)
    required = {"effective_date", "symbol"}
    if not required.issubset(existing.columns):
        raise ValueError(
            f"snapshot CSV {path} requires columns: effective_date, symbol"
        )
    return existing[["effective_date", "symbol"]].dropna()


def backfill_snapshots(
    name: str,
    *,
    output: str | Path,
    since: date | None = None,
    until: date | None = None,
    min_symbols: int | None = None,
    replace_existing: bool = False,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> BackfillResult:
    """Write a ``symbol,effective_date`` snapshot CSV from archived crawls.

    Rows already in ``output`` are never touched, so this composes with
    ``universes sync``, which appends today's live membership going forward.
    A live snapshot is a first-hand observation and an archived crawl of the
    same day is not, so a date the file already carries is left alone and
    reported as a conflict. Pass ``replace_existing`` to overwrite those dates
    instead, which is what you want after a bad backfill wrote wrong rows.

    ``min_symbols`` rejects a crawl that parsed into an implausibly short list;
    it defaults to the floor the source declares. A truncated or error-page
    capture would otherwise erase most of the index for the entire window that
    snapshot covers, which reads as a plausible result rather than as the fetch
    failure it is.
    """
    source = get_backfill_source(name)
    if min_symbols is None:
        min_symbols = source.min_symbols
    if min_symbols < 1:
        raise ValueError(f"min_symbols must be at least 1, got {min_symbols}")
    snapshots, warnings = list_archived_snapshots(
        name, since=since, until=until, session=session, timeout=timeout
    )
    all_warnings = list(warnings)
    owned = session is None
    session = session or requests.Session()
    accepted: list[tuple[ArchivedSnapshot, tuple[str, ...]]] = []
    previous: frozenset[str] | None = None
    try:
        for snapshot in snapshots:
            try:
                members = fetch_snapshot_members(
                    snapshot, source, session=session, timeout=timeout
                )
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                all_warnings.append(
                    f"skipped crawl {snapshot.timestamp} of {snapshot.url}: {exc!r}"
                )
                continue
            if len(members) < min_symbols:
                all_warnings.append(
                    f"skipped crawl {snapshot.timestamp} of {snapshot.url}: "
                    f"{len(members)} symbols is below min_symbols {min_symbols}"
                )
                continue
            current = frozenset(members)
            # Distinct bytes can still be the same membership (a reordered or
            # recolumned CSV). Only a real change earns a new window boundary.
            if previous is not None and current == previous:
                continue
            previous = current
            accepted.append((snapshot, members))
    finally:
        if owned:
            session.close()

    if not accepted:
        detail = f"; {all_warnings[0]}" if all_warnings else ""
        raise RuntimeError(f"no usable archived snapshots for {name}{detail}")

    path = Path(output).expanduser()
    existing = _read_existing(path)
    existing_dates = set(existing["effective_date"]) if not existing.empty else set()
    if existing_dates and not replace_existing:
        conflicting = [
            snapshot
            for snapshot, _members in accepted
            if snapshot.observed.isoformat() in existing_dates
        ]
        for snapshot in conflicting:
            all_warnings.append(
                f"kept the existing snapshot for {snapshot.observed.isoformat()} "
                f"rather than the crawl {snapshot.timestamp} of {snapshot.url}; "
                "pass replace_existing to overwrite it"
            )
        accepted = [
            (snapshot, members)
            for snapshot, members in accepted
            if snapshot.observed.isoformat() not in existing_dates
        ]

    addition = pd.DataFrame(
        {
            "effective_date": [
                snapshot.observed.isoformat()
                for snapshot, members in accepted
                for _symbol in members
            ],
            "symbol": [symbol for _snapshot, members in accepted for symbol in members],
        }
    )
    if not existing.empty and not addition.empty:
        existing = existing[
            ~existing["effective_date"].isin(set(addition["effective_date"]))
        ]
    frames = [frame for frame in (existing, addition) if not frame.empty]
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["effective_date", "symbol"])
    )
    combined = combined.drop_duplicates(subset=["effective_date", "symbol"])
    combined = combined.sort_values(["effective_date", "symbol"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    LOG.info("backfilled %d snapshots for %s into %s", len(accepted), name, path)
    return BackfillResult(
        path=path,
        snapshots=tuple(snapshot for snapshot, _members in accepted),
        symbols=tuple(dict.fromkeys(combined["symbol"].tolist())),
        rows=len(combined),
        warnings=tuple(all_warnings),
    )
