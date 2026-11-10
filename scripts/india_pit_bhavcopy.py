#!/usr/bin/env python
"""Survivorship-free India universe and adjusted prices from NSE bhavcopy.

The yfinance path has two point-in-time gaps for India.
It has no bars for delisted or merged names, and a renamed name (ZOMATO ->
ETERNAL) splits into two tickers, so the old one has no history.
This pipeline rebuilds both from NSE's own daily cash bhavcopy:

    uv run python -m scripts.india_pit_bhavcopy download --start 2013-01-01
    uv run python -m scripts.india_pit_bhavcopy parse
    uv run python -m scripts.india_pit_bhavcopy universe
    uv run python -m scripts.india_pit_bhavcopy actions
    uv run python -m scripts.india_pit_bhavcopy prices
    uv run python -m scripts.india_pit_bhavcopy check

Entities: two symbols are one company when they share an ISIN or when NSE's
symbol-change file links them. The entity takes its latest symbol, so it
matches today's yfinance ticker.

Universe: on the first session of each month, rank EQ/BE names by the median
traded value of the prior 126 sessions (nothing from the snapshot day), with
trades on at least 80% of those sessions. Keep the top N.

Prices: raw bhavcopy OHLCV, back-adjusted with NSE corporate actions for
splits, bonuses, rights (TERP), demergers (ex-date opening gap) and cash
dividends (``adj_close`` only). Read them with ``BhavcopyPriceFetcher``.

This is research data, not a vendor-certified series.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SCREENER_HOME = Path.home() / ".screener"
DATA_DIR = SCREENER_HOME / "india_pit"
OLD_BHAV_DIR = SCREENER_HOME / "bhavcopy_cm"
UDIFF_BHAV_DIR = SCREENER_HOME / "bhavcopy_udiff"
YAHOO_CACHE_DIR = SCREENER_HOME / "prices"
UNIVERSE_DIR = ROOT / "data" / "universes"
NIFTY500_SNAPSHOTS = UNIVERSE_DIR / "nifty500_pit_snapshots.csv"
NIFTY500_ENTITY = UNIVERSE_DIR / "nifty500_pit_entity.csv.gz"

UDIFF_START = date(2024, 7, 8)
OLD_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{year}/{mon}/cm{day}{mon}{year}bhav.csv.zip"
)
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)
SYMBOL_CHANGE_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"
ACTIONS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept": "application/json,text/html",
    "Accept-Language": "en-US,en;q=0.9",
}

#: Renames that NSE's symbol-change file does not list (checked by hand).
MANUAL_RENAMES = (
    ("CASTROL", "CASTROLIND", pd.Timestamp("2014-03-14")),
    ("SUBEX", "SUBEXLTD", pd.Timestamp("2020-11-05")),
)
TURNOVER_WINDOW = 126
MIN_TRADED_SHARE = 0.8
#: Clean split / bonus ratios accepted when no source explains a gap.
SPLIT_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0)
#: NSE price bands cap a normal day at 20%, so a 1/3+ overnight gap is an event.
EVENT_GAP = 1.45
MIN_INFERRED_SPLIT_PRICE = 20.0
NUM = r"(\d+(?:\.\d+)?)"
#: NSE publishes each bhavcopy the same evening; a 404 this much later is a holiday.
HOLIDAY_SETTLE = timedelta(days=7)


# -- download ---------------------------------------------------------------


def bhavcopy_target(day: date) -> tuple[str, Path]:
    """Return the archive URL and local path for one session's cash bhavcopy."""
    if day < UDIFF_START:
        mon = day.strftime("%b").upper()
        name = f"cm{day:%d}{mon}{day:%Y}bhav.csv.zip"
        url = OLD_URL.format(year=f"{day:%Y}", mon=mon, day=f"{day:%d}")
        return url, OLD_BHAV_DIR / name
    name = f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    return UDIFF_URL.format(ymd=f"{day:%Y%m%d}"), UDIFF_BHAV_DIR / name


def download_bhavcopy(day: date, session: requests.Session) -> str:
    """Fetch one bhavcopy zip once; a ``.404`` marker records a holiday.

    A marker is final only when the 404 came ``HOLIDAY_SETTLE`` after the
    session, so a bhavcopy that was not yet published is fetched again.
    """
    url, path = bhavcopy_target(day)
    missing = path.with_name(path.name + ".404")
    if path.exists() and path.stat().st_size:
        return "cached"
    if (
        missing.exists()
        and date.fromtimestamp(missing.stat().st_mtime) >= day + HOLIDAY_SETTLE
    ):
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, headers=HEADERS, timeout=30)
    if response.status_code == 404:
        missing.touch()
        return "holiday"
    response.raise_for_status()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(response.content)
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            tmp.unlink()
            raise ValueError(f"corrupt bhavcopy zip: {url}")
    tmp.rename(path)
    return "downloaded"


def download(start: date, end: date, workers: int) -> None:
    # Every calendar day: NSE holds rare weekend sessions (2024-01-20), and
    # the archive's 404 marks the holidays.
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    session = requests.Session()

    def one(day: date) -> str:
        for attempt in range(3):
            try:
                return download_bhavcopy(day, session)
            except (requests.RequestException, ValueError, zipfile.BadZipFile):
                time.sleep(2 + 3 * attempt)
        return "failed"

    with ThreadPoolExecutor(workers) as pool:
        status = pd.Series(list(pool.map(one, days))).value_counts()
    print(status.to_dict())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = session.get(SYMBOL_CHANGE_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    (DATA_DIR / "symbolchange.csv").write_bytes(response.content)


# -- parse ------------------------------------------------------------------


def read_bhavcopy_zip(path: Path) -> pd.DataFrame:
    """Normalize one old-format (to 2024-07) or UDiFF cash bhavcopy."""
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open(archive.namelist()[0]))
    frame.columns = [str(column).strip() for column in frame.columns]
    if "TckrSymb" in frame.columns:
        mapping = {
            "TradDt": "date",
            "TckrSymb": "symbol",
            "SctySrs": "series",
            "ISIN": "isin",
            "OpnPric": "open",
            "HghPric": "high",
            "LwPric": "low",
            "ClsPric": "close",
            "TtlTradgVol": "volume",
            "TtlTrfVal": "value",
        }
        dates = pd.to_datetime(frame["TradDt"], format="%Y-%m-%d")
    else:
        mapping = {
            "TIMESTAMP": "date",
            "SYMBOL": "symbol",
            "SERIES": "series",
            "ISIN": "isin",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "TOTTRDQTY": "volume",
            "TOTTRDVAL": "value",
        }
        # A few 2020 files write the year with two digits ("13-Jul-20").
        stamp = frame["TIMESTAMP"].astype(str).str.strip()
        dates = pd.to_datetime(stamp, format="%d-%b-%Y", errors="coerce").fillna(
            pd.to_datetime(stamp, format="%d-%b-%y", errors="coerce")
        )
    out = frame[list(mapping)].rename(columns=mapping)
    out["date"] = dates
    for column in ("symbol", "series", "isin"):
        out[column] = out[column].astype(str).str.strip()
    return out


def parse() -> None:
    files = sorted(OLD_BHAV_DIR.glob("*.zip")) + sorted(UDIFF_BHAV_DIR.glob("*.zip"))
    with ProcessPoolExecutor(8) as pool:
        frames = list(pool.map(read_bhavcopy_zip, files, chunksize=20))
    bhav = pd.concat(frames, ignore_index=True)
    bhav = bhav[bhav["series"].isin(["EQ", "BE"]) & bhav["isin"].str.startswith("INE")]
    bhav = bhav.drop_duplicates(["date", "symbol", "series"]).sort_values(
        ["date", "symbol"]
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bhav.to_parquet(DATA_DIR / "bhav.parquet", index=False)
    print(
        f"{len(files)} files, {len(bhav)} rows, {bhav.date.min().date()} to "
        f"{bhav.date.max().date()}, {bhav.symbol.nunique()} symbols"
    )


# -- entities and universe ----------------------------------------------------


def link_entities(
    listings: pd.DataFrame, renames: Iterable[tuple[str, str, pd.Timestamp]]
) -> dict[str, str]:
    """Map each symbol to its company's latest symbol.

    ``listings`` has ``date``, ``symbol`` and ``isin``. Symbols sharing an ISIN
    and each known ``(old, new, effective)`` rename are one company. A rename
    links only when the old symbol traded before its effective date, so a
    later, unrelated listing that reuses the old symbol stays separate.
    """
    parent: dict[str, str] = {}

    def find(symbol: str) -> str:
        parent.setdefault(symbol, symbol)
        while parent[symbol] != symbol:
            parent[symbol] = parent[parent[symbol]]
            symbol = parent[symbol]
        return symbol

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    for symbol in listings["symbol"].unique():
        find(symbol)
    for _, group in listings[["isin", "symbol"]].drop_duplicates().groupby("isin"):
        symbols = group["symbol"].tolist()
        for symbol in symbols[1:]:
            union(symbols[0], symbol)
    known = set(parent)
    first_seen = listings.groupby("symbol")["date"].min()
    for old, new, effective in renames:
        if old in known and new in known and first_seen[old] < effective:
            union(old, new)
    last_seen = listings.groupby("symbol")["date"].max()
    groups: dict[str, list[str]] = {}
    for symbol in known:
        groups.setdefault(find(symbol), []).append(symbol)
    entity_of: dict[str, str] = {}
    for members in groups.values():
        name = max(members, key=lambda symbol: (last_seen[symbol], symbol))
        entity_of.update(dict.fromkeys(members, name))
    return entity_of


def entity_sessions(bhav: pd.DataFrame, entity_of: Mapping[str, str]) -> pd.DataFrame:
    """One row per entity and session: the busiest listing that day."""
    frame = bhav.assign(entity=bhav["symbol"].map(entity_of))
    frame = frame.sort_values("value").drop_duplicates(["date", "entity"], keep="last")
    return frame.sort_values(["entity", "date"]).reset_index(drop=True)


def turnover_snapshots(
    sessions: pd.DataFrame,
    top: int,
    *,
    window: int = TURNOVER_WINDOW,
    min_traded_share: float = MIN_TRADED_SHARE,
) -> pd.DataFrame:
    """Monthly top-``top`` entities by prior median traded value.

    A snapshot dated on a month's first session reads only earlier sessions,
    and only once a full ``window`` of sessions exists. Months with fewer than 90% of ``top`` eligible names are skipped.
    """
    value = sessions.pivot(index="date", columns="entity", values="value")
    need = int(np.ceil(min_traded_share * window))
    median = value.rolling(window, min_periods=need).median().shift(1)
    traded = value.notna().rolling(window).sum().shift(1)
    median = median.where(traded >= need)
    days = value.index
    firsts = days.to_series().groupby([days.year, days.month]).min()
    rows: list[tuple[date, str]] = []
    for day in firsts:
        ranked = median.loc[day].dropna().sort_values(ascending=False).head(top)
        if len(ranked) < 0.9 * top:
            continue
        rows.extend((day.date(), f"{entity}.NS") for entity in ranked.index)
    return pd.DataFrame(rows, columns=["effective_date", "symbol"])


def remap_snapshots(
    snapshots: pd.DataFrame, entity_of: Mapping[str, str]
) -> pd.DataFrame:
    """Rename snapshot symbols to entities so old tickers find their bars."""
    base = snapshots["symbol"].str.removesuffix(".NS")
    out = snapshots.assign(symbol=base.map(entity_of).fillna(base) + ".NS")
    return out.drop_duplicates().reset_index(drop=True)


def read_renames(path: Path) -> list[tuple[str, str, pd.Timestamp]]:
    changes = pd.read_csv(path, encoding="latin1")
    changes.columns = [str(column).strip() for column in changes.columns]
    old = changes["OLD SYMBOL"].astype(str).str.strip()
    new = changes["NEW SYMBOL"].astype(str).str.strip()
    effective = pd.to_datetime(
        changes["EFFECTIVE DATE"].astype(str).str.strip(), format="%d-%b-%Y"
    )
    return [*zip(old, new, effective), *MANUAL_RENAMES]


def universe_path(top: int) -> Path:
    return UNIVERSE_DIR / f"india_turnover_top{top}.csv.gz"


def build_universe(tops: list[int]) -> None:
    bhav = pd.read_parquet(DATA_DIR / "bhav.parquet")
    entity_of = link_entities(bhav, read_renames(DATA_DIR / "symbolchange.csv"))
    sessions = entity_sessions(bhav, entity_of)
    sessions.to_parquet(DATA_DIR / "bhav_entity.parquet", index=False)
    pd.Series(entity_of, name="entity").rename_axis("symbol").sort_index().to_csv(
        DATA_DIR / "entity_map.csv"
    )
    for top in tops:
        out = turnover_snapshots(sessions, top)
        out.to_csv(universe_path(top), index=False)
        print(
            f"top {top}: {out.effective_date.nunique()} snapshots "
            f"{out.effective_date.min()} to {out.effective_date.max()}, "
            f"{out.symbol.nunique()} names"
        )
    nifty = remap_snapshots(pd.read_csv(NIFTY500_SNAPSHOTS), entity_of)
    nifty.to_csv(NIFTY500_ENTITY, index=False)
    known = set(sessions["entity"])
    missing = sorted(set(nifty.symbol.str.removesuffix(".NS")) - known)
    print(
        f"nifty500 entity map: {nifty.symbol.nunique()} names, no bhavcopy: {missing}"
    )


def universe_entities(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names |= set(pd.read_csv(path)["symbol"].str.removesuffix(".NS"))
    return names


def default_universes() -> list[Path]:
    return [*sorted(UNIVERSE_DIR.glob("india_turnover_top*.csv.gz")), NIFTY500_ENTITY]


def entity_members() -> dict[str, list[str]]:
    mapping = pd.read_csv(DATA_DIR / "entity_map.csv")
    return mapping.groupby("entity")["symbol"].apply(list).to_dict()


# -- corporate actions -------------------------------------------------------


def actions_path(symbol: str) -> Path:
    return DATA_DIR / "ca" / f"{symbol.replace('/', '_')}.json"


def fetch_actions(universes: list[Path]) -> None:
    """Cache NSE corporate actions for every symbol of each universe entity."""
    members = entity_members()
    symbols = sorted(
        {
            s
            for entity in universe_entities(universes)
            for s in members.get(entity, [entity])
        }
    )
    todo = [symbol for symbol in symbols if not actions_path(symbol).exists()]
    print(f"{len(symbols)} symbols, {len(todo)} to fetch", flush=True)
    actions_path("x").parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.get(ACTIONS_REFERER, headers=HEADERS, timeout=20)
    for symbol in todo:
        for attempt in range(4):
            try:
                response = session.get(
                    ACTIONS_URL,
                    params={
                        "index": "equities",
                        "symbol": symbol,
                        "from_date": "01-01-2010",
                        "to_date": date.today().strftime("%d-%m-%Y"),
                    },
                    headers={**HEADERS, "Referer": ACTIONS_REFERER},
                    timeout=20,
                )
                rows = response.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 + 3 * attempt)
                session = requests.Session()
                session.get(ACTIONS_REFERER, headers=HEADERS, timeout=20)
        else:
            print("failed", symbol, flush=True)
            continue
        actions_path(symbol).write_text(json.dumps(rows))
        time.sleep(0.15)


@dataclass(frozen=True)
class ActionTerms:
    """Machine-readable terms of one NSE corporate-action subject line."""

    split: float = 1.0
    dividend: float = 0.0
    rights: tuple[float, float, float] | None = None
    demerger: bool = False


def parse_action_subject(subject: str) -> ActionTerms:
    """Read share-count, dividend, rights and demerger terms from a subject.

    ``split`` is the share multiplier: 2.0 for a 1:1 bonus or a Rs 10 -> Rs 5
    split. ``rights`` is ``(new, held, premium)`` for "Rights new:held @ Prem"
    or "at par"; the premium is NaN when the price is missing or in tranches.
    Debenture, preference and DVR actions do not touch the equity line.
    """
    text = re.sub(r"\s+", " ", subject.replace("Valus", "Value")).lower()
    if any(word in text for word in ("debenture", "ncrps", "dvr", "preference")):
        return ActionTerms()
    split = 1.0
    bonus = re.search(rf"bonus\s*-?\s*(?:issue\s*)?{NUM}\s*:\s*{NUM}", text)
    if bonus:
        split *= (float(bonus[1]) + float(bonus[2])) / float(bonus[2])
    face = re.search(
        rf"(?:split|splt|sub-division).*?(?:(?:from|frm)\s*)?(?:r[se]\.?\s*)?{NUM}"
        rf".*?to\s*(?:face value\s*|r[se]\.?\s*)?{NUM}",
        text,
    )
    if face and float(face[2]) > 0:
        split *= float(face[1]) / float(face[2])
    dividend = sum(
        float(match[1])
        for match in re.finditer(rf"dividend[^/]*?r[se]\.?\s*-?\s*{NUM}", text)
    )
    rights = None
    rights_match = re.search(
        rf"rights?(?:\s*issue)?(?:\s*eq)?\s*[-:]?\s*{NUM}\s*:\s*{NUM}([^/]*)", text
    )
    if rights_match:
        # NaN premium: the price is missing or split across tranches. The
        # date still marks a rights issue, so the gap is not read as a split.
        premium = float("nan")
        if len(re.findall(rf"{NUM}\s*:\s*{NUM}", text)) == 1:
            terms = rights_match[3]
            stated = re.search(
                rf"(?:@|\bat\b)\s*prem(?:ium)?\.?\s*(?:of\s*)?r[se]\.?\s*{NUM}",
                terms,
            )
            if stated:
                premium = float(stated[1])
            elif re.search(r"(?:@|\bat\b)\s*par\b", terms):
                premium = 0.0
        rights = (float(rights_match[1]), float(rights_match[2]), premium)
    demerger = any(word in text for word in ("demerger", "arrangement", "arangement"))
    return ActionTerms(split, dividend, rights, demerger)


@dataclass(frozen=True)
class CorporateAction:
    ex_date: pd.Timestamp
    terms: ActionTerms
    face_value: float


def load_actions(symbols: Iterable[str]) -> list[CorporateAction]:
    """Equity-line actions of every symbol, deduplicated on (ex-date, subject)."""
    rows: dict[tuple[str, str], dict] = {}
    for symbol in symbols:
        path = actions_path(symbol)
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row.get("series") in (None, "EQ", "BE", "BZ", "-"):
                rows[(row["exDate"], row["subject"].strip())] = row
    actions = []
    for (ex_date, subject), row in rows.items():
        try:
            day = pd.to_datetime(ex_date, format="%d-%b-%Y")
        except ValueError:
            continue
        try:
            face = float(row.get("faceVal") or 0)
        except ValueError:
            face = 0.0
        actions.append(CorporateAction(day, parse_action_subject(subject), face))
    return actions


def yahoo_split_events(symbols: Iterable[str]) -> dict[pd.Timestamp, float]:
    """Split events from the local yfinance cache, a second split source."""
    events: dict[pd.Timestamp, float] = {}
    for symbol in symbols:
        path = YAHOO_CACHE_DIR / f"{symbol}.NS__full_turnover_v3.parquet"
        if not path.exists():
            continue
        splits = pd.read_parquet(path, columns=["stock_splits"])["stock_splits"]
        splits.index = pd.to_datetime(splits.index)
        events.update({day: float(ratio) for day, ratio in splits[splits > 0].items()})
    return events


# -- adjusted prices ----------------------------------------------------------


def _session_position(dates: pd.DatetimeIndex, day: pd.Timestamp) -> int | None:
    """First session on or within a week after ``day``; never the first bar."""
    position = int(dates.searchsorted(day))
    if position <= 0 or position >= len(dates) or (dates[position] - day).days > 7:
        return None
    return position


def adjust_prices(
    sessions: pd.DataFrame,
    actions: Iterable[CorporateAction],
    yahoo_splits: Mapping[pd.Timestamp, float],
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, str, float, str]]]:
    """Back-adjust one entity's raw bhavcopy sessions.

    ``sessions`` is date-indexed with raw ``open/high/low/close/volume/value``.
    OHLC and volume are adjusted for share-count events only. ``adj_close``
    adds dividends, rights and demergers (total return), the contract
    ``apply_full_price_adjustment`` expects. ``dollar_volume`` stays the raw
    traded value. Returns the frame and an audit log of every decision.
    """
    log: list[tuple[pd.Timestamp, str, float, str]] = []
    raw = sessions[["open", "high", "low", "close", "volume"]].astype(float)
    move = np.log(raw["close"]).diff()
    bad_print = (move.abs() > 0.5) & ((move + move.shift(-1)).abs() < 0.1)
    log.extend((day, "bad_print_dropped", 0.0, "") for day in raw.index[bad_print])
    raw = raw[~bad_print]
    value = sessions.loc[~bad_print, "value"].astype(float)
    dates = raw.index
    close = raw["close"].to_numpy()
    open_ = raw["open"].to_numpy()
    gap = np.r_[np.nan, close[:-1] / open_[1:]]

    nse_split: dict[int, float] = {}
    dividends: dict[int, float] = {}
    rights: dict[int, tuple[float, float, float, float]] = {}
    demergers: set[int] = set()
    for action in actions:
        position = _session_position(dates, action.ex_date)
        if position is None:
            continue
        terms = action.terms
        if terms.split != 1.0:
            nse_split[position] = nse_split.get(position, 1.0) * terms.split
        if terms.dividend > 0:
            dividends[position] = dividends.get(position, 0.0) + terms.dividend
        if terms.demerger and terms.split == 1.0:
            demergers.add(position)
        if terms.rights is not None:
            rights[position] = (*terms.rights, action.face_value)
    yahoo: dict[int, float] = {}
    for day, ratio in yahoo_splits.items():
        position = _session_position(dates, day)
        if position is not None:
            yahoo[position] = ratio

    split = np.ones(len(dates))
    # NSE subjects sometimes misstate the ratio, so take whichever source
    # sits closest to the observed gap and require the gap to confirm it.
    for position in sorted(set(nse_split) | set(yahoo)):
        candidates = [r for r in (nse_split.get(position), yahoo.get(position)) if r]
        best = min(candidates, key=lambda ratio: abs(np.log(gap[position] / ratio)))
        note = f"nse={nse_split.get(position)} yf={yahoo.get(position)} gap={gap[position]:.2f}"
        if 0.6 * best <= gap[position] <= 1.6 * best:
            split[position] *= best
            log.append((dates[position], "split", best, note))
        else:
            log.append((dates[position], "split_skipped", best, note))
    for position in np.flatnonzero((gap >= EVENT_GAP) | (gap <= 1 / EVENT_GAP)):
        if split[position] != 1.0 or position in demergers or position in rights:
            continue
        ratio = min(SPLIT_RATIOS, key=lambda r: abs(np.log(gap[position] / r)))
        tolerance = 0.01 if ratio < 2 else 0.04
        if (
            close[position - 1] >= MIN_INFERRED_SPLIT_PRICE
            and abs(np.log(gap[position] / ratio)) <= tolerance
        ):
            split[position] *= ratio
            log.append(
                (dates[position], "split_inferred", ratio, f"gap={gap[position]:.3f}")
            )
        else:
            log.append(
                (dates[position], "gap_unexplained", round(gap[position], 3), "")
            )

    total_return = np.ones(len(dates))
    for position, amount in dividends.items():
        prior = close[position - 1] / split[position]
        if amount < 0.99 * prior:
            total_return[position] *= 1 - amount / prior
    for position, (new, held, premium, face) in rights.items():
        prior = close[position - 1] / split[position]
        issue = (face / split[position] if face else 0.0) + premium
        if split[position] != 1.0 or not 0 < issue < prior:
            log.append(
                (dates[position], "rights_skipped", 1.0, f"{new}:{held}@{issue}")
            )
            continue
        terp = (held * prior + new * issue) / (new + held)
        total_return[position] *= terp / prior
        log.append(
            (dates[position], "rights", round(terp / prior, 4), f"{new}:{held}@{issue}")
        )
    for position in demergers:
        retained = open_[position] / close[position - 1]
        if split[position] == 1.0 and 0.02 < retained < 0.95:
            total_return[position] *= retained
            log.append((dates[position], "demerger", round(retained, 3), ""))

    # An event on bar t rescales every bar before t.
    later_split = np.r_[np.cumprod(split[::-1])[::-1][1:], 1.0]
    later_return = np.r_[np.cumprod(total_return[::-1])[::-1][1:], 1.0]
    out = raw.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = raw[column] / later_split
    out["volume"] = raw["volume"] * later_split
    out["adj_close"] = out["close"] * later_return
    out["dollar_volume"] = value
    out.index.name = "Date"
    return out, log


def build_prices(universes: list[Path]) -> None:
    sessions = pd.read_parquet(DATA_DIR / "bhav_entity.parquet")
    members = entity_members()
    out_dir = DATA_DIR / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    by_entity = dict(tuple(sessions.groupby("entity")))
    entities = sorted(universe_entities(universes) & set(by_entity))
    for entity in entities:
        symbols = members.get(entity, [entity])
        frame, log = adjust_prices(
            by_entity[entity].set_index("date").sort_index(),
            load_actions(symbols),
            yahoo_split_events(symbols),
        )
        frame.to_parquet(out_dir / f"{entity.replace('/', '_')}.parquet")
        log_rows.extend((entity, day.date(), *rest) for day, *rest in log)
    log_frame = pd.DataFrame(
        log_rows, columns=["entity", "date", "kind", "factor", "note"]
    )
    log_frame.to_csv(DATA_DIR / "adjust_log.csv", index=False)
    print(
        f"{len(entities)} entities written; {log_frame.kind.value_counts().to_dict()}"
    )


# -- check --------------------------------------------------------------------


def check(universe: Path) -> None:
    """Compare total return with the yfinance cache and list large daily moves."""
    members = pd.read_csv(universe, parse_dates=["effective_date"])
    spans = members.groupby("symbol")["effective_date"].agg(["min", "max"])
    compare = []
    moves = []
    for path in sorted((DATA_DIR / "prices").glob("*.parquet")):
        ticker = f"{path.stem}.NS"
        if ticker not in spans.index:
            continue
        ours = pd.read_parquet(path)
        first, last = spans.loc[ticker]
        window = ours.loc[first - pd.Timedelta(days=400) : last + pd.Timedelta(days=60)]
        step = window["adj_close"] / window["adj_close"].shift(1)
        moves.extend(
            (path.stem, day.date(), round(ratio, 3))
            for day, ratio in step[(step < 0.7) | (step > EVENT_GAP)].items()
        )
        yahoo_path = YAHOO_CACHE_DIR / f"{ticker}__full_turnover_v3.parquet"
        if not yahoo_path.exists():
            continue
        yahoo = pd.read_parquet(yahoo_path, columns=["adj_close"])["adj_close"]
        yahoo.index = pd.to_datetime(yahoo.index)
        joined = pd.concat(
            [ours["adj_close"], yahoo], axis=1, keys=["ours", "yf"], sort=True
        ).dropna()
        if len(joined) > 250:
            growth = joined.iloc[-1] / joined.iloc[0]
            compare.append((path.stem, len(joined), growth["ours"] / growth["yf"]))
    result = pd.DataFrame(compare, columns=["entity", "sessions", "tr_ratio"])
    result.to_csv(DATA_DIR / "yf_compare.csv", index=False)
    pd.DataFrame(moves, columns=["entity", "date", "ratio"]).to_csv(
        DATA_DIR / "large_moves.csv", index=False
    )
    error = np.log(result["tr_ratio"]).abs()
    print(
        f"{len(result)} names vs yfinance: total return within 2% "
        f"{(error < 0.02).mean():.1%}, within 10% {(error < 0.1).mean():.1%}, "
        f"off by >25%: {(error > 0.25).sum()}"
    )
    print(f"in-universe daily moves below -30% or above +45%: {len(moves)}")


# -- engine integration -------------------------------------------------------


class BhavcopyPriceFetcher:
    """``PriceFetcher`` over the adjusted bhavcopy frames.

    ``.NS`` tickers come from ``prices/<ENTITY>.parquet``; a ``.NS`` ticker
    with no file returns an empty frame rather than a survivor-only yfinance
    series. Other tickers (``^NSEI``) go to ``fallback``.
    """

    def __init__(self, fallback, directory: Path = DATA_DIR / "prices") -> None:
        from screener.backtester.price_frames import apply_full_price_adjustment

        self.fallback = fallback
        self.directory = directory
        self._adjust = apply_full_price_adjustment

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        rest = []
        for ticker in tickers:
            if not ticker.endswith(".NS"):
                rest.append(ticker)
                continue
            path = (
                self.directory
                / f"{ticker.removesuffix('.NS').replace('/', '_')}.parquet"
            )
            if not path.exists():
                out[ticker] = pd.DataFrame()
                continue
            frame = pd.read_parquet(path)
            frame.index = pd.to_datetime(frame.index).normalize()
            out[ticker] = self._adjust(
                frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
            )
        if rest:
            out.update(self.fallback.fetch(rest, start, end))
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "download", help="NSE cash bhavcopy zips and symbol changes"
    )
    fetch.add_argument("--start", type=date.fromisoformat, default=date(2013, 1, 1))
    fetch.add_argument("--end", type=date.fromisoformat, default=date.today())
    fetch.add_argument("--workers", type=int, default=4)
    commands.add_parser("parse", help="zips -> bhav.parquet (EQ/BE, INE ISINs)")
    universe = commands.add_parser(
        "universe", help="entities and monthly turnover snapshots"
    )
    universe.add_argument("--top", type=int, action="append")
    for name in ("actions", "prices"):
        sub = commands.add_parser(name)
        sub.add_argument("--universe", type=Path, action="append")
    checker = commands.add_parser("check", help="compare with the yfinance cache")
    checker.add_argument("--universe", type=Path, default=universe_path(750))
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.start, args.end, args.workers)
    elif args.command == "parse":
        parse()
    elif args.command == "universe":
        build_universe(args.top or [500, 750])
    elif args.command == "actions":
        fetch_actions(args.universe or default_universes())
    elif args.command == "prices":
        build_prices(args.universe or default_universes())
    else:
        check(args.universe)


if __name__ == "__main__":
    main()
