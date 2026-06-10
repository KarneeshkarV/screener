"""Current index constituent loaders used by rolling backtests."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
import requests

from screener.resilience import call_with_resilience


CACHE_DIR = Path.home() / ".screener" / "universes"
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
    return CACHE_DIR / f"{name}_{as_of.isoformat()}.txt"


def _write_cache(
    name: UniverseName, as_of: date, symbols: list[str], source: str
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name, as_of)
    lines = [
        f"# universe={name}",
        f"# as_of={as_of.isoformat()}",
        f"# source={source}",
        *symbols,
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_cache(name: UniverseName, as_of: date) -> Universe | None:
    path = _cache_path(name, as_of)
    if not path.exists():
        return None
    source = "cache"
    symbols: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# source="):
            source = line.split("=", 1)[1]
            continue
        if line.startswith("#"):
            continue
        symbols.append(line)
    if not symbols:
        return None
    return Universe(name=name, symbols=tuple(symbols), source=source, cached_path=path)


def load_current_universe(
    name: UniverseName,
    *,
    as_of: date | None = None,
    use_cache: bool = True,
) -> Universe:
    as_of = as_of or date.today()
    if use_cache:
        cached = _read_cache(name, as_of)
        if cached is not None:
            return cached
    if name == "sp500":
        symbols, source = _fetch_sp500()
    elif name == "nifty50":
        symbols, source = _fetch_nifty50()
    else:
        raise ValueError(f"unknown universe: {name}")
    path = _write_cache(name, as_of, symbols, source)
    return Universe(name=name, symbols=tuple(symbols), source=source, cached_path=path)


_SP500_SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_sp500_table() -> pd.DataFrame:
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
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("S&P 500 constituents table missing Symbol column")
    return df


def _normalize_sp500_symbols(raw: pd.Series) -> pd.Series:
    return raw.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)


def _fetch_sp500() -> tuple[list[str], str]:
    df = _fetch_sp500_table()
    symbols = _normalize_sp500_symbols(df["Symbol"].dropna()).tolist()
    return _dedupe(symbols), _SP500_SOURCE


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
            pass

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
