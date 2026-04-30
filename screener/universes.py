"""Current index constituent loaders used by rolling backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests


CACHE_DIR = Path.home() / ".screener" / "universes"
UniverseName = Literal["sp500", "nifty50"]


@dataclass(frozen=True)
class Universe:
    name: UniverseName
    symbols: tuple[str, ...]
    source: str
    cached_path: Path


def _cache_path(name: UniverseName, as_of: date) -> Path:
    return CACHE_DIR / f"{name}_{as_of.isoformat()}.txt"


def _write_cache(name: UniverseName, as_of: date, symbols: list[str], source: str) -> Path:
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


def _fetch_sp500() -> tuple[list[str], str]:
    source = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = requests.get(source, headers=headers, timeout=30)
    resp.raise_for_status()
    from io import StringIO

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("S&P 500 constituents table not found")
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("S&P 500 constituents table missing Symbol column")
    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return _dedupe(symbols), source


def _fetch_nifty50() -> tuple[list[str], str]:
    source = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = requests.get(source, headers=headers, timeout=30)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    symbol_col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    if symbol_col not in df.columns:
        raise RuntimeError("Nifty 50 constituents CSV missing Symbol column")
    symbols = (
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    return _dedupe(symbols), source


def _dedupe(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(s for s in symbols if s))
