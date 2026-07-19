"""Point-in-time historical F&O lot sizes from a user-maintained CSV.

NSE's ``fo_mktlots.csv`` is current-only and historical lot sizes are not
reliably downloadable, so we never fabricate history. Users who need correct
pre-2024 rupee notionals maintain an optional CSV at
``~/.screener/lot_sizes_history.csv`` with columns::

    symbol,effective_from,lot_size

``effective_from`` is an ISO date; each row records the lot that took effect on
that date. :func:`historical_lot_sizes` returns, for a given ``as_of``, the
latest lot per symbol whose ``effective_from`` is on or before ``as_of``. A
missing or malformed file yields an empty mapping.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

LOT_HISTORY_PATH = Path.home() / ".screener" / "lot_sizes_history.csv"

_REQUIRED_COLUMNS = frozenset({"symbol", "effective_from", "lot_size"})

# Cache the parsed file keyed by its mtime so edits are picked up automatically
# without re-reading on every call.
_ParsedHistory = dict[str, list[tuple[date, float]]]
_CACHE: dict[tuple[str, float], _ParsedHistory] = {}


def _load(path: Path) -> _ParsedHistory:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = (str(path), mtime)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if not _REQUIRED_COLUMNS <= set(frame.columns):
        return {}
    parsed: _ParsedHistory = {}
    for symbol_raw, eff_raw, lot_raw in zip(
        frame["symbol"], frame["effective_from"], frame["lot_size"]
    ):
        symbol = str(symbol_raw).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        eff = pd.to_datetime(eff_raw, errors="coerce")
        lot = pd.to_numeric(lot_raw, errors="coerce")
        if pd.isna(eff) or pd.isna(lot) or lot <= 0:
            continue
        parsed.setdefault(symbol, []).append((eff.date(), float(lot)))
    for entries in parsed.values():
        entries.sort(key=lambda item: item[0])
    _CACHE.clear()
    _CACHE[key] = parsed
    return parsed


def historical_lot_sizes(as_of: date, *, path: Path | None = None) -> dict[str, float]:
    """Return {SYMBOL: lot} effective on or before ``as_of`` per symbol.

    The most recent ``effective_from`` not after ``as_of`` wins. Symbols whose
    earliest record starts after ``as_of`` are omitted. Missing/malformed file
    yields ``{}``.
    """
    parsed = _load(path or LOT_HISTORY_PATH)
    result: dict[str, float] = {}
    for symbol, entries in parsed.items():
        latest: float | None = None
        for eff, lot in entries:
            if eff <= as_of:
                latest = lot
            else:
                break
        if latest is not None:
            result[symbol] = latest
    return result


__all__ = ["LOT_HISTORY_PATH", "historical_lot_sizes"]
