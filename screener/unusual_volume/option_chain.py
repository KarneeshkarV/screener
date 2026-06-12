"""NSE option-chain overlay — per-stock PCR and call/put OI ratio.

NSE serves the equity option chain live only (no historical archive), so this
overlay attaches the current snapshot to surviving scan events and the service
layer persists a daily row to ``~/.screener/panels/option_chain.parquet`` so a
backtestable history accumulates over time.
"""

from __future__ import annotations

import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Optional, cast

from .detector import Event
from .nse_client import nse_cached_json

_OC_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={sym}"
_OC_PAGE = "https://www.nseindia.com/option-chain"


def fetch_option_chain(symbol: str, *, refresh: bool = False) -> Optional[dict]:
    url = _OC_URL.format(sym=urllib.parse.quote(symbol.upper()))
    raw = nse_cached_json(
        "nse_option_chain",
        ("oc", symbol.upper(), str(date.today())),
        url,
        f"option chain {symbol}",
        refresh=refresh,
        extra_prime_page=_OC_PAGE,
    )
    return raw if isinstance(raw, dict) else None


def _safe_ratio(num: float | None, denom: float | None) -> Optional[float]:
    if num is None or denom is None or denom == 0:
        return None
    return round(float(num) / float(denom), 4)


def compute_oc_metrics(raw: dict) -> dict:
    """Extract CE/PE total OI and derive call_put_oi_ratio + pcr.

    Prefers NSE's precomputed near-expiry ``filtered.CE/PE.totOI``; falls back
    to summing per-strike ``records.data[*].CE/PE.openInterest``.
    """
    ce_oi: Optional[float] = None
    pe_oi: Optional[float] = None
    filtered = raw.get("filtered") if isinstance(raw, dict) else None
    if isinstance(filtered, dict):
        ce = filtered.get("CE") or {}
        pe = filtered.get("PE") or {}
        ce_oi = _as_float(ce.get("totOI"))
        pe_oi = _as_float(pe.get("totOI"))
    if ce_oi is None or pe_oi is None:
        records = (raw.get("records") or {}).get("data") or []
        ce_sum = pe_sum = 0.0
        for row in records:
            ce_sum += _as_float((row.get("CE") or {}).get("openInterest")) or 0.0
            pe_sum += _as_float((row.get("PE") or {}).get("openInterest")) or 0.0
        ce_oi = ce_sum or None
        pe_oi = pe_sum or None
    # A zero OI leg is meaningless data — both ratios collapse to None.
    if not ce_oi:
        ce_oi = None
    if not pe_oi:
        pe_oi = None
    return {
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
        "call_put_oi_ratio": _safe_ratio(ce_oi, pe_oi),
        "pcr": _safe_ratio(pe_oi, ce_oi),
    }


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        # value is object; the try/except guards non-numeric inputs.
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def overlay_option_chain(
    events: list[Event], *, refresh: bool = False, max_workers: int = 6
) -> dict[str, dict]:
    """Mutate events with call_put_oi_ratio / pcr; return {symbol: metrics}."""
    if not events:
        return {}
    symbols = sorted({ev.symbol.upper() for ev in events})

    def _one(sym: str) -> tuple[str, Optional[dict]]:
        raw = fetch_option_chain(sym, refresh=refresh)
        return sym, (compute_oc_metrics(raw) if raw else None)

    metrics: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed([pool.submit(_one, s) for s in symbols]):
            sym, m = fut.result()
            if m is not None:
                metrics[sym] = m
    for ev in events:
        m = metrics.get(ev.symbol.upper())
        if m is not None:
            ev.call_put_oi_ratio = m["call_put_oi_ratio"]
            ev.pcr = m["pcr"]
    return metrics
