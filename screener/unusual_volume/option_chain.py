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
from typing import Optional

from screener.options.metrics import compute_chain_metrics
from screener.options.nse_live import NSELiveOptionsProvider

from .detector import Event
from .nse_client import nse_cached_json

_OC_EQUITY_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={sym}"
_OC_INDEX_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={sym}"
_OC_INDEX_SYMBOLS = frozenset(
    {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
)
_OC_PAGE = "https://www.nseindia.com/option-chain"
EVENT_FIELDS = ("call_put_oi_ratio", "pcr")


def fetch_option_chain(symbol: str, *, refresh: bool = False) -> Optional[dict]:
    normalized = symbol.upper()
    template = _OC_INDEX_URL if normalized in _OC_INDEX_SYMBOLS else _OC_EQUITY_URL
    url = template.format(sym=urllib.parse.quote(normalized))
    raw = nse_cached_json(
        "nse_option_chain",
        ("oc", normalized, str(date.today())),
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


def overlay_option_chain(
    events: list[Event], *, refresh: bool = False, max_workers: int = 6
) -> dict[str, dict]:
    """Mutate events with call_put_oi_ratio / pcr; return {symbol: metrics}."""
    if not events:
        return {}
    symbols = sorted({ev.symbol.upper() for ev in events})

    provider = NSELiveOptionsProvider(raw_fetcher=fetch_option_chain)

    def _one(sym: str) -> tuple[str, Optional[dict]]:
        chain = provider.fetch_chain(sym, "india", refresh=refresh)
        if chain is None:
            return sym, None
        derived = compute_chain_metrics(chain)
        ce_oi = derived.call_oi or None
        pe_oi = derived.put_oi or None
        return sym, {
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "call_put_oi_ratio": _safe_ratio(ce_oi, pe_oi),
            "pcr": _safe_ratio(pe_oi, ce_oi),
        }

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
