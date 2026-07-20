"""NSE option-chain overlay — per-stock PCR and call/put OI ratio.

NSE serves the equity option chain live only (no historical archive), so this
overlay attaches the current snapshot to surviving scan events and the service
layer persists a daily row to ``~/.screener/panels/option_chain.parquet`` so a
backtestable history accumulates over time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from screener.options.metrics import compute_chain_metrics

# The raw NSE option-chain transport now lives in the options package; import it
# here so unusual_volume depends on options (not the reverse). Re-exported for
# existing callers (earnings sentiment, the earnings data facade) and tests that
# reference the transport/URL seam at this module.
from screener.options.nse_live import _OC_PAGE as _OC_PAGE
from screener.options.nse_live import NSELiveOptionsProvider
from screener.options.nse_live import fetch_option_chain as fetch_option_chain

from .detector import Event

EVENT_FIELDS = ("call_put_oi_ratio", "pcr")


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
