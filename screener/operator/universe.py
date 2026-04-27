"""Universe construction for the Operator Intent screener.

Two universes are stitched together (per user choice ``fo+cash``):

  F&O list (~210 names) — SYMBOLs that appear in today's F&O UDiff bhavcopy
    as stock futures (FinInstrmTp == 'STF'). These are the names where
    Operator_Action signals are computable.

  Top-500 cash universe — TradingView screener filtered to NSE-listed stocks
    above a price floor, ranked by daily volume. Reused from
    ``run_pinescript_strategies.load_universe`` so we share the same survivor
    bias (or lack thereof) as the existing daily picks pipeline.

Non-F&O cash names get blank OI columns and ``Operator_Action == None``
in the final output, but their delivery + price metrics still appear.
"""
from __future__ import annotations

import logging
from datetime import date

from .fetch import fetch_fo_bhavcopy

LOG = logging.getLogger(__name__)


def fno_symbols(d: date) -> list[str]:
    """SYMBOLs that have a stock-future contract in the F&O bhavcopy for ``d``."""
    df = fetch_fo_bhavcopy(d)
    return sorted(df["SYMBOL"].unique().tolist())


def cash_top_500() -> list[str]:
    """Top-500 NSE cash names by volume (price-floored).

    Reuses the existing universe loader so the operator screener stays in
    sync with the autoresearch / scan_today.py pick pipeline.
    """
    from run_pinescript_strategies import load_universe
    return load_universe("india", None)


def combined_universe(d: date, *, mode: str = "fo+cash") -> tuple[list[str], set[str]]:
    """Return ``(all_symbols, fno_set)`` for the requested mode.

    ``mode``:
      ``fo``       — F&O list only (~210 names; every row gets OI signal logic).
      ``fo+cash``  — Union of F&O + top-500 cash. Default.

    The returned tuple's second element is the set of F&O-eligible symbols,
    used downstream to know which rows are eligible for Operator_Action.
    """
    fno = fno_symbols(d)
    fno_set = set(fno)
    if mode == "fo":
        return fno, fno_set
    if mode == "fo+cash":
        try:
            cash = cash_top_500()
        except Exception as exc:
            LOG.warning("cash universe load failed (%s); falling back to F&O only", exc)
            return fno, fno_set
        union = sorted(set(fno) | set(cash))
        return union, fno_set
    raise ValueError(f"unknown universe mode: {mode}")
