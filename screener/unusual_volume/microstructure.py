"""Independent India microstructure enrichment stages."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

from screener.pledge import EVENT_FIELDS as PLEDGE_EVENT_FIELDS
from screener.unusual_volume.detector import Event
from screener.unusual_volume.enrichment import Enrichment, EnrichmentDiagnostic
from screener.unusual_volume.fii_dii import EVENT_FIELDS as FII_DII_EVENT_FIELDS
from screener.unusual_volume.option_chain import (
    EVENT_FIELDS as OPTION_CHAIN_EVENT_FIELDS,
)

_OVERLAY_ERRORS = (
    requests.RequestException,
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
)

_EVENT_FIELDS: dict[Enrichment, tuple[str, ...]] = {
    Enrichment.OPTION_CHAIN: OPTION_CHAIN_EVENT_FIELDS,
    Enrichment.FII_DII: FII_DII_EVENT_FIELDS,
    Enrichment.PLEDGE: PLEDGE_EVENT_FIELDS,
}


@dataclass(frozen=True)
class _StageOutcome:
    diagnostic: EnrichmentDiagnostic
    events: list[Event]


def _option_chain_stage(
    events: list[Event], snapshot_date: date, refresh: bool
) -> _StageOutcome:
    from screener.cache import append_panel_snapshot
    from screener.unusual_volume.option_chain import overlay_option_chain

    metrics = overlay_option_chain(events, refresh=refresh)
    if metrics:
        rows = pd.DataFrame(
            [
                {
                    "as_of": snapshot_date,
                    "SYMBOL": symbol,
                    "ce_oi": values.get("ce_oi"),
                    "pe_oi": values.get("pe_oi"),
                    "call_put_oi_ratio": values.get("call_put_oi_ratio"),
                    "pcr": values.get("pcr"),
                }
                for symbol, values in metrics.items()
            ]
        )
        append_panel_snapshot("option_chain", rows, dedupe_keys=["as_of", "SYMBOL"])
    return _StageOutcome(
        diagnostic=EnrichmentDiagnostic(
            enrichment=Enrichment.OPTION_CHAIN,
            status="applied",
            message=f"Option-chain overlay: {len(metrics)} symbol(s).",
        ),
        events=events,
    )


def _fii_dii_stage(
    events: list[Event], snapshot_date: date, refresh: bool
) -> _StageOutcome:
    from screener.unusual_volume.fii_dii import overlay_fii_dii

    metrics = overlay_fii_dii(events, snapshot_date, refresh=refresh)
    message = "FII/DII overlay: no market-wide metrics available."
    if metrics is not None:
        message = (
            f"FII/DII (market-wide): 5d FII={metrics['fii_5d_net']} "
            f"5d DII={metrics['dii_5d_net']} trend={metrics['fii_trend']}."
        )
    return _StageOutcome(
        diagnostic=EnrichmentDiagnostic(
            enrichment=Enrichment.FII_DII,
            status="applied",
            message=message,
        ),
        events=events,
    )


def _pledge_stage(
    events: list[Event], snapshot_date: date, refresh: bool
) -> _StageOutcome:
    del snapshot_date
    from screener.pledge import overlay_pledge

    overlay_pledge(events, refresh=refresh)
    return _StageOutcome(
        diagnostic=EnrichmentDiagnostic(
            enrichment=Enrichment.PLEDGE,
            status="applied",
            message="Pledge overlay applied.",
        ),
        events=events,
    )


_STAGES: dict[Enrichment, Callable[[list[Event], date, bool], _StageOutcome]] = {
    Enrichment.OPTION_CHAIN: _option_chain_stage,
    Enrichment.FII_DII: _fii_dii_stage,
    Enrichment.PLEDGE: _pledge_stage,
}


def _run_stage(
    enrichment: Enrichment,
    events: list[Event],
    snapshot_date: date,
    refresh: bool,
) -> _StageOutcome:
    try:
        return _STAGES[enrichment](events, snapshot_date, refresh)
    except _OVERLAY_ERRORS as exc:
        labels = {
            Enrichment.OPTION_CHAIN: "Option-chain overlay",
            Enrichment.FII_DII: "FII/DII overlay",
            Enrichment.PLEDGE: "Pledge overlay",
        }
        return _StageOutcome(
            diagnostic=EnrichmentDiagnostic(
                enrichment=enrichment,
                status="failed",
                message=f"{labels[enrichment]} failed: {exc}. Skipping.",
            ),
            events=events,
        )


def _merge_event_fields(
    targets: list[Event], source_events: list[Event], enrichment: Enrichment
) -> None:
    sources = {event.symbol: event for event in source_events}
    for target in targets:
        source = sources.get(target.symbol)
        if source is None:
            continue
        for field in _EVENT_FIELDS[enrichment]:
            setattr(target, field, getattr(source, field))


def run_microstructure_enrichments(
    events: list[Event],
    selected: frozenset[Enrichment],
    *,
    scan_date: date,
    snapshot_date: date,
    refresh: bool,
) -> list[EnrichmentDiagnostic]:
    """Run selected independent stages concurrently, then merge live fields."""
    ordered = [
        enrichment
        for enrichment in (
            Enrichment.OPTION_CHAIN,
            Enrichment.FII_DII,
            Enrichment.PLEDGE,
        )
        if enrichment in selected
    ]
    if not events or not ordered:
        return []

    attach_to_events = scan_date == snapshot_date
    runnable: list[Enrichment] = []
    outcomes: list[_StageOutcome] = []
    for enrichment in ordered:
        if enrichment is Enrichment.PLEDGE and not attach_to_events:
            outcomes.append(
                _StageOutcome(
                    diagnostic=EnrichmentDiagnostic(
                        enrichment=enrichment,
                        status="skipped",
                        message="Pledge overlay skipped for historical scan.",
                    ),
                    events=[],
                )
            )
        else:
            runnable.append(enrichment)

    def execute(enrichment: Enrichment) -> _StageOutcome:
        copies = [event.model_copy(deep=True) for event in events]
        return _run_stage(enrichment, copies, snapshot_date, refresh)

    if len(runnable) == 1:
        completed = [execute(runnable[0])]
    elif runnable:
        with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
            completed = list(executor.map(execute, runnable))
    else:
        completed = []
    by_enrichment = {outcome.diagnostic.enrichment: outcome for outcome in completed}
    outcomes.extend(
        by_enrichment[enrichment]
        for enrichment in ordered
        if enrichment in by_enrichment
    )
    if attach_to_events:
        for outcome in outcomes:
            if outcome.diagnostic.status == "applied":
                _merge_event_fields(
                    events, outcome.events, outcome.diagnostic.enrichment
                )
    return [outcome.diagnostic for outcome in outcomes]


__all__ = ["run_microstructure_enrichments"]
