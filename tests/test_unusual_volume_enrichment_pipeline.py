from datetime import date
import threading

from screener import pledge
from screener.unusual_volume import fii_dii, option_chain
from screener.unusual_volume.detector import Event
from screener.unusual_volume.enrichment import Enrichment
from screener.unusual_volume.microstructure import run_microstructure_enrichments
from screener.unusual_volume.service import UnusualVolumeRequest


def _event() -> Event:
    return Event(
        symbol="RELIANCE",
        date=date(2026, 5, 15),
        close=2_500.0,
        pct_change=1.0,
        volume=150_000.0,
        avg_volume_20d=50_000.0,
        rvol=3.0,
        rvol_5d=3.0,
        rvol_50d=3.0,
        rvol_90d=3.0,
        z_score=2.5,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="HIGH",
    )


def test_request_collects_legacy_flags_into_typed_enrichments() -> None:
    request = UnusualVolumeRequest(
        market="india",
        as_of=date(2026, 5, 15),
        universe=["RELIANCE"],
        buildup_enabled=True,
        option_chain=True,
    )

    assert request.enrichments == {
        Enrichment.BUILDUP,
        Enrichment.OPTION_CHAIN,
    }
    assert request.buildup_enabled
    assert request.option_chain
    assert not request.pledge

    updated = request.model_copy(update={"pledge": True})
    assert updated.pledge
    assert updated.option_chain


def test_independent_microstructure_stages_run_concurrently(monkeypatch) -> None:
    barrier = threading.Barrier(2, timeout=2)

    def option_stage(events, refresh=False):
        barrier.wait()
        return {}

    def fii_stage(events, snapshot_date, refresh=False):
        barrier.wait()
        return None

    monkeypatch.setattr(option_chain, "overlay_option_chain", option_stage)
    monkeypatch.setattr(fii_dii, "overlay_fii_dii", fii_stage)

    diagnostics = run_microstructure_enrichments(
        [_event()],
        frozenset({Enrichment.OPTION_CHAIN, Enrichment.FII_DII}),
        scan_date=date(2026, 5, 15),
        snapshot_date=date(2026, 5, 15),
        refresh=False,
    )

    assert [diagnostic.status for diagnostic in diagnostics] == [
        "applied",
        "applied",
    ]


def test_stage_results_merge_only_owned_fields(monkeypatch) -> None:
    def option_stage(events, refresh=False):
        events[0].pcr = 2.0
        return {}

    def fii_stage(events, snapshot_date, refresh=False):
        events[0].fii_5d_net = 100.0
        return {"fii_5d_net": 100.0, "dii_5d_net": None, "fii_trend": None}

    monkeypatch.setattr(option_chain, "overlay_option_chain", option_stage)
    monkeypatch.setattr(fii_dii, "overlay_fii_dii", fii_stage)
    event = _event()

    run_microstructure_enrichments(
        [event],
        frozenset({Enrichment.OPTION_CHAIN, Enrichment.FII_DII}),
        scan_date=event.date,
        snapshot_date=event.date,
        refresh=False,
    )

    assert event.pcr == 2.0
    assert event.fii_5d_net == 100.0


def test_stage_failure_is_structured_and_isolated(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("NSE unavailable")

    monkeypatch.setattr(option_chain, "overlay_option_chain", fail)
    diagnostics = run_microstructure_enrichments(
        [_event()],
        frozenset({Enrichment.OPTION_CHAIN}),
        scan_date=date(2026, 5, 15),
        snapshot_date=date(2026, 5, 15),
        refresh=False,
    )

    assert diagnostics[0].status == "failed"
    assert diagnostics[0].enrichment is Enrichment.OPTION_CHAIN
    assert "NSE unavailable" in diagnostics[0].message


def _changed_fields(before: Event, after: Event) -> set[str]:
    return {
        name
        for name, value in before.model_dump().items()
        if after.model_dump()[name] != value
    }


def test_overlays_only_mutate_their_declared_event_fields(monkeypatch) -> None:
    event = _event()
    before = event.model_copy(deep=True)
    monkeypatch.setattr(
        option_chain,
        "fetch_option_chain",
        lambda *args, **kwargs: {
            "filtered": {"CE": {"totOI": 10}, "PE": {"totOI": 20}}
        },
    )
    option_chain.overlay_option_chain([event], max_workers=1)
    assert _changed_fields(before, event) <= set(option_chain.EVENT_FIELDS)

    event = _event()
    before = event.model_copy(deep=True)
    monkeypatch.setattr(fii_dii, "fetch_fii_dii_today", lambda **kwargs: None)
    monkeypatch.setattr(
        fii_dii,
        "read_frame",
        lambda path: __import__("pandas").DataFrame(
            [{"date": event.date, "fii_net": 10.0, "dii_net": 5.0}]
        ),
    )
    fii_dii.overlay_fii_dii([event], event.date)
    assert _changed_fields(before, event) <= set(fii_dii.EVENT_FIELDS)

    event = _event()
    before = event.model_copy(deep=True)
    monkeypatch.setattr(pledge, "resolve_pledge_pct", lambda *args, **kwargs: 3.5)
    pledge.overlay_pledge([event], max_workers=1)
    assert _changed_fields(before, event) <= set(pledge.EVENT_FIELDS)
