from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import screener.screen_workflow as sw
from screener.criteria import FilterCriteriaSelection
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    run_screen_workflow,
)

_AS_OF = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _request(
    *,
    output_csv: bool = False,
    report_path: Path | None = None,
    earnings: bool = False,
    defer_report: bool = False,
) -> ScreenRequest:
    return ScreenRequest(
        market="us",
        criteria_names=("ema",),
        limit=5,
        order_by="setup_score",
        output_csv=output_csv,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=report_path,
        earnings=earnings,
        defer_report=defer_report,
    )


def _df(*names: str) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names), "description": list(names)})


def _patch(monkeypatch, **overrides) -> None:
    for name, fn in overrides.items():
        monkeypatch.setattr(sw, name, fn)


def test_screen_workflow_csv_short_circuits_history_and_report(monkeypatch, tmp_path):
    calls: list[str] = []
    frame = _df("AAA", "BBB")

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=lambda **kwargs: calls.append("scan") or (2, frame, _AS_OF),
        save_run=lambda *args: calls.append("save") or 1,
        previous_run=lambda *args: calls.append("previous") or None,
        render_screen_report=lambda *args, **kwargs: (
            calls.append("report") or tmp_path / "unused.html"
        ),
        enrich_days_to_earnings=lambda df, market: df.assign(days_to_earnings=None),
    )

    outcome = run_screen_workflow(_request(output_csv=True, earnings=True))

    assert outcome.mode is ScreenMode.CSV
    assert outcome.df is not None
    assert outcome.df["name"].tolist() == ["AAA", "BBB"]
    assert "days_to_earnings" in outcome.df.columns
    # as_of is the scan fetch time the workflow received, not run time.
    assert outcome.as_of == _AS_OF
    assert calls == ["scan"]


def test_screen_workflow_skips_earnings_enrichment_by_default(monkeypatch, tmp_path):
    frame = _df("AAA")

    def unexpected_enrichment(df, market):
        raise AssertionError("earnings enrichment must be opt-in")

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        enrich_days_to_earnings=unexpected_enrichment,
    )

    outcome = run_screen_workflow(_request(output_csv=True))

    assert outcome.df is not None
    assert "days_to_earnings" not in outcome.df.columns


def test_screen_workflow_first_run_uses_default_report_path(monkeypatch, tmp_path):
    frame = _df("AAA")
    report = tmp_path / "screen.html"
    rendered: dict[str, object] = {}

    def render_report(*args, **kwargs):
        rendered["path"] = args[4]
        rendered["first_run"] = kwargs["first_run"]
        Path(args[4]).write_text("report", encoding="utf-8")
        return Path(args[4])

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: report,
        render_screen_report=render_report,
    )

    outcome = run_screen_workflow(_request())

    assert outcome.mode is ScreenMode.RESULTS
    assert outcome.first_run is True
    assert outcome.added == ()
    assert outcome.removed == ()
    assert outcome.report_path == report
    assert report.read_text(encoding="utf-8") == "report"
    assert rendered == {"path": report, "first_run": True}


def test_screen_workflow_forwards_strict_timeout_retries_to_scan(monkeypatch):
    seen: dict = {}

    def fake_scan(**kwargs):
        seen.update(kwargs)
        return 0, _df("AAA"), _AS_OF

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=fake_scan,
    )
    request = ScreenRequest(
        market="us",
        criteria_names=("ema",),
        limit=5,
        order_by="setup_score",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
        strict=True,
        timeout=5.0,
        retries=2,
    )

    run_screen_workflow(request)

    assert seen["strict"] is True
    assert seen["timeout"] == 5.0
    assert seen["retries"] == 2


def test_screen_workflow_previous_run_diff_uses_explicit_report_path(
    monkeypatch, tmp_path
):
    frame = _df("AAA")
    prev = pd.DataFrame({"ticker": ["BBB"]})
    explicit = tmp_path / "explicit.html"

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema+value", ["EMA", "VALUE"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 8,
        previous_run=lambda *args: prev,
        diff=lambda current, previous: (["AAA"], ["BBB"]),
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )

    outcome = run_screen_workflow(_request(report_path=explicit))

    assert outcome.label == "ema+value"
    assert outcome.first_run is False
    assert outcome.added == ("AAA",)
    assert outcome.removed == ("BBB",)
    assert outcome.report_path == explicit
    assert explicit.exists()


def test_bar_screen_warns_when_the_prefilter_scan_hits_its_cap(monkeypatch, caplog):
    """A truncated scan drops names the bar rule never saw, so it must say so.

    The prefilter is an optimisation, never a rule: it may only remove names
    the bar rule would have removed anyway (D21). A scan capped at
    ``_PREFILTER_CANDIDATE_CAP`` and ordered by volume breaks that on any field
    wider than the cap, and nothing downstream can tell the difference between
    a short field and a truncated one.
    """
    from screener.screen_candidates import ScreenStrategy
    from screener.strategies.spec import discover_plugins, resolve_strategy_spec

    discover_plugins()
    spec = resolve_strategy_spec("breakout")
    assert spec is not None
    strategy = ScreenStrategy(criterion="breakout", spec=spec)
    returned = _df(*(f"T{i}" for i in range(3)))
    returned["ticker"] = returned["name"]

    _patch(
        monkeypatch,
        resolve_screen_strategy=lambda names: strategy,
        # The vendor matched 9000 names; the capped scan returned 3.
        scan=lambda **kwargs: (9000, returned, _AS_OF),
        screen_candidates=lambda *args, **kwargs: _df("T0"),
        save_run=lambda *args: 1,
        previous_run=lambda *args: None,
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )

    with caplog.at_level(logging.WARNING, logger=sw.LOG.name):
        run_screen_workflow(_request())

    assert any(
        "prefilter scan returned 3 of 9000" in record.message
        for record in caplog.records
    ), caplog.text


def test_bar_screen_stays_quiet_when_the_scan_returns_the_whole_field(
    monkeypatch, caplog
):
    """The warning must fire on truncation only, or it is noise on every run."""
    from screener.screen_candidates import ScreenStrategy
    from screener.strategies.spec import discover_plugins, resolve_strategy_spec

    discover_plugins()
    spec = resolve_strategy_spec("breakout")
    assert spec is not None
    strategy = ScreenStrategy(criterion="breakout", spec=spec)
    returned = _df(*(f"T{i}" for i in range(3)))
    returned["ticker"] = returned["name"]

    _patch(
        monkeypatch,
        resolve_screen_strategy=lambda names: strategy,
        scan=lambda **kwargs: (3, returned, _AS_OF),
        screen_candidates=lambda *args, **kwargs: _df("T0"),
        save_run=lambda *args: 1,
        previous_run=lambda *args: None,
        render_screen_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
    )

    with caplog.at_level(logging.WARNING, logger=sw.LOG.name):
        run_screen_workflow(_request())

    assert not any("prefilter scan returned" in r.message for r in caplog.records)


def test_deferred_report_is_written_only_when_the_caller_asks(monkeypatch, tmp_path):
    """``defer_report`` hands the render back; the workflow must not run it."""
    frame = _df("AAA")
    report = tmp_path / "screen.html"
    rendered: list[Path] = []

    def render_report(*args, **kwargs):
        rendered.append(Path(args[4]))
        Path(args[4]).write_text("report", encoding="utf-8")
        return Path(args[4])

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: report,
        render_screen_report=render_report,
    )

    outcome = run_screen_workflow(_request(defer_report=True))

    assert outcome.report_path == report
    assert rendered == []
    assert not report.exists()

    assert outcome.render_report is not None
    assert outcome.render_report() == report
    assert rendered == [report]
    assert report.read_text(encoding="utf-8") == "report"


def test_a_report_written_for_the_caller_leaves_no_render_hook(monkeypatch, tmp_path):
    """Without ``defer_report`` the report is already written, so there is
    nothing left to call - a caller that ran the hook anyway would render it
    twice."""
    frame = _df("AAA")
    report = tmp_path / "screen.html"

    _patch(
        monkeypatch,
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        scan=lambda **kwargs: (1, frame, _AS_OF),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        temp_report_path=lambda prefix: report,
        render_screen_report=lambda *args, **kwargs: Path(args[4]),
    )

    outcome = run_screen_workflow(_request())

    assert outcome.render_report is None
