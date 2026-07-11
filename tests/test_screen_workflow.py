from __future__ import annotations

from pathlib import Path

import pandas as pd

from screener.criteria import FilterCriteriaSelection, PipelineCriteriaSelection
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    ScreenWorkflowDeps,
    run_screen_workflow,
)


def _request(
    *,
    output_csv: bool = False,
    report_path: Path | None = None,
    earnings: bool = False,
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
    )


def _df(*names: str) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names), "description": list(names)})


def test_screen_workflow_csv_short_circuits_history_and_report(tmp_path):
    calls: list[str] = []
    frame = _df("AAA", "BBB")

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        parse_cache_ttl=lambda raw: 900.0,
        scan=lambda **kwargs: calls.append("scan") or (2, frame),
        save_run=lambda *args: calls.append("save") or 1,
        previous_run=lambda *args: calls.append("previous") or None,
        diff=lambda current, previous: ([], []),
        temp_report_path=lambda prefix: tmp_path / f"{prefix}.html",
        render_report=lambda *args, **kwargs: (
            calls.append("report") or tmp_path / "unused.html"
        ),
        enrich_days_to_earnings=lambda df, market: df.assign(days_to_earnings=None),
    )

    outcome = run_screen_workflow(_request(output_csv=True, earnings=True), deps)

    assert outcome.mode is ScreenMode.CSV
    assert outcome.df is not None
    assert outcome.df["name"].tolist() == ["AAA", "BBB"]
    assert "days_to_earnings" in outcome.df.columns
    assert calls == ["scan"]


def test_screen_workflow_skips_earnings_enrichment_by_default(tmp_path):
    frame = _df("AAA")

    def unexpected_enrichment(df, market):
        raise AssertionError("earnings enrichment must be opt-in")

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        parse_cache_ttl=lambda raw: 900.0,
        scan=lambda **kwargs: (1, frame),
        save_run=lambda *args: 1,
        previous_run=lambda *args: None,
        diff=lambda current, previous: ([], []),
        temp_report_path=lambda prefix: tmp_path / f"{prefix}.html",
        render_report=lambda *args, **kwargs: tmp_path / "unused.html",
        enrich_days_to_earnings=unexpected_enrichment,
    )

    outcome = run_screen_workflow(_request(output_csv=True), deps)

    assert outcome.df is not None
    assert "days_to_earnings" not in outcome.df.columns


def test_screen_workflow_first_run_uses_default_report_path(tmp_path):
    frame = _df("AAA")
    report = tmp_path / "screen.html"
    rendered: dict[str, object] = {}

    def render_report(*args, **kwargs):
        rendered["path"] = args[4]
        rendered["first_run"] = kwargs["first_run"]
        Path(args[4]).write_text("report", encoding="utf-8")
        return Path(args[4])

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema", ["FILTER"]
        ),
        parse_cache_ttl=lambda raw: 900.0,
        scan=lambda **kwargs: (1, frame),
        save_run=lambda *args: 7,
        previous_run=lambda *args: None,
        diff=lambda current, previous: (["unused"], ["unused"]),
        temp_report_path=lambda prefix: report,
        render_report=render_report,
        enrich_days_to_earnings=lambda df, market: df,
    )

    outcome = run_screen_workflow(_request(), deps)

    assert outcome.mode is ScreenMode.RESULTS
    assert outcome.first_run is True
    assert outcome.added == ()
    assert outcome.removed == ()
    assert outcome.report_path == report
    assert report.read_text(encoding="utf-8") == "report"
    assert rendered == {"path": report, "first_run": True}


def test_screen_workflow_previous_run_diff_uses_explicit_report_path(
    tmp_path,
):
    frame = _df("AAA")
    prev = pd.DataFrame({"ticker": ["BBB"]})
    explicit = tmp_path / "explicit.html"

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: FilterCriteriaSelection(
            tuple(names), "ema+value", ["EMA", "VALUE"]
        ),
        parse_cache_ttl=lambda raw: 123.0,
        scan=lambda **kwargs: (1, frame),
        save_run=lambda *args: 8,
        previous_run=lambda *args: prev,
        diff=lambda current, previous: (["AAA"], ["BBB"]),
        temp_report_path=lambda prefix: tmp_path / "unused.html",
        render_report=lambda *args, **kwargs: (
            Path(args[4]).write_text("report", encoding="utf-8") or Path(args[4])
        ),
        enrich_days_to_earnings=lambda df, market: df,
    )

    outcome = run_screen_workflow(_request(report_path=explicit), deps)

    assert outcome.label == "ema+value"
    assert outcome.first_run is False
    assert outcome.added == ("AAA",)
    assert outcome.removed == ("BBB",)
    assert outcome.report_path == explicit
    assert explicit.exists()


def test_screen_workflow_pipeline_bypasses_filter_scan_history_and_report(tmp_path):
    calls: list[tuple[str, object]] = []

    def runner(**kwargs):
        calls.append(("runner", kwargs))

    deps = ScreenWorkflowDeps(
        resolve_criteria=lambda names: PipelineCriteriaSelection("garp", runner),
        parse_cache_ttl=lambda raw: 900.0,
        scan=lambda **kwargs: calls.append(("scan", kwargs)) or (0, pd.DataFrame()),
        save_run=lambda *args: calls.append(("save", args)) or 1,
        previous_run=lambda *args: None,
        diff=lambda current, previous: ([], []),
        temp_report_path=lambda prefix: tmp_path / f"{prefix}.html",
        render_report=lambda *args, **kwargs: (
            calls.append(("report", args)) or tmp_path / "unused.html"
        ),
        enrich_days_to_earnings=lambda df, market: (
            calls.append(("enrich", market)) or df
        ),
    )

    outcome = run_screen_workflow(
        ScreenRequest(
            market="india",
            criteria_names=("garp",),
            limit=3,
            order_by="setup_score",
            output_csv=True,
            detail=True,
            refresh=True,
            cache_ttl="1d",
            report_path=tmp_path / "ignored.html",
            open_report=True,
        ),
        deps,
    )

    assert outcome.mode is ScreenMode.PIPELINE
    assert outcome.label == "garp"
    assert calls == [
        (
            "runner",
            {
                "market": "india",
                "limit": 3,
                "output_csv": True,
                "refresh": True,
                "cache_ttl": "1d",
            },
        )
    ]
