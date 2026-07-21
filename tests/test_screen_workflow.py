from __future__ import annotations

from pathlib import Path

import pandas as pd

import screener.screen_workflow as sw
from screener.criteria import FilterCriteriaSelection
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
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
        scan=lambda **kwargs: calls.append("scan") or (2, frame),
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
        scan=lambda **kwargs: (1, frame),
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
        scan=lambda **kwargs: (1, frame),
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
        scan=lambda **kwargs: (1, frame),
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
