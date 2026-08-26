"""M-4 regression: grid optimisation output must carry a prominent in-sample /
selection-bias disclaimer steering users toward walk-forward.

Unlike walk-forward, the grid table reports best-of-grid IN-SAMPLE metrics, so
the headline Sharpe is optimistically biased. The disclaimer must make that
explicit so the figure is not mistaken for an out-of-sample estimate.
"""

from __future__ import annotations

import json

from rich.console import Console

from screener.backtester.optimization.grid import GridSearchResult
from screener.backtester.optimization.reporting import (
    GRID_IN_SAMPLE_DISCLAIMER,
    print_grid_table,
    write_html_report,
    write_json_report,
)


def _capture_grid_report() -> str:
    results = [
        GridSearchResult(
            params={"foo": 1},
            score=1.2345,
            metrics={
                "starting_equity": 100_000.0,
                "final_equity": 110_000.0,
                "sharpe": 1.5,
                "profit_factor": 2.0,
                "max_drawdown": -0.1,
            },
            trade_count=42,
        ),
        GridSearchResult(
            params={"foo": 2},
            score=0.9,
            metrics={"sharpe": 0.8, "profit_factor": 1.2, "max_drawdown": -0.2},
            trade_count=30,
        ),
    ]
    # Wide console so styled text isn't wrapped mid-phrase.
    console = Console(record=True, width=200)
    print_grid_table(results, console=console)
    return console.export_text()


def test_grid_report_contains_in_sample_warning():
    text = _capture_grid_report().lower()
    # Rendered on a width=200 console, so header words stay on one line.
    assert "starting capital" in text
    assert "final equity" in text
    assert "in-sample" in text
    assert "selection bias" in text


def test_grid_report_steers_to_walk_forward():
    text = _capture_grid_report().lower()
    assert "walk-forward" in text


def test_grid_report_flags_metrics_as_biased():
    text = _capture_grid_report().lower()
    # Must warn the headline figure is biased / not out-of-sample.
    assert "biased" in text
    assert "out-of-sample" in text


def test_grid_json_export_carries_disclaimer(tmp_path):
    # The exported JSON artifact (used non-interactively) must carry the warning
    # so it is not mistaken for an out-of-sample estimate.
    payload = {"warning": GRID_IN_SAMPLE_DISCLAIMER, "results": [{"sharpe": 1.5}]}
    path = tmp_path / "grid.json"
    write_json_report(payload, path)
    loaded = json.loads(path.read_text())
    assert "in-sample" in loaded["warning"].lower()
    assert "selection bias" in loaded["warning"].lower()
    assert loaded["results"] == [{"sharpe": 1.5}]


def test_grid_html_export_carries_disclaimer(tmp_path):
    path = tmp_path / "grid.html"
    write_html_report(
        [{"sharpe": 1.5}],
        path,
        "Grid Search Report",
        disclaimer=GRID_IN_SAMPLE_DISCLAIMER,
    )
    html = path.read_text().lower()
    assert "in-sample" in html
    assert "selection bias" in html
    assert "walk-forward" in html


def test_walk_forward_html_has_no_in_sample_disclaimer(tmp_path):
    # The generic writer must NOT inject an in-sample warning when none is passed
    # (walk-forward is out-of-sample and should not carry the grid disclaimer).
    path = tmp_path / "wf.html"
    write_html_report([{"sharpe": 1.5}], path, "Walk-Forward Report")
    assert "selection bias" not in path.read_text().lower()
