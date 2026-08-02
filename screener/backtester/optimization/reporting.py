"""Console, JSON, and HTML reports for optimization results."""

from __future__ import annotations

import html as html_lib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from screener import agentio
from screener.backtester.metrics import (
    format_result_value,
    result_view,
    result_view_columns,
)
from screener.backtester.optimization.grid import GridSearchResult
from screener.backtester.optimization.walk_forward import WalkForwardSummary
from screener.html_report import html_page

GRID_IN_SAMPLE_DISCLAIMER = (
    "IN-SAMPLE / SELECTION BIAS WARNING: These metrics are computed on the SAME "
    "data the grid selected the best parameters from. The headline best-of-grid "
    "Sharpe is optimistically biased and is NOT an out-of-sample estimate of "
    "future performance. Use `optimize walk-forward` for an honest, out-of-sample "
    "assessment before trusting these parameters."
)

_REPORT_CSS = """
    body { font-family: system-ui, sans-serif; margin: 32px; color: #1a1a1a; }
    h1, h2 { margin-top: 1.6em; }
    h1 { margin-top: 0; }
    pre { background: #f5f5f5; border: 1px solid #ddd; padding: 16px; overflow: auto; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }
    th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; }
    th { background: #f0f0f0; }
    tr:nth-child(even) { background: #fafafa; }
    .banner { background:#fff3cd; border:1px solid #ffeeba; color:#856404;
              padding:12px 16px; border-radius:4px; font-weight:600; }
    .verdict { background:#e8f5e9; border:1px solid #c8e6c9; padding:12px 16px;
               border-radius:4px; font-weight:600; }
    .verdict.caution { background:#fff3e0; border-color:#ffe0b2; }
    .verdict.fail { background:#ffebee; border-color:#ffcdd2; }
    .meta { color: #555; font-size: 13px; }
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump()
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return str(value)


def write_json_report(data: Any, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, default=_json_default))


def _print_table(table: Table, console: Console) -> None:
    """Use the shared bounded table primitive when an agent drives the CLI."""
    if agentio.is_agent_mode():
        agentio.render_table(table, console)
    else:
        console.print(table)


def _view_cells(metrics: Mapping[str, Any]) -> dict[str, str]:
    return {row.key: row.formatted for row in result_view(metrics)}


def print_grid_table(
    results: Iterable[GridSearchResult], console: Console | None = None
) -> None:
    console = console or Console()
    rows = list(results)
    columns = result_view_columns(result.metrics for result in rows)
    table = Table(title="Grid Search Results", show_header=True, header_style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    for column in columns:
        table.add_column(column.label, justify="right")
    table.add_column("Params")
    for rank, result in enumerate(rows, start=1):
        cells = _view_cells(result.metrics)
        table.add_row(
            str(rank),
            format_result_value(result.score, "ratio"),
            *(cells.get(column.key, "-") for column in columns),
            json.dumps(result.params, sort_keys=True),
        )
    _print_table(table, console)
    console.print(
        GRID_IN_SAMPLE_DISCLAIMER.replace(
            "IN-SAMPLE / SELECTION BIAS WARNING:",
            "[bold yellow]IN-SAMPLE / SELECTION BIAS WARNING:[/bold yellow]",
        )
    )


def print_walk_forward_table(
    summary: WalkForwardSummary, console: Console | None = None
) -> None:
    console = console or Console()
    columns = result_view_columns(result.test_metrics for result in summary.windows)
    table = Table(title="Walk-Forward Results", show_header=True, header_style="bold")
    table.add_column("Window")
    table.add_column("Train Score", justify="right")
    for column in columns:
        table.add_column(f"Test {column.label}", justify="right")
    table.add_column("Params")
    for result in summary.windows:
        window = result.window
        cells = _view_cells(result.test_metrics)
        table.add_row(
            f"{window.train_start}..{window.test_end}",
            format_result_value(result.best_train.score, "ratio"),
            *(cells.get(column.key, "-") for column in columns),
            json.dumps(result.best_train.params, sort_keys=True),
        )
    _print_table(table, console)
    console.print(
        "Stability: "
        f"{format_result_value(summary.stability_score, 'ratio')}  "
        "Train/Test score ratio: "
        f"{format_result_value(summary.train_test_score_ratio, 'ratio')}  "
        f"Overfit flag: {summary.overfit_flag}"
    )


def write_html_report(
    data: Any,
    path: Path | str,
    title: str = "Optimization Report",
    disclaimer: str | None = None,
) -> None:
    payload = json.dumps(data, indent=2, default=_json_default)
    banner = (
        f'  <p style="background:#fff3cd; border:1px solid #ffeeba; color:#856404; '
        f'padding:12px 16px; border-radius:4px; font-weight:600;">{disclaimer}</p>\n'
        if disclaimer
        else ""
    )
    css = (
        "    body { font-family: system-ui, sans-serif; margin: 32px; }\n"
        "    pre { background: #f5f5f5; border: 1px solid #ddd; padding: 16px; "
        "overflow: auto; }"
    )
    body = f"  <h1>{title}</h1>\n{banner}  <pre>{payload}</pre>"
    html = html_page(title, css, body, viewport=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html_lib.escape(str(h))}</th>" for h in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{html_lib.escape(str(c))}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_research_html_report(data: Mapping[str, Any], path: Path | str) -> None:
    """Combined single-page research report (config / grid / WF / MC).

    Composes the same dependency-free HTML style as :func:`write_html_report`
    (inline CSS, plain tables via :func:`html_page`) with structured sections.
    """
    title = "Research Report"
    config = data.get("config") or {}
    grid = data.get("grid") or {}
    walk_forward = data.get("walk_forward") or {}
    monte_carlo = data.get("monte_carlo") or {}
    summary = data.get("summary") or {}
    metric = str(config.get("metric") or "sharpe")

    verdict = str(summary.get("verdict") or "")
    verdict_class = "verdict"
    if verdict.startswith("FAIL"):
        verdict_class += " fail"
    elif verdict.startswith("CAUTION"):
        verdict_class += " caution"

    config_rows = [
        ("Market", config.get("market")),
        ("Strategy", config.get("strategy_name")),
        ("Entry", config.get("entry_expr")),
        ("Exit", config.get("exit_expr")),
        ("Start", config.get("start_date")),
        ("End", config.get("end_date")),
        (
            "Train / test / step days",
            (
                f"{config.get('train_days')} / {config.get('test_days')} / "
                f"{config.get('step_days')}"
            ),
        ),
        ("Metric", metric),
        (
            "Parameter grid",
            json.dumps(config.get("parameter_grid") or {}, sort_keys=True),
        ),
        ("Tickers", config.get("tickers")),
        ("MC iterations", config.get("mc_iterations")),
    ]
    config_table = _html_table(
        ["Field", "Value"],
        [
            (
                key,
                value
                if isinstance(value, str)
                else format_result_value(value, "ratio"),
            )
            for key, value in config_rows
        ],
    )

    grid_results = grid.get("results") or []
    grid_columns = result_view_columns(
        (result.get("metrics") or {}) for result in grid_results
    )
    grid_rows = []
    for rank, result in enumerate(grid_results, start=1):
        cells = _view_cells(result.get("metrics") or {})
        grid_rows.append(
            [
                rank,
                format_result_value(result.get("score"), "ratio"),
                *(cells.get(column.key, "-") for column in grid_columns),
                json.dumps(result.get("params") or {}, sort_keys=True),
            ]
        )
    grid_table = _html_table(
        ["Rank", "Score", *(column.label for column in grid_columns), "Params"],
        grid_rows,
    )

    stability = grid.get("stability") or []
    stability_rows = [
        [
            row.get("parameter"),
            row.get("shape"),
            row.get("best_value"),
            format_result_value(row.get("best_score"), "ratio"),
            format_result_value(row.get("score_min"), "ratio"),
            format_result_value(row.get("score_max"), "ratio"),
            format_result_value(row.get("score_range"), "ratio"),
            format_result_value(row.get("score_std"), "ratio"),
        ]
        for row in stability
    ]
    stability_table = _html_table(
        [
            "Parameter",
            "Shape",
            "Best value",
            "Best score",
            "Min",
            "Max",
            "Range",
            "Std",
        ],
        stability_rows,
    )

    windows = walk_forward.get("windows") or []
    wf_columns = result_view_columns(
        (item.get("test_metrics") or {}) for item in windows
    )
    wf_rows = []
    for item in windows:
        window = item.get("window") or {}
        best_train = item.get("best_train") or {}
        cells = _view_cells(item.get("test_metrics") or {})
        wf_rows.append(
            [
                f"{window.get('train_start')}..{window.get('test_end')}",
                format_result_value(best_train.get("score"), "ratio"),
                *(cells.get(column.key, "-") for column in wf_columns),
                json.dumps(best_train.get("params") or {}, sort_keys=True),
            ]
        )
    wf_table = _html_table(
        [
            "Window",
            "Train score",
            *(f"Test {column.label}" for column in wf_columns),
            "Params",
        ],
        wf_rows,
    )

    mc_rows = [
        ("Iterations", monte_carlo.get("iterations")),
        ("Trade count", monte_carlo.get("trade_count")),
        ("Trade source", monte_carlo.get("trade_source")),
        (
            "Median return",
            format_result_value(monte_carlo.get("median_return"), "ratio"),
        ),
        ("Return p05", format_result_value(monte_carlo.get("return_p05"), "ratio")),
        ("Return p95", format_result_value(monte_carlo.get("return_p95"), "ratio")),
        (
            "Median drawdown",
            format_result_value(monte_carlo.get("median_drawdown"), "ratio"),
        ),
        ("Drawdown p05", format_result_value(monte_carlo.get("drawdown_p05"), "ratio")),
        (
            "Worst drawdown",
            format_result_value(monte_carlo.get("worst_drawdown"), "ratio"),
        ),
        (
            "P(profit)",
            format_result_value(monte_carlo.get("probability_of_profit"), "ratio"),
        ),
        ("Risk of ruin", format_result_value(monte_carlo.get("risk_of_ruin"), "ratio")),
    ]
    mc_table = _html_table(["Metric", "Value"], mc_rows)

    summary_rows = [
        ("Best params", json.dumps(summary.get("best_params") or {}, sort_keys=True)),
        (f"IS {metric}", format_result_value(summary.get("is_metric"), "ratio")),
        (f"OOS {metric}", format_result_value(summary.get("oos_metric"), "ratio")),
        ("Degradation", format_result_value(summary.get("degradation"), "ratio")),
        (
            "Train/test score ratio",
            format_result_value(summary.get("train_test_score_ratio"), "ratio"),
        ),
        ("Overfit flag", summary.get("overfit_flag")),
        (
            "MC 5th-pct return",
            format_result_value(summary.get("mc_return_p05"), "ratio"),
        ),
        ("Verdict", verdict),
    ]
    summary_table = _html_table(["Field", "Value"], summary_rows)

    disclaimer = grid.get("warning") or GRID_IN_SAMPLE_DISCLAIMER
    body = f"""  <h1>{html_lib.escape(title)}</h1>
  <p class="{verdict_class}">{html_lib.escape(verdict)}</p>

  <h2>Run config</h2>
  {config_table}

  <h2>Grid search</h2>
  <p class="banner">{html_lib.escape(str(disclaimer))}</p>
  <p class="meta">Best params: {html_lib.escape(json.dumps(grid.get("best_params") or {}, sort_keys=True))}
  &nbsp;|&nbsp; Best score: {html_lib.escape(format_result_value(grid.get("best_score"), "ratio"))}</p>
  {grid_table}

  <h2>Parameter stability</h2>
  {stability_table}

  <h2>Walk-forward</h2>
  <p class="meta">Stability score: {html_lib.escape(format_result_value(walk_forward.get("stability_score"), "ratio"))}
  &nbsp;|&nbsp; Train/test ratio: {html_lib.escape(format_result_value(walk_forward.get("train_test_score_ratio"), "ratio"))}
  &nbsp;|&nbsp; Overfit flag: {html_lib.escape(str(walk_forward.get("overfit_flag")))}
  &nbsp;|&nbsp; Degradation: {html_lib.escape(format_result_value(summary.get("degradation"), "ratio"))}</p>
  {wf_table}

  <h2>Monte Carlo</h2>
  {mc_table}

  <h2>Summary</h2>
  {summary_table}
"""
    page = html_page(title, _REPORT_CSS, body, viewport=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
