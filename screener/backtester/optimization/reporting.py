"""Console, JSON, and HTML reports for optimization results."""

from __future__ import annotations

import html as html_lib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

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
from screener.backtester.optimization.walk_forward import (
    CAPITAL_POLICY,
    FOLD_BOUNDARY_POLICY,
    WINDOW_LENGTH_UNIT,
    WalkForwardSummary,
)
from screener.html_report import html_page
from screener.report_theme import bar_chart, chart_html, signed_colors

GRID_IN_SAMPLE_DISCLAIMER = (
    "IN-SAMPLE / SELECTION BIAS WARNING: These metrics are computed on the SAME "
    "data the grid selected the best parameters from. The headline best-of-grid "
    "Sharpe is optimistically biased and is NOT an out-of-sample estimate of "
    "future performance. Use `optimize walk-forward` for an honest, out-of-sample "
    "assessment before trusting these parameters."
)

EVIDENCE_INTEGRITY_NOTE = (
    "Explicit evidence thresholds are minimum data-integrity checks, not "
    "statistical proof of alpha or live-trading approval."
)

_REPORT_CSS = """
    .panel { margin-bottom: var(--gap); }
    .panel > h2 { margin-bottom: 12px; }
    .verdict {
      margin: 0;
      padding: 12px 16px;
      font-weight: 600;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: var(--surface-2);
    }
    .verdict.pass {
      border-color: color-mix(in oklab, var(--status-good) 50%, var(--line));
      background: color-mix(in oklab, var(--status-good) 16%, var(--surface));
    }
    .verdict.caution {
      border-color: color-mix(in oklab, var(--status-warning) 55%, var(--line));
      background: color-mix(in oklab, var(--status-warning) 16%, var(--surface));
    }
    .verdict.fail {
      border-color: color-mix(in oklab, var(--status-critical) 50%, var(--line));
      background: color-mix(in oklab, var(--status-critical) 16%, var(--surface));
    }
    .callout { margin: 0 0 10px; }
    .meta { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
    .status-good td {
      background: color-mix(in oklab, var(--status-good) 16%, var(--surface));
    }
    .status-warning td {
      background: color-mix(in oklab, var(--status-warning) 16%, var(--surface));
    }
    .status-critical td {
      background: color-mix(in oklab, var(--status-critical) 16%, var(--surface));
    }
    .status-neutral td { background: var(--surface-2); }
"""

# Metrics stored as fractions, shown as percentages on chart axes and labels.
_PERCENT_METRICS = frozenset({"total_return", "cagr", "max_drawdown"})


def _json_safe(value: Any) -> Any:
    """Replace the float values JSON has no syntax for.

    ``json.dumps`` writes ``NaN`` and ``Infinity`` as bare tokens by default,
    which are not JSON and which strict parsers reject, so a report written
    that way only loads back in Python. Non-finite floats become null, which
    is what a missing metric means and what the terminal and HTML renderers
    already show as "-".
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    # Reached by the numpy scalar types, which are not ``float`` subclasses and
    # so slip past :func:`_json_safe`.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return None
    return str(value)


def write_json_report(data: Any, path: Path | str) -> None:
    """Write a report payload as JSON that a strict parser will accept.

    :func:`_json_safe` rewrites the non-finite floats first and ``allow_nan``
    is off behind it, so anything it missed fails loudly here rather than
    reaching the file as a bare ``NaN`` token no strict parser accepts.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(data), indent=2, default=_json_default, allow_nan=False)
    )


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
    evidence = summary.evidence or {}
    window_unit = evidence.get("window_length_unit") or WINDOW_LENGTH_UNIT
    boundary = summary.fold_boundary_policy or FOLD_BOUNDARY_POLICY
    capital = summary.capital_policy or CAPITAL_POLICY
    console.print(
        "Stability: "
        f"{format_result_value(summary.stability_score, 'ratio')}  "
        "Train/Test score ratio: "
        f"{format_result_value(summary.train_test_score_ratio, 'ratio')}  "
        f"Overfit flag: {summary.overfit_flag}  "
        f"Insufficient data: {summary.insufficient_data}"
    )
    console.print(
        f"Window lengths: {window_unit}  Fold boundary: {boundary}  Capital: {capital}"
    )
    if summary.insufficient_data:
        console.print(
            "[bold yellow]INSUFFICIENT DATA[/bold yellow]: combined OOS evidence "
            "does not meet the minimum criteria."
        )
    console.print(
        "[dim]Evidence integrity PASS/FAIL here is a data-adequacy check, "
        "not validated alpha or live-trading approval.[/dim]"
    )
    for warning in evidence.get("holding_period_warnings") or []:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _header(eyebrow: str, title: str, subhead: Sequence[str]) -> str:
    items = "".join(f"<span>{html_lib.escape(item)}</span>" for item in subhead if item)
    return (
        "<header>"
        f'<div class="eyebrow">{html_lib.escape(eyebrow)}</div>'
        f"<h1>{html_lib.escape(title)}</h1>"
        f'<div class="subhead">{items}</div>'
        "</header>"
    )


def _callout(text: str) -> str:
    return (
        '<div class="callout"><span class="badge warning">! Warning</span> '
        f"{html_lib.escape(text)}</div>"
    )


def _sign_status(value: Any) -> str | None:
    """Good/critical/neutral by sign; None for missing or non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    if value > 0:
        return "good"
    if value < 0:
        return "critical"
    return "neutral"


def _flag_status(value: Any, bad: str) -> str | None:
    if value is None:
        return None
    return bad if value else "good"


def _verdict_status(verdict: str) -> str | None:
    if verdict.startswith("PASS"):
        return "good"
    if verdict.startswith("CAUTION"):
        return "warning"
    if verdict.startswith("FAIL"):
        return "critical"
    return None


def write_html_report(
    data: Any,
    path: Path | str,
    title: str = "Optimization Report",
    disclaimer: str | None = None,
) -> None:
    payload = html_lib.escape(json.dumps(data, indent=2, default=_json_default))
    warning = (
        f'<section class="panel">{_callout(disclaimer)}</section>\n'
        if disclaimer
        else ""
    )
    body = (
        f"{_header('Optimization report', title, [])}\n"
        "<main>\n"
        f"{warning}"
        '<section class="panel">\n'
        "<h2>Report data</h2>\n"
        f"<pre>{payload}</pre>\n"
        "</section>\n"
        "</main>"
    )
    html = html_page(title, _REPORT_CSS, body, viewport=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    row_status: Sequence[str | None] | None = None,
) -> str:
    head = "".join(f"<th>{html_lib.escape(str(h))}</th>" for h in headers)
    body_rows: list[str] = []
    for index, row in enumerate(rows):
        status = row_status[index] if row_status is not None else None
        row_attr = f' class="status-{status}"' if status else ""
        cells = "".join(f"<td>{html_lib.escape(str(c))}</td>" for c in row)
        body_rows.append(f"<tr{row_attr}>{cells}</tr>")
    return (
        '<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def write_research_html_report(data: Mapping[str, Any], path: Path | str) -> None:
    """Combined single-page research report (config / grid / WF / MC).

    Renders on the shared Slate theme: one ``.panel`` per section, status-tinted
    rows via token classes, and ECharts charts for the top grid combos and the
    per-window out-of-sample metric.
    """
    title = "Research Report"
    config = data.get("config") or {}
    grid = data.get("grid") or {}
    walk_forward = data.get("walk_forward") or {}
    monte_carlo = data.get("monte_carlo") or {}
    summary = data.get("summary") or {}
    metric = str(config.get("metric") or "sharpe")

    verdict = str(summary.get("verdict") or "")
    verdict_state = _verdict_status(verdict)
    verdict_class = "verdict" + (
        {"good": " pass", "warning": " caution", "critical": " fail"}.get(
            verdict_state or "", ""
        )
    )

    config_rows = [
        ("Market", config.get("market")),
        ("Strategy", config.get("strategy_name")),
        ("Entry", config.get("entry_expr")),
        ("Exit", config.get("exit_expr")),
        ("Start", config.get("start_date")),
        ("End", config.get("end_date")),
        (
            "Train / test / step (calendar days)",
            (
                f"{config.get('train_days')} / {config.get('test_days')} / "
                f"{config.get('step_days')}"
            ),
        ),
        (
            "Window length unit",
            config.get("window_length_unit") or WINDOW_LENGTH_UNIT,
        ),
        (
            "Fold boundary",
            config.get("fold_boundary_policy") or FOLD_BOUNDARY_POLICY,
        ),
        (
            "Capital policy",
            config.get("capital_policy") or CAPITAL_POLICY,
        ),
        ("Experiment id", config.get("experiment_id")),
        ("Trial db", config.get("trial_db_path")),
        ("Metric", metric),
        ("Annual risk-free hurdle", config.get("risk_free_rate", 0.0)),
        (
            "Parameter grid",
            json.dumps(config.get("parameter_grid") or {}, sort_keys=True),
        ),
        ("Tickers", config.get("tickers")),
        ("MC iterations", config.get("mc_iterations")),
        (
            "Holding-period warnings",
            "; ".join(config.get("holding_period_warnings") or []) or None,
        ),
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

    # Results arrive in rank order from the grid, so the top 20 are the first 20.
    top_combos = grid_results[:20]
    grid_chart = ""
    if top_combos:
        combo_labels = [
            f"#{rank} "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted((result.get("params") or {}).items())
            )
            for rank, result in enumerate(top_combos, start=1)
        ]
        combo_scores = [result.get("score") for result in top_combos]
        grid_option = bar_chart(
            combo_labels,
            combo_scores,
            name=metric,
            value_format="num2",
            horizontal=True,
            colors=signed_colors(combo_scores),
            labels=True,
        )
        # Leave room past the most negative bar so its value label does not
        # run into the category labels on the left edge.
        scores = [float(v) for v in combo_scores if isinstance(v, int | float)]
        if scores and min(scores) < 0:
            grid_option["xAxis"]["min"] = round(min(scores) * 1.4, 3)
        grid_chart = chart_html(
            "grid-top-chart",
            grid_option,
            height=max(220, 26 * len(top_combos) + 60),
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
    wf_status: list[str | None] = []
    wf_labels: list[str] = []
    wf_oos: list[Any] = []
    for item in windows:
        window = item.get("window") or {}
        best_train = item.get("best_train") or {}
        test_metrics = item.get("test_metrics") or {}
        label = f"{window.get('train_start')}..{window.get('test_end')}"
        cells = _view_cells(test_metrics)
        wf_rows.append(
            [
                label,
                format_result_value(best_train.get("score"), "ratio"),
                *(cells.get(column.key, "-") for column in wf_columns),
                json.dumps(best_train.get("params") or {}, sort_keys=True),
            ]
        )
        oos_value = test_metrics.get(metric)
        wf_labels.append(label)
        wf_oos.append(oos_value)
        wf_status.append(_sign_status(oos_value))
    wf_table = _html_table(
        [
            "Window",
            "Train score",
            *(f"Test {column.label}" for column in wf_columns),
            "Params",
        ],
        wf_rows,
        wf_status,
    )
    wf_chart = ""
    if wf_oos:
        wf_chart = chart_html(
            "wf-oos-chart",
            bar_chart(
                wf_labels,
                wf_oos,
                name=f"OOS {metric}",
                value_format="pct1" if metric in _PERCENT_METRICS else "num2",
                colors=signed_colors(wf_oos),
                labels=True,
            ),
            height=280,
        )

    mc_rows = [
        ("Iterations", monte_carlo.get("iterations")),
        ("Source", monte_carlo.get("source") or monte_carlo.get("trade_source")),
        ("Method", monte_carlo.get("method")),
        ("Equity bars", monte_carlo.get("equity_bars")),
        ("Block", monte_carlo.get("block")),
        ("OOS trade count", monte_carlo.get("trade_count")),
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

    integrity_note = str(
        summary.get("evidence_integrity_note")
        or data.get("evidence_integrity_note")
        or config.get("evidence_integrity_note")
        or EVIDENCE_INTEGRITY_NOTE
    )
    summary_rows = [
        (
            "Best params",
            json.dumps(summary.get("best_params") or {}, sort_keys=True),
            None,
        ),
        (
            f"IS {metric}",
            format_result_value(summary.get("is_metric"), "ratio"),
            None,
        ),
        (
            f"OOS {metric}",
            format_result_value(summary.get("oos_metric"), "ratio"),
            _sign_status(summary.get("oos_metric")),
        ),
        ("Degradation", format_result_value(summary.get("degradation"), "ratio"), None),
        (
            "Train/test score ratio",
            format_result_value(summary.get("train_test_score_ratio"), "ratio"),
            None,
        ),
        (
            "Overfit flag",
            summary.get("overfit_flag"),
            _flag_status(summary.get("overfit_flag"), "warning"),
        ),
        (
            "Insufficient data",
            summary.get("insufficient_data"),
            _flag_status(summary.get("insufficient_data"), "critical"),
        ),
        (
            "MC 5th-pct return",
            format_result_value(summary.get("mc_return_p05"), "ratio"),
            _sign_status(summary.get("mc_return_p05")),
        ),
        ("MC source", summary.get("mc_source"), None),
        ("MC method", summary.get("mc_method"), None),
        (
            "MC reason",
            summary.get("mc_reason") or monte_carlo.get("reason"),
            None,
        ),
        ("Evidence note", integrity_note, None),
        ("Verdict", verdict, verdict_state),
    ]
    summary_table = _html_table(
        ["Field", "Value"],
        [(label, value) for label, value, _ in summary_rows],
        [status for _, _, status in summary_rows],
    )

    disclaimer = grid.get("warning") or GRID_IN_SAMPLE_DISCLAIMER
    grid_role = grid.get("role") or "descriptive_full_period_only"
    wf_insufficient = bool(
        walk_forward.get("insufficient_data") or summary.get("insufficient_data")
    )
    wf_evidence = walk_forward.get("evidence") or {}
    holding_warnings = (
        config.get("holding_period_warnings")
        or wf_evidence.get("holding_period_warnings")
        or []
    )

    header = _header(
        "Research report",
        title,
        [
            f"Strategy: {config.get('strategy_name') or '-'}",
            f"Market: {config.get('market') or '-'}",
            f"Metric: {metric}",
        ],
    )
    sections = [
        (
            '<section class="panel">'
            "<h2>Verdict</h2>"
            f'<p class="{verdict_class}">{html_lib.escape(verdict)}</p>'
            f'<p class="meta">{html_lib.escape(integrity_note)}</p>'
            "</section>"
        ),
        (f'<section class="panel"><h2>Run config</h2>{config_table}</section>'),
        (
            '<section class="panel">'
            "<h2>Grid search (descriptive full-period only)</h2>"
            f"{_callout(str(disclaimer))}"
            '<p class="meta">'
            f"Role: {html_lib.escape(str(grid_role))}"
            " &nbsp;|&nbsp; Descriptive best params: "
            f"{html_lib.escape(json.dumps(grid.get('best_params') or {}, sort_keys=True))}"
            " &nbsp;|&nbsp; Descriptive best score: "
            f"{html_lib.escape(format_result_value(grid.get('best_score'), 'ratio'))}"
            "</p>"
            f"{grid_chart}"
            f"{grid_table}"
            "</section>"
        ),
        (
            '<section class="panel">'
            "<h2>Parameter stability</h2>"
            f"{stability_table}"
            "</section>"
        ),
        (
            '<section class="panel">'
            "<h2>Walk-forward (per-fold selection on original grid)</h2>"
            '<p class="meta">'
            "Stability score: "
            f"{html_lib.escape(format_result_value(walk_forward.get('stability_score'), 'ratio'))}"
            " &nbsp;|&nbsp; Train/test ratio: "
            f"{html_lib.escape(format_result_value(walk_forward.get('train_test_score_ratio'), 'ratio'))}"
            " &nbsp;|&nbsp; Overfit flag: "
            f"{html_lib.escape(str(walk_forward.get('overfit_flag')))}"
            " &nbsp;|&nbsp; Insufficient data: "
            f"{html_lib.escape(str(wf_insufficient))}"
            " &nbsp;|&nbsp; Window unit: "
            f"{html_lib.escape(str(config.get('window_length_unit') or wf_evidence.get('window_length_unit') or WINDOW_LENGTH_UNIT))}"
            " &nbsp;|&nbsp; Boundary: "
            f"{html_lib.escape(str(walk_forward.get('fold_boundary_policy') or wf_evidence.get('fold_boundary_policy') or FOLD_BOUNDARY_POLICY))}"
            " &nbsp;|&nbsp; Capital: "
            f"{html_lib.escape(str(walk_forward.get('capital_policy') or wf_evidence.get('capital_policy') or CAPITAL_POLICY))}"
            " &nbsp;|&nbsp; Trial scope: "
            f"{html_lib.escape(str(wf_evidence.get('trial_register_scope') or 'train_folds_only'))}"
            "</p>"
            '<p class="meta">Integrity PASS is a minimum data check only; it is not '
            "validated alpha or live-trading approval.</p>"
            + "".join(_callout(str(warning)) for warning in holding_warnings)
            + f"{wf_chart}"
            + f"{wf_table}"
            + "</section>"
        ),
        (
            '<section class="panel">'
            "<h2>Monte Carlo (OOS equity block bootstrap)</h2>"
            f"{mc_table}"
            "</section>"
        ),
        (f'<section class="panel"><h2>Summary</h2>{summary_table}</section>'),
    ]
    body = f"{header}\n<main>\n" + "\n".join(sections) + "\n</main>"
    page = html_page(
        title,
        _REPORT_CSS,
        body,
        viewport=True,
        charts=bool(grid_chart or wf_chart),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
