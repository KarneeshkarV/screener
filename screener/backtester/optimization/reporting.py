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


def print_grid_table(
    results: Iterable[GridSearchResult], console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(title="Grid Search Results", show_header=True, header_style="bold")
    for col in [
        "Rank",
        "Score",
        "Trades",
        "Sharpe",
        "Profit Factor",
        "Max DD",
        "Params",
    ]:
        table.add_column(col, justify="right" if col != "Params" else "left")
    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{result.score:.4f}",
            str(result.trade_count),
            f"{float(result.metrics.get('sharpe', 0.0)):.3f}",
            f"{float(result.metrics.get('profit_factor', 0.0)):.3f}",
            f"{float(result.metrics.get('max_drawdown', 0.0)) * 100:.2f}%",
            json.dumps(result.params, sort_keys=True),
        )
    console.print(table)
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
    table = Table(title="Walk-Forward Results", show_header=True, header_style="bold")
    for col in ["Window", "Train Score", "Test Sharpe", "Test Trades", "Params"]:
        table.add_column(
            col, justify="right" if col != "Params" and col != "Window" else "left"
        )
    for result in summary.windows:
        w = result.window
        table.add_row(
            f"{w.train_start}..{w.test_end}",
            f"{result.best_train.score:.4f}",
            f"{float(result.test_metrics.get('sharpe', 0.0)):.3f}",
            str(result.test_trade_count),
            json.dumps(result.best_train.params, sort_keys=True),
        )
    console.print(table)
    console.print(
        f"Stability: {summary.stability_score:.3f}  "
        f"Train/Test score ratio: {summary.train_test_score_ratio:.3f}  "
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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


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
        [(k, _fmt(v) if not isinstance(v, str) else v) for k, v in config_rows],
    )

    grid_results = grid.get("results") or []
    grid_rows = []
    for rank, result in enumerate(grid_results, start=1):
        metrics = result.get("metrics") or {}
        grid_rows.append(
            [
                rank,
                _fmt(result.get("score")),
                result.get("trade_count"),
                _fmt(metrics.get("sharpe"), 3),
                _fmt(metrics.get("profit_factor"), 3),
                _fmt(
                    (metrics.get("max_drawdown") or 0.0) * 100
                    if isinstance(metrics.get("max_drawdown"), (int, float))
                    else metrics.get("max_drawdown"),
                    2,
                )
                + (
                    "%" if isinstance(metrics.get("max_drawdown"), (int, float)) else ""
                ),
                json.dumps(result.get("params") or {}, sort_keys=True),
            ]
        )
    grid_table = _html_table(
        ["Rank", "Score", "Trades", "Sharpe", "Profit Factor", "Max DD", "Params"],
        grid_rows,
    )

    stability = grid.get("stability") or []
    stability_rows = [
        [
            row.get("parameter"),
            row.get("shape"),
            row.get("best_value"),
            _fmt(row.get("best_score")),
            _fmt(row.get("score_min")),
            _fmt(row.get("score_max")),
            _fmt(row.get("score_range")),
            _fmt(row.get("score_std")),
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
    wf_rows = []
    for item in windows:
        window = item.get("window") or {}
        best_train = item.get("best_train") or {}
        test_metrics = item.get("test_metrics") or {}
        wf_rows.append(
            [
                f"{window.get('train_start')}..{window.get('test_end')}",
                _fmt(best_train.get("score")),
                _fmt(test_metrics.get(metric), 3),
                _fmt(test_metrics.get("sharpe"), 3),
                item.get("test_trade_count"),
                json.dumps(best_train.get("params") or {}, sort_keys=True),
            ]
        )
    wf_table = _html_table(
        [
            "Window",
            "Train score",
            f"Test {metric}",
            "Test Sharpe",
            "Test trades",
            "Params",
        ],
        wf_rows,
    )

    mc_rows = [
        ("Iterations", monte_carlo.get("iterations")),
        ("Trade count", monte_carlo.get("trade_count")),
        ("Trade source", monte_carlo.get("trade_source")),
        ("Median return", _fmt(monte_carlo.get("median_return"))),
        ("Return p05", _fmt(monte_carlo.get("return_p05"))),
        ("Return p95", _fmt(monte_carlo.get("return_p95"))),
        ("Median drawdown", _fmt(monte_carlo.get("median_drawdown"))),
        ("Drawdown p05", _fmt(monte_carlo.get("drawdown_p05"))),
        ("Worst drawdown", _fmt(monte_carlo.get("worst_drawdown"))),
        ("P(profit)", _fmt(monte_carlo.get("probability_of_profit"), 3)),
        ("Risk of ruin", _fmt(monte_carlo.get("risk_of_ruin"), 3)),
    ]
    mc_table = _html_table(["Metric", "Value"], mc_rows)

    summary_rows = [
        ("Best params", json.dumps(summary.get("best_params") or {}, sort_keys=True)),
        (f"IS {metric}", _fmt(summary.get("is_metric"))),
        (f"OOS {metric}", _fmt(summary.get("oos_metric"))),
        ("Degradation", _fmt(summary.get("degradation"), 3)),
        ("Train/test score ratio", _fmt(summary.get("train_test_score_ratio"), 3)),
        ("Overfit flag", summary.get("overfit_flag")),
        ("MC 5th-pct return", _fmt(summary.get("mc_return_p05"))),
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
  &nbsp;|&nbsp; Best score: {html_lib.escape(_fmt(grid.get("best_score")))}</p>
  {grid_table}

  <h2>Parameter stability</h2>
  {stability_table}

  <h2>Walk-forward</h2>
  <p class="meta">Stability score: {html_lib.escape(_fmt(walk_forward.get("stability_score"), 3))}
  &nbsp;|&nbsp; Train/test ratio: {html_lib.escape(_fmt(walk_forward.get("train_test_score_ratio"), 3))}
  &nbsp;|&nbsp; Overfit flag: {html_lib.escape(str(walk_forward.get("overfit_flag")))}
  &nbsp;|&nbsp; Degradation: {html_lib.escape(_fmt(summary.get("degradation"), 3))}</p>
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
