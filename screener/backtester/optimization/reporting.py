"""Console, JSON, and HTML reports for optimization results."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

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
