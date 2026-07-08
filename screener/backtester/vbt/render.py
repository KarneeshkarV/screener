"""Rich rendering helpers for vectorbt sweep output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.vbt.config import DISCLAIMER, MetricName, WALK_FORWARD_OOS_COLUMNS
from screener.backtester.vbt.walk_forward import WalkForwardSweepSummary


def _fmt_int_or_dash(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return str(int(f))


def print_results_table(
    df: pd.DataFrame,
    *,
    top_n: int,
    metric: MetricName,
    console: Console | None = None,
) -> None:
    out = console or Console()
    out.print(DISCLAIMER)
    table = Table(title=f"Top {top_n} by {metric}")
    has_indicator = "indicator" in df.columns
    columns = (["indicator"] if has_indicator else []) + [
        "fast",
        "slow",
        "hold",
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    ]
    for col in columns:
        table.add_column(col)
    for _, row in df.head(top_n).iterrows():
        cells: list[str] = []
        if has_indicator:
            cells.append(str(row["indicator"]))
        cells.extend(
            [
                _fmt_int_or_dash(row["fast"]),
                _fmt_int_or_dash(row["slow"]),
                _fmt_int_or_dash(row["hold"]),
                f"{row['sharpe']:.3f}" if np.isfinite(row["sharpe"]) else "n/a",
                f"{row['total_return'] * 100:+.2f}%",
                f"{row['calmar']:.3f}" if np.isfinite(row["calmar"]) else "n/a",
                f"{row['max_drawdown'] * 100:+.2f}%",
                f"{row['win_rate'] * 100:.1f}%"
                if np.isfinite(row["win_rate"])
                else "n/a",
                _fmt_int_or_dash(row["trades"]),
            ]
        )
        table.add_row(*cells)
    out.print(table)


def print_walk_forward_sweep_table(
    summary: WalkForwardSweepSummary,
    *,
    console: Console | None = None,
) -> None:
    out = console or Console()
    out.print(DISCLAIMER)
    metric = summary.metric
    table = Table(title=f"Walk-Forward Sweep by {metric}")
    for col in [
        "train",
        "test",
        "indicator",
        "fast",
        "slow",
        "hold",
        f"IS {metric}",
        f"OOS {metric}",
        "OOS return",
        "OOS trades",
    ]:
        table.add_column(col)
    for _, row in summary.windows.iterrows():
        table.add_row(
            f"{row['train_start']}..{row['train_end']}",
            f"{row['test_start']}..{row['test_end']}",
            str(row["indicator"]),
            _fmt_int_or_dash(row["fast"]),
            _fmt_int_or_dash(row["slow"]),
            _fmt_int_or_dash(row["hold"]),
            f"{row['is_score']:.3f}" if np.isfinite(row["is_score"]) else "n/a",
            f"{row['oos_score']:.3f}" if np.isfinite(row["oos_score"]) else "n/a",
            f"{row['oos_total_return'] * 100:+.2f}%",
            _fmt_int_or_dash(row["oos_trades"]),
        )
    out.print(table)
    if summary.windows.empty:
        out.print("[yellow]No walk-forward windows produced results.[/yellow]")
        return
    agg = summary.aggregate_oos_metrics
    parts: list[str] = []
    for col in WALK_FORWARD_OOS_COLUMNS:
        if col not in agg:
            continue
        if col in ("total_return", "max_drawdown"):
            parts.append(f"{col}={agg[col] * 100:+.2f}%")
        elif col == "win_rate":
            parts.append(f"{col}={agg[col] * 100:.1f}%")
        else:
            parts.append(f"{col}={agg[col]:.3f}")
    parts.append(f"trades={int(agg.get('trades', 0))}")
    out.print("Aggregate OOS (trade-weighted): " + "  ".join(parts))
    out.print(
        f"Aggregate IS {metric}: {summary.aggregate_is_score:.3f}  "
        f"Aggregate OOS {metric}: {summary.aggregate_oos_score:.3f}  "
        f"WF efficiency (OOS/IS): {summary.efficiency:.3f}  "
        f"Parameter stability: {summary.parameter_stability:.3f}"
    )
