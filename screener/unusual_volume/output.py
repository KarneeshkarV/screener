"""Rendering helpers for unusual-volume events.

Three output sinks: rich table to stdout, JSON file (full schema), and
Markdown summary. Mirrors the pattern in ``scan_today.py``.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.table import Table

from .detector import Event


_STRENGTH_RANK = {"EXTREME": 3, "HIGH": 2, "MODERATE": 1}


def _fmt_volume(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:+.2f}%"


def _fmt_float(v, ndp: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{ndp}f}"


def _fmt_mcap(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v:,.0f}"


def sort_events(events: Iterable[Event]) -> list[Event]:
    """Sort by strength desc, then RVOL desc."""
    return sorted(
        events,
        key=lambda e: (
            _STRENGTH_RANK.get(e.strength, 0),
            e.rvol if not (isinstance(e.rvol, float) and math.isnan(e.rvol)) else 0.0,
        ),
        reverse=True,
    )


def render_rich(events: list[Event], market: str, as_of, console: Console) -> None:
    if not events:
        console.print(
            f"[dim]No unusual-volume events on {as_of} for {market.upper()}.[/dim]"
        )
        return
    sorted_events = sort_events(events)
    is_india = market == "india"

    title = (
        f"[bold]Unusual Volume — {market.upper()}[/bold]  "
        f"[dim]as of {as_of} • {len(events)} events[/dim]"
    )
    console.print(title)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker", no_wrap=True)
    table.add_column("Dir", no_wrap=True)
    table.add_column("Strength", no_wrap=True)
    table.add_column("Close", justify="right")
    table.add_column("Chg", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("RVOL", justify="right")
    table.add_column("Z", justify="right")
    if is_india:
        table.add_column("Deliv%", justify="right")
        table.add_column("DelvRVOL", justify="right")
        table.add_column("Conv", justify="right")
    table.add_column("Build", justify="right")
    table.add_column("Sector", no_wrap=False, max_width=20)
    table.add_column("Notes", no_wrap=False, max_width=40)

    for ev in sorted_events:
        row = [
            ev.symbol,
            _color_direction(ev.direction),
            _color_strength(ev.strength),
            _fmt_float(ev.close),
            _fmt_pct(ev.pct_change),
            _fmt_volume(ev.volume),
            _fmt_float(ev.rvol),
            _fmt_float(ev.z_score),
        ]
        if is_india:
            row.extend(
                [
                    _fmt_float(ev.delivery_pct, 1),
                    _fmt_float(ev.delivery_rvol),
                    _fmt_float(ev.conviction_score),
                ]
            )
        row.append(_fmt_float(ev.buildup_score, 3))
        row.extend([ev.sector or "-", ev.notes or "-"])
        table.add_row(*row)
    console.print(table)


def _color_direction(d: str) -> str:
    return {
        "BUYING": "[green]BUYING[/green]",
        "SELLING": "[red]SELLING[/red]",
        "REVERSAL": "[yellow]REVERSAL[/yellow]",
        "CHURN": "[dim]CHURN[/dim]",
        "QUIET_ACCUMULATION": "[cyan]QUIET ACC[/cyan]",
        "BUILDUP": "[magenta]BUILDUP[/magenta]",
    }.get(d, d)


def _color_strength(s: str) -> str:
    return {
        "EXTREME": "[bold red]EXTREME[/bold red]",
        "HIGH": "[bold]HIGH[/bold]",
        "MODERATE": "[dim]MODERATE[/dim]",
    }.get(s, s)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Real) and not isinstance(value, bool):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        if isinstance(value, Integral):
            return int(value)
        return as_float
    return value


def write_json(events: list[Event], path: Path) -> None:
    payload = [_json_safe(ev.to_dict()) for ev in sort_events(events)]
    Path(path).write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))


def write_markdown(
    events: list[Event], path: Path, market: str, as_of
) -> None:
    sorted_events = sort_events(events)
    is_india = market == "india"
    lines: list[str] = []
    lines.append(f"# Unusual Volume — {market.upper()} ({as_of})")
    lines.append("")
    lines.append(f"**Events:** {len(events)}")
    lines.append("")

    buckets = {
        "BUYING": [e for e in sorted_events if e.direction == "BUYING"],
        "SELLING": [e for e in sorted_events if e.direction == "SELLING"],
        "REVERSAL": [e for e in sorted_events if e.direction == "REVERSAL"],
        "CHURN": [e for e in sorted_events if e.direction == "CHURN"],
    }
    if is_india:
        buckets["QUIET_ACCUMULATION"] = [
            e for e in sorted_events if e.direction == "QUIET_ACCUMULATION"
        ]
    buildups = [e for e in sorted_events if e.direction == "BUILDUP"]
    if buildups:
        buckets["BUILDUP"] = buildups

    for label, evs in buckets.items():
        if not evs:
            continue
        lines.append(f"## {label} ({len(evs)})")
        lines.append("")
        if label == "BUILDUP":
            # Build-ups don't have meaningful RVOL/Z (they failed the volume
            # filter), so render score + flags instead.
            lines.append(
                "| # | Ticker | Score | Flags | Close | Chg | Sector |"
            )
            lines.append(
                "|---|--------|-----:|-------|------:|----:|--------|"
            )
            for i, ev in enumerate(_sort_by_buildup(evs)[:25], 1):
                lines.append(
                    "| " + " | ".join([
                        str(i),
                        f"**{ev.symbol}**",
                        _fmt_float(ev.buildup_score, 3),
                        ", ".join(ev.buildup_flags or []) or "-",
                        _fmt_float(ev.close),
                        _fmt_pct(ev.pct_change),
                        ev.sector or "-",
                    ]) + " |"
                )
            lines.append("")
            continue
        if is_india:
            lines.append(
                "| # | Ticker | Strength | Close | Chg | RVOL | Z | Deliv% | DelvRVOL | Conv | Build | Sector |"
            )
            lines.append(
                "|---|--------|----------|------:|----:|-----:|--:|-------:|---------:|-----:|------:|--------|"
            )
        else:
            lines.append(
                "| # | Ticker | Strength | Close | Chg | Volume | RVOL | Z | Build | Sector |"
            )
            lines.append(
                "|---|--------|----------|------:|----:|-------:|-----:|--:|------:|--------|"
            )
        for i, ev in enumerate(evs[:25], 1):
            base = [
                str(i),
                f"**{ev.symbol}**",
                ev.strength,
                _fmt_float(ev.close),
                _fmt_pct(ev.pct_change),
            ]
            if is_india:
                row = base + [
                    _fmt_float(ev.rvol),
                    _fmt_float(ev.z_score),
                    _fmt_float(ev.delivery_pct, 1),
                    _fmt_float(ev.delivery_rvol),
                    _fmt_float(ev.conviction_score),
                    _fmt_float(ev.buildup_score, 3),
                    ev.sector or "-",
                ]
            else:
                row = base + [
                    _fmt_volume(ev.volume),
                    _fmt_float(ev.rvol),
                    _fmt_float(ev.z_score),
                    _fmt_float(ev.buildup_score, 3),
                    ev.sector or "-",
                ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def _sort_by_buildup(evs: list[Event]) -> list[Event]:
    return sorted(
        evs,
        key=lambda e: (
            e.buildup_score if e.buildup_score is not None else 0.0
        ),
        reverse=True,
    )
