"""Static HTML reports for the main screen command."""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from screener.backtester.dashboard import table_html
from screener.display import COLUMN_LABELS, COLUMNS, _format_value
from screener.html_report import html_page
from screener.report_theme import (
    bar_chart,
    chart_html,
    histogram_chart,
    signed_colors,
)

_TOP_CHANGE_BARS = 25
_TOP_CHANGE_ROW_PX = 22
_TOP_CHANGE_PAD_PX = 80
_CHART_HEIGHT_PX = 320


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _summary_cards(
    *,
    market: str,
    criteria_name: str,
    total: int,
    shown: int,
    added: Sequence[str],
    removed: Sequence[str],
) -> str:
    cards = [
        ("Market", market.upper()),
        ("Criteria", criteria_name),
        ("Matches", total),
        ("Shown", shown),
        ("Added", len(added)),
        ("Removed", len(removed)),
    ]
    parts = []
    for label, value in cards:
        if label == "Criteria":
            # A criteria name is one long token; truncate it so the card keeps its grid width.
            parts.append(
                f'<article class="metric"><span>{html.escape(label)}</span>'
                f'<strong class="truncate" title="{html.escape(criteria_name)}">'
                f"{_fmt(value)}</strong></article>"
            )
        else:
            parts.append(
                f'<article class="metric"><span>{html.escape(label)}</span>'
                f"<strong>{_fmt(value)}</strong></article>"
            )
    return "".join(parts)


def _format_screen_cell(col: str, value: object, market: str) -> str:
    if col in COLUMNS:
        return _format_value(col, value, market)
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:,.2f}"
    return str(value)


def _screen_table_frame(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Format screen rows with CLI column formatters so HTML skips scientific notation."""
    formatted = pd.DataFrame(index=df.index)
    for col in df.columns:
        label = COLUMN_LABELS.get(col, str(col))
        formatted[label] = [
            _format_screen_cell(col, v, market) for v in df[col].tolist()
        ]
    return formatted


def _describe_numeric(
    df: pd.DataFrame, columns: Sequence[str], market: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": COLUMN_LABELS.get(col, col),
                "min": _format_screen_cell(col, float(series.min()), market),
                "median": _format_screen_cell(col, float(series.median()), market),
                "max": _format_screen_cell(col, float(series.max()), market),
            }
        )
    return pd.DataFrame(rows)


def _top_change_height(n_bars: int) -> int:
    return max(_CHART_HEIGHT_PX, _TOP_CHANGE_ROW_PX * n_bars + _TOP_CHANGE_PAD_PX)


def _ticker_list(title: str, tickers: Sequence[str], section_id: str) -> str:
    if not tickers:
        body = '<p class="empty">None.</p>'
    else:
        body = (
            '<div class="chips">'
            + "".join(f"<span>{html.escape(t)}</span>" for t in tickers)
            + "</div>"
        )
    return f'<section class="panel" id="{section_id}"><h2>{title}</h2>{body}</section>'


def render_screen_report(
    df: pd.DataFrame,
    total: int,
    market: str,
    criteria_name: str,
    output_file: str | Path,
    *,
    added: Sequence[str] = (),
    removed: Sequence[str] = (),
    first_run: bool = False,
    detail: bool = False,
    refresh: bool = False,
    cache_ttl: str = "",
    order_by: str = "",
) -> Path:
    """Render a static, self-contained screen report and return its path."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    if not df.empty and "setup_score" in df.columns:
        option = histogram_chart(
            df["setup_score"].tolist(),
            bins=20,
            name="Count",
            value_format="num1",
            signed=False,
        )
        sections.append(
            '<section class="panel" id="setup-score-distribution">'
            "<h2>Setup Score Distribution</h2>"
            + chart_html(
                "screen-setup-score-distribution", option, height=_CHART_HEIGHT_PX
            )
            + "</section>"
        )
    if not df.empty and "change" in df.columns:
        ranked = df.copy()
        ranked["change"] = pd.to_numeric(ranked["change"], errors="coerce")
        ranked = ranked.dropna(subset=["change"]).sort_values("change", ascending=False)
        if not ranked.empty:
            top = ranked.head(_TOP_CHANGE_BARS).copy()
            if "name" not in top.columns:
                top["name"] = top.index.astype(str)
            top["name"] = top["name"].astype(str)
            changes = top["change"].tolist()
            # Horizontal bar_chart puts the first category at the top, and the
            # frame is already sorted largest first.
            option = bar_chart(
                top["name"].tolist(),
                changes,
                name="Change %",
                value_format="pctv1",
                horizontal=True,
                colors=signed_colors(changes),
                labels=True,
            )
            sections.append(
                '<section class="panel" id="top-change">'
                "<h2>Top Change</h2>"
                + chart_html(
                    "screen-top-change", option, height=_top_change_height(len(top))
                )
                + "</section>"
            )

    numeric_summary = _describe_numeric(
        df,
        ["close", "change", "volume", "market_cap_basic", "setup_score", "RSI"],
        market,
    )
    notes = [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Sort: {order_by or '-'}",
        f"Detail: {detail}",
        f"Refresh: {refresh}",
        f"Cache TTL: {cache_ttl or '-'}",
    ]
    if first_run:
        notes.append("No prior run was available; this run was saved as the baseline.")
    note_items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    title = f"{criteria_name.upper()} Screen"
    subhead = (
        f"{html.escape(market.upper())} market, {html.escape(criteria_name)} criteria"
    )
    _css = """\
    main {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }
    .metrics, .wide { grid-column: 1 / -1; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chips span {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      background: var(--surface-2);
      color: var(--ink);
      font-size: 12px;
    }
    .metric strong.truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .notes { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .metrics, .wide { grid-column: auto; }
    }"""
    page_html = html_page(
        html.escape(title),
        _css,
        f"""\
  <header>
    <div class="eyebrow">Screen report</div>
    <h1>{html.escape(title)}</h1>
    <div class="subhead">{subhead}</div>
  </header>
  <main>
    <section class="metrics" id="screen-summary">{_summary_cards(market=market, criteria_name=criteria_name, total=total, shown=len(df), added=added, removed=removed)}</section>
    {"".join(sections)}
    <section class="panel wide" id="numeric-summary"><h2>Important Metrics</h2><div class="table-wrap">{table_html(numeric_summary, "screen-numeric-summary")}</div></section>
    <section class="panel wide" id="screen-results"><h2>Results</h2><div class="table-wrap">{table_html(_screen_table_frame(df, market), "screen-results-table", limit=500)}</div></section>
    {_ticker_list("Added Since Previous Run", added, "added-tickers")}
    {_ticker_list("Removed Since Previous Run", removed, "removed-tickers")}
    <section class="panel wide" id="report-notes"><h2>Notes</h2><ul class="notes">{note_items}</ul></section>
  </main>""",
        charts=True,
    )
    output_path.write_text(page_html, encoding="utf-8")
    return output_path
