"""Static HTML reports for the main screen command."""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs

from screener.backtester.dashboard import figure_html, table_html
from screener.display import COLUMN_LABELS, COLUMNS, _format_value
from screener.html_report import html_page

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
    return "".join(
        f'<article class="metric"><span>{html.escape(label)}</span>'
        f"<strong>{_fmt(value)}</strong></article>"
        for label, value in cards
    )


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
        fig = px.histogram(
            df,
            x="setup_score",
            nbins=20,
            labels={"setup_score": "Setup Score"},
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Count")
        sections.append(
            '<section class="panel" id="setup-score-distribution">'
            "<h2>Setup Score Distribution</h2>"
            + figure_html(
                fig, "screen-setup-score-distribution", height=_CHART_HEIGHT_PX
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
            # Plotly draws the first row at the bottom; reverse so the
            # largest change sits at the top of the horizontal bars.
            top = top.iloc[::-1]
            fig = px.bar(
                top,
                x="change",
                y="name",
                orientation="h",
                labels={"change": "Change %", "name": "Ticker"},
            )
            fig.update_layout(showlegend=False)
            fig.update_yaxes(title_text="", automargin=True)
            sections.append(
                '<section class="panel" id="top-change">'
                "<h2>Top Change</h2>"
                + figure_html(
                    fig, "screen-top-change", height=_top_change_height(len(top))
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
    plotly_js = get_plotlyjs()
    title = f"{criteria_name.upper()} Screen"
    _css = """\
    :root {
      --ink: #e5e7eb;
      --muted: #9ca3af;
      --paper: #07090d;
      --panel: #0d1117;
      --panel-strong: #111827;
      --line: #242b36;
      --accent: #22c55e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "IBM Plex Sans", Aptos, sans-serif;
      letter-spacing: 0;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 24px 32px 18px;
      background: #0b0f16;
    }
    h1, h2 { margin: 0; font-weight: 700; }
    h1 { font-size: 28px; }
    h2 { font-size: 17px; margin-bottom: 14px; }
    main {
      padding: 22px 32px 36px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }
    .metrics {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    .metric, .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
    }
    .metric { padding: 13px 14px; }
    .metric span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      font-size: 22px;
      overflow-wrap: break-word;
    }
    .panel { padding: 16px; min-width: 0; }
    .wide { grid-column: 1 / -1; }
    .table-wrap { overflow: auto; max-height: 560px; }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }
    .data-table th {
      position: sticky;
      top: 0;
      background: var(--panel-strong);
      color: var(--ink);
      text-align: left;
      z-index: 1;
    }
    .data-table th, .data-table td {
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chips span {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: var(--panel-strong);
      font-size: 12px;
    }
    .empty, .notes { color: var(--muted); font-size: 13px; }
    .notes { margin: 0; padding-left: 18px; }
    @media (max-width: 900px) {
      header, main { padding-left: 16px; padding-right: 16px; }
      main { grid-template-columns: 1fr; }
      .wide, .metrics { grid-column: auto; }
      .metric strong { font-size: 18px; }
    }"""
    page_html = html_page(
        html.escape(title),
        _css,
        f"""\
  <header>
    <h1>{html.escape(title)}</h1>
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
        head_extra=f"<script>{plotly_js}</script>",
    )
    output_path.write_text(page_html, encoding="utf-8")
    return output_path
