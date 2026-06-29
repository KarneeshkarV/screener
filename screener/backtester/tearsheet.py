"""Static, self-contained HTML tear-sheet rendering for backtest results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from screener.backtester.dashboard import (
    _figure_html,
    _metric_cards,
    _pct,
    _table_html,
    dashboard_frames,
)
from screener.backtester.models import BacktestResult

_MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _empty_section(section_id: str, title: str, message: str) -> str:
    return (
        f'<section class="panel" id="{section_id}">'
        f'<h2>{html.escape(title)}</h2><p class="empty">{html.escape(message)}</p></section>'
    )


def _heatmap_cell(value: float) -> str:
    if pd.isna(value):
        return '<td class="hm-empty"></td>'
    alpha = min(abs(float(value)) / 0.10, 1.0) * 0.85
    color = "15,118,110" if value >= 0 else "185,28,28"
    return (
        f'<td style="background:rgba({color},{alpha:.2f})">'
        f"{float(value) * 100:+.1f}%</td>"
    )


def _monthly_heatmap_html(monthly: pd.DataFrame) -> str:
    """Render monthly returns as a year x month table with colored cells."""
    if monthly.empty:
        return '<p class="empty">No monthly returns.</p>'
    frame = monthly.copy()
    frame["year"] = frame["month"].str[:4]
    frame["mon"] = frame["month"].str[5:7].astype(int)
    pivot = frame.pivot(index="year", columns="mon", values="return_pct")
    header = "".join(f"<th>{label}</th>" for label in _MONTH_LABELS)
    rows: list[str] = []
    for year in sorted(pivot.index):
        cells = "".join(
            _heatmap_cell(
                cast(float, pivot.at[year, mon])
                if mon in pivot.columns
                else float("nan")
            )
            for mon in range(1, 13)
        )
        rows.append(f"<tr><th>{html.escape(str(year))}</th>{cells}</tr>")
    return (
        '<table class="data-table heatmap" id="monthly-heatmap-table">'
        f"<thead><tr><th>Year</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _winners_losers_frames(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        c
        for c in [
            "ticker",
            "entry_date",
            "exit_date",
            "exit_reason",
            "return_pct",
            "pnl",
        ]
        if c in trades.columns
    ]
    ranked = trades.sort_values("return_pct", ascending=False)[cols]
    winners = ranked.head(10).copy()
    losers = ranked.tail(10).iloc[::-1].copy()
    for frame in (winners, losers):
        if "return_pct" in frame.columns:
            frame["return_pct"] = frame["return_pct"].map(_pct)
        if "pnl" in frame.columns:
            frame["pnl"] = frame["pnl"].map(lambda v: f"{float(v):,.2f}")
    return winners, losers


def _trade_ledger_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Return display-ready trade ledger rows without dropping columns."""
    ledger = trades.copy()
    if ledger.empty:
        return ledger
    for col in ["return_pct"]:
        if col in ledger.columns:
            ledger[col] = ledger[col].map(_pct)
    for col in ["entry_price", "exit_price", "shares", "pnl"]:
        if col in ledger.columns:
            ledger[col] = ledger[col].map(lambda v: f"{float(v):,.2f}")
    return ledger


def _trade_timeline_html(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '<p class="empty">No trades.</p>'
    frame = trades.copy().sort_values(["entry_date", "exit_date", "ticker"])
    frame["label"] = frame["ticker"].astype(str) + " #" + frame["rank"].astype(str)
    frame["return_label"] = frame["return_pct"].map(_pct)
    frame["pnl_label"] = frame["pnl"].map(lambda v: f"{float(v):,.2f}")
    frame["holding_days"] = (
        pd.to_datetime(frame["exit_date"]) - pd.to_datetime(frame["entry_date"])
    ).dt.days
    fig = px.timeline(
        frame,
        x_start="entry_date",
        x_end="exit_date",
        y="label",
        color="return_pct",
        color_continuous_scale=["#ef4444", "#1f2937", "#22c55e"],
        hover_data={
            "ticker": True,
            "rank": True,
            "return_label": True,
            "pnl_label": True,
            "exit_reason": True,
            "holding_days": True,
            "return_pct": False,
            "label": False,
        },
        labels={"label": "Trade", "return_pct": "Return"},
    )
    fig.update_yaxes(autorange="reversed")
    return _figure_html(fig, "tearsheet-trade-timeline")


def _config_rows(result: BacktestResult) -> str:
    dump = result.config.model_dump(exclude={"slippage_model"})
    if dump.get("membership_added"):
        dump["membership_added"] = f"{len(dump['membership_added'])} dated symbols"
    tickers = dump.get("tickers")
    if tickers and len(tickers) > 20:
        dump["tickers"] = f"{len(tickers)} tickers"
    rows = []
    for key, value in dump.items():
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td></tr>"
        )
    return "".join(rows)


def render_tearsheet(
    result: BacktestResult,
    output_file: str | Path,
    *,
    title: str = "Backtest Tear Sheet",
    extra_notes: Sequence[str] = (),
) -> Path:
    """Render a static, self-contained HTML tear-sheet and return its path."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = dashboard_frames(result)
    curves = frames["curves"]
    trades = frames["trades"]
    monthly = frames["monthly"]

    sections: list[str] = []
    ledger_html = (
        '<p class="empty">No trades.</p>'
        if trades.empty
        else _table_html(_trade_ledger_frame(trades), "trade-ledger-table", limit=5000)
    )

    if curves.empty:
        sections.append(
            _empty_section(
                "equity-vs-benchmark", "Equity vs Benchmark", "No equity curve data."
            )
        )
        sections.append(
            _empty_section("drawdown-curve", "Drawdown", "No drawdown data.")
        )
    else:
        perf = go.Figure()
        perf.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves["strategy_return"],
                name="Strategy",
                mode="lines",
                line={"color": "#0f766e", "width": 3},
            )
        )
        perf.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves["benchmark_return"],
                name="Benchmark",
                mode="lines",
                line={"color": "#7c3aed", "width": 2},
            )
        )
        perf.update_yaxes(tickformat=".0%")
        sections.append(
            '<section class="panel wide" id="equity-vs-benchmark"><h2>Equity vs Benchmark</h2>'
            + _figure_html(perf, "tearsheet-equity-vs-benchmark")
            + "</section>"
        )

        dd = px.area(curves, x="date", y="drawdown", labels={"drawdown": "Drawdown"})
        dd.update_traces(line_color="#b91c1c", fillcolor="rgba(185,28,28,.18)")
        dd.update_yaxes(tickformat=".0%")
        sections.append(
            '<section class="panel wide" id="drawdown-curve"><h2>Drawdown</h2>'
            + _figure_html(dd, "tearsheet-drawdown-curve")
            + "</section>"
        )

    sections.append(
        '<section class="panel" id="monthly-heatmap"><h2>Monthly Returns</h2>'
        '<div class="table-wrap">' + _monthly_heatmap_html(monthly) + "</div></section>"
    )

    if trades.empty:
        sections.append(
            _empty_section("trade-timeline", "Trade Timeline", "No trades.")
        )
        sections.append(
            _empty_section("trade-histogram", "Trade Return Distribution", "No trades.")
        )
        sections.append(
            _empty_section("winners-losers", "Top Winners & Losers", "No trades.")
        )
    else:
        sections.append(
            '<section class="panel wide" id="trade-timeline"><h2>Trade Timeline</h2>'
            + _trade_timeline_html(trades)
            + "</section>"
        )
        hist = px.histogram(
            trades,
            x="return_pct",
            nbins=24,
            labels={"return_pct": "Trade Return"},
        )
        hist.update_xaxes(tickformat=".0%")
        sections.append(
            '<section class="panel" id="trade-histogram"><h2>Trade Return Distribution</h2>'
            + _figure_html(hist, "tearsheet-trade-histogram")
            + "</section>"
        )
        winners, losers = _winners_losers_frames(trades)
        sections.append(
            '<section class="panel wide" id="winners-losers"><h2>Top Winners &amp; Losers</h2>'
            '<div class="chart-grid two">'
            '<div class="table-wrap"><h3>Top 10 Winners</h3>'
            + _table_html(winners, "top-winners-table")
            + '</div><div class="table-wrap"><h3>Top 10 Losers</h3>'
            + _table_html(losers, "top-losers-table")
            + "</div></div></section>"
        )

    notes = [*extra_notes, *result.warnings]
    warnings_html = (
        "".join(f"<li>{html.escape(note)}</li>" for note in notes)
        or "<li>No warnings.</li>"
    )
    cfg = result.config
    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script>{get_plotlyjs()}</script>
  <style>
    :root {{
      --ink: #e5e7eb;
      --muted: #9ca3af;
      --paper: #07090d;
      --panel: #0d1117;
      --panel-strong: #111827;
      --line: #242b36;
      --accent: #22c55e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "IBM Plex Sans", Aptos, sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 24px 32px 18px;
      background: #0b0f16;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 700; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 17px; margin-bottom: 14px; }}
    h3 {{ font-size: 14px; margin-bottom: 8px; }}
    .subhead {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 8px;
      font-size: 13px;
    }}
    main {{
      padding: 22px 32px 36px;
    }}
    .tab-radio {{ position: absolute; opacity: 0; pointer-events: none; }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .tabs label {{
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      cursor: pointer;
      padding: 8px 12px;
      background: var(--panel);
      font-size: 13px;
    }}
    #tab-overview:checked ~ .tabs label[for="tab-overview"],
    #tab-ledger:checked ~ .tabs label[for="tab-ledger"] {{
      color: var(--ink);
      border-color: var(--accent);
      background: #10261c;
    }}
    .tab-panel {{ display: none; }}
    #tab-overview:checked ~ #overview-panel,
    #tab-ledger:checked ~ #ledger-panel {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .metrics {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .metric, .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
    }}
    .metric {{ padding: 13px 14px; }}
    .metric span {{
      color: var(--muted);
      display: block;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 22px; }}
    .panel {{ padding: 16px; min-width: 0; }}
    .wide {{ grid-column: 1 / -1; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .table-wrap {{ overflow: auto; max-height: 520px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
      text-align: left;
    }}
    .data-table th {{ background: var(--panel-strong); color: var(--ink); }}
    .heatmap td {{ text-align: right; }}
    .empty, .warnings {{ color: var(--muted); font-size: 13px; }}
    .warnings {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      #tab-overview:checked ~ #overview-panel,
      #tab-ledger:checked ~ #ledger-panel,
      .chart-grid {{ grid-template-columns: 1fr; }}
      .wide, .metrics {{ grid-column: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="subhead">
      <span>{html.escape(cfg.market.upper())}</span>
      <span>{html.escape(cfg.strategy_name or "custom expression")}</span>
      <span>as-of {html.escape(str(cfg.as_of))}</span>
      <span>hold {cfg.hold}</span>
      <span>top {cfg.top}</span>
      <span>benchmark {html.escape(cfg.benchmark)}</span>
    </div>
  </header>
  <main>
    <input class="tab-radio" type="radio" name="report-tab" id="tab-overview" checked>
    <input class="tab-radio" type="radio" name="report-tab" id="tab-ledger">
    <nav class="tabs" aria-label="Backtest report tabs">
      <label for="tab-overview">Overview</label>
      <label for="tab-ledger">Trade Ledger</label>
    </nav>
    <section class="tab-panel" id="overview-panel">
      <section class="metrics" id="metrics-summary">{_metric_cards(result)}</section>
      {"".join(sections)}
      <section class="panel" id="config"><h2>Config</h2><div class="table-wrap"><table class="data-table" id="config-table">{_config_rows(result)}</table></div></section>
      <section class="panel" id="warnings"><h2>Warnings</h2><ul class="warnings">{warnings_html}</ul></section>
    </section>
    <section class="tab-panel" id="ledger-panel">
      <section class="panel wide" id="trade-ledger"><h2>Trade Ledger</h2><div class="table-wrap ledger-wrap">{ledger_html}</div></section>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(page_html, encoding="utf-8")
    return output_path
