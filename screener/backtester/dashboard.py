"""Static dashboard rendering for rolling backtest results."""

from __future__ import annotations

import html
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.io import to_html

from screener.backtester.display import trades_dataframe
from screener.backtester.models import BacktestResult


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _pct(value: Any) -> str:
    if not isinstance(value, (float, int)):
        return str(value)
    return f"{float(value) * 100:+.2f}%"


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.3f}"
    return str(value)


def _normalise_curve(curve: pd.Series, name: str) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame(columns=["date", name])
    frame = curve.rename(name).reset_index()
    frame.columns = ["date", name]
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def dashboard_frames(result: BacktestResult) -> dict[str, pd.DataFrame]:
    """Return the data frames used by the dashboard charts and tables."""
    equity = _normalise_curve(result.equity_curve, "equity")
    benchmark = _normalise_curve(result.benchmark_curve, "benchmark")
    curves = pd.merge(equity, benchmark, on="date", how="outer").sort_values("date")
    if not curves.empty:
        first_equity = curves["equity"].dropna().iloc[0]
        first_benchmark = curves["benchmark"].dropna().iloc[0]
        curves["strategy_return"] = curves["equity"] / first_equity - 1.0
        curves["benchmark_return"] = curves["benchmark"] / first_benchmark - 1.0
        curves["drawdown"] = curves["equity"] / curves["equity"].cummax() - 1.0

    trades = trades_dataframe(result)
    if not trades.empty:
        for col in ["signal_date", "entry_date", "exit_date"]:
            trades[col] = pd.to_datetime(trades[col])
        trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days

    monthly = pd.DataFrame(columns=["month", "return_pct"])
    if not equity.empty:
        monthly_equity = equity.set_index("date")["equity"].resample("ME").last()
        monthly = (
            monthly_equity.pct_change().dropna().rename("return_pct").reset_index()
        )
        monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
        monthly = monthly[["month", "return_pct"]]

    selection = result.selection.copy()
    if not selection.empty and "signal_date" in selection:
        selection["signal_date"] = pd.to_datetime(selection["signal_date"])

    return {
        "curves": curves,
        "trades": trades,
        "monthly": monthly,
        "selection": selection,
    }


def _figure_html(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#f7f5ef",
        plot_bgcolor="#fbfaf6",
        font={"family": "IBM Plex Sans, Aptos, sans-serif", "size": 12},
        margin={"l": 56, "r": 28, "t": 44, "b": 42},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _empty_panel(panel_id: str, title: str, message: str) -> str:
    return (
        f'<section class="panel" id="{panel_id}">'
        f'<h2>{html.escape(title)}</h2><p class="empty">{html.escape(message)}</p></section>'
    )


def _table_html(df: pd.DataFrame, table_id: str, limit: int = 250) -> str:
    if df.empty:
        return '<p class="empty">No rows.</p>'
    table = df.head(limit).copy()
    for col in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[col]):
            table[col] = table[col].dt.date.astype(str)
    return table.to_html(
        index=False,
        table_id=table_id,
        classes="data-table",
        border=0,
        justify="left",
    )


def _metric_cards(result: BacktestResult) -> str:
    pct_keys = {
        "total_return",
        "invested_return",
        "cagr",
        "vol_annual",
        "max_drawdown",
        "hit_rate",
        "alpha_annual",
        "exposure",
        "benchmark_return",
    }
    labels = {
        "total_return": "Total Return",
        "benchmark_return": "Benchmark",
        "max_drawdown": "Max DD",
        "sharpe": "Sharpe",
        "trade_count": "Trades",
        "unique_tickers": "Tickers",
        "exposure": "Exposure",
        "hit_rate": "Hit Rate",
    }
    cards: list[str] = []
    for key in [
        "total_return",
        "benchmark_return",
        "max_drawdown",
        "sharpe",
        "trade_count",
        "unique_tickers",
        "exposure",
        "hit_rate",
    ]:
        if key not in result.metrics:
            continue
        value = result.metrics[key]
        formatted = _pct(value) if key in pct_keys else _num(value)
        cards.append(
            f'<article class="metric"><span>{labels[key]}</span><strong>{formatted}</strong></article>'
        )
    return "".join(cards)


def render_dashboard(result: BacktestResult, output_dir: str | Path) -> Path:
    """Render a self-contained HTML dashboard and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    strategy = result.config.strategy_name or "expression"
    html_path = output_path / f"rolling-{result.config.market}-{strategy}-{stamp}.html"

    frames = dashboard_frames(result)
    curves = frames["curves"]
    trades = frames["trades"]
    monthly = frames["monthly"]
    selection = frames["selection"]

    sections: list[str] = []
    if curves.empty:
        sections.append(
            _empty_panel("performance-chart", "Performance", "No equity curve data.")
        )
        sections.append(_empty_panel("drawdown-chart", "Drawdown", "No drawdown data."))
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
            '<section class="panel" id="performance-chart"><h2>Performance</h2>'
            + _figure_html(perf, "equity-vs-benchmark")
            + "</section>"
        )

        dd = px.area(curves, x="date", y="drawdown", labels={"drawdown": "Drawdown"})
        dd.update_traces(line_color="#b91c1c", fillcolor="rgba(185,28,28,.18)")
        dd.update_yaxes(tickformat=".0%")
        sections.append(
            '<section class="panel" id="drawdown-chart"><h2>Drawdown</h2>'
            + _figure_html(dd, "drawdown-curve")
            + "</section>"
        )

    if monthly.empty:
        sections.append(
            _empty_panel("monthly-returns", "Monthly Returns", "No monthly returns.")
        )
    else:
        monthly_fig = px.bar(
            monthly,
            x="month",
            y="return_pct",
            labels={"return_pct": "Return", "month": "Month"},
            color="return_pct",
            color_continuous_scale=["#b91c1c", "#f7f5ef", "#0f766e"],
        )
        monthly_fig.update_yaxes(tickformat=".0%")
        sections.append(
            '<section class="panel" id="monthly-returns"><h2>Monthly Returns</h2>'
            + _figure_html(monthly_fig, "monthly-return-bars")
            + "</section>"
        )

    if trades.empty:
        sections.append(
            _empty_panel("trade-diagnostics", "Trade Diagnostics", "No trades.")
        )
    else:
        ret_fig = px.histogram(
            trades,
            x="return_pct",
            nbins=24,
            labels={"return_pct": "Trade Return"},
        )
        ret_fig.update_xaxes(tickformat=".0%")
        exit_fig = px.bar(
            trades.groupby("exit_reason").size().rename("count").reset_index(),
            x="exit_reason",
            y="count",
            labels={"exit_reason": "Exit", "count": "Trades"},
        )
        hold_fig = px.histogram(
            trades,
            x="holding_days",
            nbins=20,
            labels={"holding_days": "Holding Days"},
        )
        contrib = (
            trades.groupby("ticker")
            .agg(pnl=("pnl", "sum"), trades=("ticker", "size"))
            .reset_index()
            .sort_values("pnl", ascending=False)
            .head(30)
        )
        contrib_fig = px.bar(
            contrib,
            x="ticker",
            y="pnl",
            color="trades",
            labels={"ticker": "Ticker", "pnl": "PnL", "trades": "Trades"},
        )
        sections.append(
            '<section class="panel wide" id="trade-diagnostics"><h2>Trade Diagnostics</h2>'
            '<div class="chart-grid">'
            + _figure_html(ret_fig, "return-distribution")
            + _figure_html(exit_fig, "exit-reason-breakdown")
            + _figure_html(hold_fig, "holding-period-distribution")
            + _figure_html(contrib_fig, "ticker-contribution")
            + "</div></section>"
        )

    if selection.empty:
        sections.append(
            _empty_panel(
                "selection-diagnostics", "Selection Diagnostics", "No selected signals."
            )
        )
    else:
        by_day = selection.groupby("signal_date").size().rename("signals").reset_index()
        signal_fig = px.bar(
            by_day,
            x="signal_date",
            y="signals",
            labels={"signal_date": "Signal Date", "signals": "Signals"},
        )
        rank_fig = px.histogram(
            selection,
            x="rank",
            nbins=max(int(selection["rank"].max()), 1),
            labels={"rank": "Rank"},
        )
        sections.append(
            '<section class="panel wide" id="selection-diagnostics"><h2>Selection Diagnostics</h2>'
            '<div class="chart-grid two">'
            + _figure_html(signal_fig, "signal-count-by-day")
            + _figure_html(rank_fig, "rank-distribution")
            + "</div></section>"
        )

    warnings = (
        "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
        or "<li>No warnings.</li>"
    )
    cfg = result.config
    plotly_js = get_plotlyjs()
    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rolling Backtest Dashboard</title>
  <script>{plotly_js}</script>
  <style>
    :root {{
      --ink: #1e2320;
      --muted: #69716b;
      --paper: #f7f5ef;
      --panel: #fffefa;
      --line: #d9d4c7;
      --accent: #0f766e;
      --warn: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "IBM Plex Sans", Aptos, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 24px 32px 18px;
      background: #ebe7dc;
    }}
    h1, h2 {{ margin: 0; font-weight: 700; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 17px; margin-bottom: 14px; }}
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
    .chart-grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .tables {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .table-wrap {{ overflow: auto; max-height: 520px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      background: #ebe7dc;
      color: var(--ink);
      text-align: left;
      z-index: 1;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
    }}
    .empty, .warnings {{ color: var(--muted); font-size: 13px; }}
    .warnings {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      main, .tables, .chart-grid, .chart-grid.two {{ grid-template-columns: 1fr; }}
      .wide, .metrics {{ grid-column: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Rolling Backtest Dashboard</h1>
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
    <section class="metrics" id="summary-metrics">{_metric_cards(result)}</section>
    {"".join(sections)}
    <section class="panel wide" id="warnings"><h2>Warnings</h2><ul class="warnings">{warnings}</ul></section>
    <section class="tables">
      <article class="panel" id="trade-ledger"><h2>Trade Ledger</h2><div class="table-wrap">{_table_html(trades, "trade-ledger-table")}</div></article>
      <article class="panel" id="selection-table"><h2>Selections</h2><div class="table-wrap">{_table_html(selection, "selection-table-data")}</div></article>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(page_html, encoding="utf-8")
    return html_path


def serve_dashboard(path: str | Path, port: int) -> None:
    """Serve the dashboard directory until interrupted."""
    dashboard_path = Path(path).resolve()
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(dashboard_path), **kwargs
    )
    with _ReusableThreadingHTTPServer(("127.0.0.1", int(port)), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
