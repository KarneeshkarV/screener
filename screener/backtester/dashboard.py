"""Static dashboard rendering for rolling backtest results."""

from __future__ import annotations

import html
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from screener.backtester.display import trades_dataframe
from screener.backtester.metrics import result_view
from screener.backtester.models import BacktestResult
from screener.html_report import html_page
from screener.report_theme import (
    bar_chart,
    chart_html,
    drawdown_chart,
    histogram_chart,
    line_chart,
    signed_colors,
)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


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
        # Guard against an all-NaN column or a zero/NaN first value: dividing
        # by it would emit inf/NaN curves (or raise IndexError on dropna()).
        equity_valid = curves["equity"].dropna()
        benchmark_valid = curves["benchmark"].dropna()
        first_equity = float(equity_valid.iloc[0]) if not equity_valid.empty else 0.0
        first_benchmark = (
            float(benchmark_valid.iloc[0]) if not benchmark_valid.empty else 0.0
        )
        if first_equity:
            curves["strategy_return"] = curves["equity"] / first_equity - 1.0
            curves["drawdown"] = curves["equity"] / curves["equity"].cummax() - 1.0
        else:
            curves["strategy_return"] = float("nan")
            curves["drawdown"] = float("nan")
        if first_benchmark:
            curves["benchmark_return"] = curves["benchmark"] / first_benchmark - 1.0
        else:
            curves["benchmark_return"] = float("nan")

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


def _empty_panel(panel_id: str, title: str, message: str) -> str:
    return (
        f'<section class="panel" id="{panel_id}">'
        f'<h2>{html.escape(title)}</h2><p class="empty">{html.escape(message)}</p></section>'
    )


def _chart_card(title: str, chart: str) -> str:
    return f"<div><h3>{html.escape(title)}</h3>{chart}</div>"


def _count_axis(option: dict[str, Any]) -> dict[str, Any]:
    """Whole-number value axis so small counts never get repeated fractional ticks."""
    option["yAxis"]["minInterval"] = 1
    return option


def table_html(df: pd.DataFrame, table_id: str, limit: int = 250) -> str:
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


def metric_cards(result: BacktestResult) -> str:
    """Render every shared result-view row as an HTML metric card."""
    return "".join(
        f'<article class="metric"><span>{html.escape(row.label)}</span>'
        f"<strong>{html.escape(row.formatted)}</strong></article>"
        for row in result_view(result.metrics)
    )


_CSS = """\
main {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--gap);
}
.metrics, .wide, .tables { grid-column: 1 / -1; }
.tables {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--gap);
}
@media (max-width: 900px) {
  main, .tables { grid-template-columns: 1fr; }
}"""


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
        dates = curves["date"].tolist()
        perf_option = line_chart(
            dates,
            [
                {
                    "name": "Strategy",
                    "values": curves["strategy_return"].tolist(),
                    "color": "series-1",
                    "width": 3,
                },
                {
                    "name": "Benchmark",
                    "values": curves["benchmark_return"].tolist(),
                    "color": "series-2",
                    "width": 2,
                },
            ],
            value_format="pct0",
        )
        sections.append(
            '<section class="panel" id="performance-chart"><h2>Performance</h2>'
            + chart_html("equity-vs-benchmark", perf_option, group="perf")
            + "</section>"
        )

        dd_option = drawdown_chart(curves["date"].tolist(), curves["drawdown"].tolist())
        sections.append(
            '<section class="panel" id="drawdown-chart"><h2>Drawdown</h2>'
            + chart_html("drawdown-curve", dd_option, group="perf")
            + "</section>"
        )

    if monthly.empty:
        sections.append(
            _empty_panel("monthly-returns", "Monthly Returns", "No monthly returns.")
        )
    else:
        monthly_option = bar_chart(
            monthly["month"].tolist(),
            monthly["return_pct"].tolist(),
            name="Return",
            value_format="pct1",
            colors=signed_colors(monthly["return_pct"]),
        )
        sections.append(
            '<section class="panel wide" id="monthly-returns"><h2>Monthly Returns</h2>'
            + chart_html("monthly-return-bars", monthly_option)
            + "</section>"
        )

    if trades.empty:
        sections.append(
            _empty_panel("trade-diagnostics", "Trade Diagnostics", "No trades.")
        )
    else:
        ret_option = histogram_chart(
            trades["return_pct"].tolist(),
            bins=24,
            name="Trades",
            value_format="pct1",
            signed=True,
        )
        exit_counts = trades.groupby("exit_reason").size()
        exit_option = _count_axis(
            bar_chart(
                exit_counts.index.tolist(),
                exit_counts.tolist(),
                name="Trades",
                value_format="num0",
                labels=True,
            )
        )
        hold_option = histogram_chart(
            trades["holding_days"].tolist(),
            bins=20,
            name="Trades",
            value_format="num1",
            signed=False,
        )
        contrib = (
            trades.groupby("ticker")
            .agg(pnl=("pnl", "sum"), trades=("ticker", "size"))
            .reset_index()
            .sort_values("pnl", ascending=False)
            .head(30)
        )
        contrib_option = bar_chart(
            contrib["ticker"].tolist(),
            contrib["pnl"].tolist(),
            name="PnL",
            value_format="num0",
            colors=signed_colors(contrib["pnl"]),
        )
        sections.append(
            '<section class="panel wide" id="trade-diagnostics"><h2>Trade Diagnostics</h2>'
            '<div class="chart-grid">'
            + _chart_card(
                "Trade return distribution",
                chart_html("return-distribution", ret_option),
            )
            + _chart_card(
                "Exit reasons",
                chart_html("exit-reason-breakdown", exit_option),
            )
            + _chart_card(
                "Holding period (days)",
                chart_html("holding-period-distribution", hold_option),
            )
            + _chart_card(
                "PnL by ticker (top 30)",
                chart_html("ticker-contribution", contrib_option),
            )
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
        signal_option = _count_axis(
            bar_chart(
                by_day["signal_date"].tolist(),
                by_day["signals"].tolist(),
                name="Signals",
                value_format="num0",
            )
        )
        rank_option = histogram_chart(
            selection["rank"].tolist(),
            bins=max(int(selection["rank"].max()), 1),
            name="Selections",
            value_format="num1",
            signed=False,
        )
        sections.append(
            '<section class="panel wide" id="selection-diagnostics"><h2>Selection Diagnostics</h2>'
            '<div class="chart-grid">'
            + _chart_card(
                "Signals per day",
                chart_html("signal-count-by-day", signal_option),
            )
            + _chart_card(
                "Rank distribution",
                chart_html("rank-distribution", rank_option),
            )
            + "</div></section>"
        )

    warnings = (
        "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
        or "<li>No warnings.</li>"
    )
    cfg = result.config
    page_html = html_page(
        "Rolling Backtest Dashboard",
        _CSS,
        f"""\
  <header>
    <div class="eyebrow">Rolling backtest</div>
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
    <section class="metrics" id="summary-metrics">{metric_cards(result)}</section>
    {"".join(sections)}
    <section class="panel wide" id="warnings"><h2>Warnings</h2><ul class="warnings">{warnings}</ul></section>
    <section class="tables">
      <article class="panel" id="trade-ledger"><h2>Trade Ledger</h2><div class="table-wrap">{table_html(trades, "trade-ledger-table")}</div></article>
      <article class="panel" id="selection-table"><h2>Selections</h2><div class="table-wrap">{table_html(selection, "selection-table-data")}</div></article>
    </section>
  </main>""",
        charts=True,
    )
    html_path.write_text(page_html, encoding="utf-8")
    return html_path


def serve_dashboard(path: str | Path, port: int) -> None:
    """Serve the dashboard directory until interrupted."""
    dashboard_path = Path(path).resolve()

    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(*args, directory=str(dashboard_path), **kwargs)

    with _ReusableThreadingHTTPServer(("127.0.0.1", int(port)), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
