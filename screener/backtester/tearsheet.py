"""Static, self-contained HTML tear-sheet rendering for backtest results."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from screener import _optional
from screener.backtester.dashboard import (
    dashboard_frames,
    figure_html,
    metric_cards,
    table_html,
)
from screener.backtester.metrics import (
    SIZING_COMPARISON_COLUMNS,
    format_result_value,
    sizing_comparison_rows,
)
from screener.backtester.models import BacktestResult
from screener.html_report import html_page

if TYPE_CHECKING:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    from screener.backtester.optimization.monte_carlo import (
        EquityMonteCarloPaths,
        EquityMonteCarloResult,
    )
else:
    # Only the plotly names have a runtime counterpart; the Monte Carlo types
    # are annotations alone, so importing them here would pull the optimization
    # package into every tear-sheet render for nothing.
    px = _optional.load("plotly.express")
    go = _optional.load("plotly.graph_objects")
    get_plotlyjs = _optional.load("plotly.offline").get_plotlyjs

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
            frame["return_pct"] = frame["return_pct"].map(
                lambda value: format_result_value(value, "pct")
            )
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
            ledger[col] = ledger[col].map(
                lambda value: format_result_value(value, "pct")
            )
    for col in ["entry_price", "exit_price", "shares", "pnl"]:
        if col in ledger.columns:
            ledger[col] = ledger[col].map(lambda v: f"{float(v):,.2f}")
    return ledger


def _trade_timeline_html(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '<p class="empty">No trades.</p>'
    frame = trades.copy().sort_values(["entry_date", "exit_date", "ticker"])
    frame["label"] = frame["ticker"].astype(str) + " #" + frame["rank"].astype(str)
    frame["return_label"] = frame["return_pct"].map(
        lambda value: format_result_value(value, "pct")
    )
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
    return figure_html(fig, "tearsheet-trade-timeline")


def _sizing_comparison_html(
    fixed: BacktestResult,
    reinvested: BacktestResult,
) -> str:
    """Render the fixed-slot vs reinvested-slot metric table."""
    header = "".join(
        f"<th>{html.escape(name)}</th>" for name in SIZING_COMPARISON_COLUMNS
    )
    rows = "".join(
        "<tr><th>{}</th>{}</tr>".format(
            html.escape(row[0]),
            "".join(f"<td>{html.escape(cell)}</td>" for cell in row[1:]),
        )
        for row in sizing_comparison_rows(fixed.metrics, reinvested.metrics)
    )
    return (
        '<section class="panel" id="sizing-comparison">'
        "<h2>Fixed slots vs reinvested slots</h2>"
        '<div class="table-wrap">'
        '<table class="data-table" id="sizing-comparison-table">'
        f"<thead><tr><th>Metric</th>{header}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        '<p class="empty">Fixed slots spend a constant '
        "initial_capital / top per entry, so profits sit as idle cash. Reinvested "
        "slots size each entry from current marked-to-market equity, so the run "
        "compounds.</p></section>"
    )


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


# How many simulated paths the fan chart draws. The bootstrap keeps more than
# this so the percentile bands stay smooth, but a browser cannot render (or a
# reader read) a thousand overlapping lines, and each one costs file size.
_FAN_LINES = 200
# How many x positions each drawn path gets. The fan is a texture rather than a
# readout, so its resolution is bounded by the pixels the chart has, not by the
# bar count: at 2,520 bars every drawn bar costs 4.4 MB of HTML against 0.7 MB
# here. The bands and the realized run stay at full resolution, because those
# are the traces a reader takes numbers off.
_FAN_BAR_POINTS = 400
# Equity levels are rounded before they reach the HTML because the full float
# repr of 200 paths dominates the file size. Six significant figures is one
# part in a million of the starting capital, which no chart pixel resolves.
_FAN_SIGNIFICANT_DIGITS = 6


# Label and colour per percentile, looked up by the band's own percentile so a
# change to ``band_percentiles`` cannot silently mislabel a line.
def _fan_decimals(initial_capital: float) -> int:
    """Decimal places that keep the fan legible at any starting capital.

    Rounding to whole currency units assumes the run started somewhere near
    100,000. ``--initial-capital`` has no floor, so at 100 that quantizes the
    fan lines, the three bands and the realized run onto the same handful of
    integers: the chart shows one staircase where the summary table reports a
    p05/p95 spread. Scaling the precision to the capital keeps the same six
    significant figures, and the same HTML size, at every scale.
    """
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        return 0
    magnitude = int(math.floor(math.log10(initial_capital)))
    return max(0, _FAN_SIGNIFICANT_DIGITS - 1 - magnitude)


_FAN_BAND_STYLE = {
    5: ("p05", "#b91c1c"),
    50: ("median", "#38bdf8"),
    95: ("p95", "#16a34a"),
}


def _fan_chart_html(
    equity: pd.Series,
    paths: "EquityMonteCarloPaths",
) -> str:
    """Draw the simulated equity paths, their percentile bands, and the real run.

    The x axis is the bar number, not the date. A resampled path draws its
    returns out of calendar order, so no bar of it belongs to any date; plotting
    against the real index would also repeat the date array once per line, which
    on its own added ~4 MB to the page.

    The bands come from ``paths.bands``, which the bootstrap took over every
    iteration. Recomputing them here from the retained sample would answer a
    different question from the summary table on the same page, and at
    ``--paths 1`` would collapse all three onto one arbitrary path.
    """
    # Empty bands mean the curve had no bar to resample. Retaining no paths is
    # a different case, and it still has bands to draw.
    if paths.bands.size == 0:
        return '<p class="empty">No bars to simulate.</p>'

    fig = go.Figure()
    # Every series on this chart is rounded to the same precision, so the fan,
    # the bands and the realized run stay comparable at any starting capital.
    decimals = _fan_decimals(paths.initial_capital)
    sample = paths.paths
    # ``--paths 0`` retains nothing, and that is a request to skip the fan, not
    # to drop the bands with it.
    if sample.size:
        # Every synthetic path starts at the same capital as the real run, so
        # prepending it puts bar 0 of a path on bar 0 of the realized curve.
        start = np.full((sample.shape[0], 1), paths.initial_capital, dtype=sample.dtype)
        curves = np.hstack([start, sample])
        drawn = curves[:: max(1, curves.shape[0] // _FAN_LINES)][:_FAN_LINES]
        bars = drawn.shape[1]
        points = min(_FAN_BAR_POINTS, bars)
        if points < bars and points > 1:
            # ``linspace`` keeps the first and last bar exactly, so the fan ends
            # on the same bar as the realized run. The spacing it samples at is
            # within one bar of the even ``dx`` grid below, which no reader can
            # see on a background texture.
            drawn = drawn[:, np.round(np.linspace(0, bars - 1, points)).astype(int)]
            dx = (bars - 1) / (points - 1)
        else:
            dx = 1.0
        for row, path in enumerate(np.round(drawn, decimals).astype(float)):
            fig.add_trace(
                go.Scatter(
                    y=path.tolist(),
                    # ``x0``/``dx`` spaces the points without emitting an x
                    # array. One flat trace with gaps between paths would need
                    # an explicit x repeated per path, which measured 1.6x
                    # larger than these per-path traces.
                    x0=0,
                    dx=dx,
                    name=f"{drawn.shape[0]} simulated paths",
                    legendgroup="paths",
                    showlegend=row == 0,
                    mode="lines",
                    line={"color": "rgba(148,163,184,.18)", "width": 1},
                    hoverinfo="skip",
                )
            )
    for pct, band in zip(paths.band_percentiles, paths.bands, strict=True):
        label, color = _FAN_BAND_STYLE.get(pct, (f"p{pct:02d}", "#94a3b8"))
        fig.add_trace(
            go.Scatter(
                y=np.round(band, decimals).tolist(),
                name=f"Simulated {label}",
                mode="lines",
                line={"color": color, "width": 2, "dash": "dot"},
            )
        )
    fig.add_trace(
        go.Scatter(
            y=np.round(equity.to_numpy(), decimals).tolist(),
            name="Realized run",
            mode="lines",
            line={"color": "#facc15", "width": 3},
        )
    )
    fig.update_xaxes(title="Bar")
    fig.update_yaxes(title="Equity")
    return figure_html(fig, "tearsheet-mc-paths", height=420)


def _distribution_html(
    values: np.ndarray,
    div_id: str,
    *,
    realized: float,
    label: str,
) -> str:
    """Histogram of a per-iteration outcome, with the realized run marked."""
    fig = px.histogram(x=values, nbins=60, labels={"x": label})
    fig.update_traces(marker_color="#0f766e")
    # One vectorized call rather than three: each ``np.percentile`` sorts the
    # whole array, and these three want the same sort.
    p05, median, p95 = (float(v) for v in np.percentile(values, [5, 50, 95]))
    markers = [
        ("p05", p05, "#b91c1c"),
        ("median", median, "#38bdf8"),
        ("p95", p95, "#16a34a"),
        ("realized", realized, "#facc15"),
    ]
    # Put labels for the same value on separate rows. Labels for different
    # values can use the first row because their marker lines separate them.
    markers.sort(key=lambda marker: marker[1])
    label_rows_by_value: dict[float, int] = {}
    for name, value, color in markers:
        label_row = label_rows_by_value.get(value, 0)
        label_rows_by_value[value] = label_row + 1
        fig.add_vline(
            x=value,
            line_color=color,
            line_dash="dot",
            annotation={
                "text": name,
                "font": {"color": color, "size": 11},
                "yshift": label_row * 15,
            },
            annotation_position="top",
        )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    return figure_html(fig, div_id)


_OUTCOME_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _monte_carlo_percentile_rows(paths: "EquityMonteCarloPaths") -> str:
    # One call per array rather than one per cell: ``np.percentile`` sorts the
    # array it is given, and the loop sorted the same two 5,000-element arrays
    # seven times each to fill seven rows.
    returns = np.percentile(paths.terminal_returns, _OUTCOME_PERCENTILES)
    drawdowns = np.percentile(paths.drawdowns, _OUTCOME_PERCENTILES)
    return "".join(
        f"<tr><th>p{pct:02d}</th><td>{ret:+.2%}</td><td>{dd:+.2%}</td></tr>"
        for pct, ret, dd in zip(_OUTCOME_PERCENTILES, returns, drawdowns, strict=True)
    )


def _monte_carlo_sections(
    result: BacktestResult,
    mc: "EquityMonteCarloResult",
    paths: "EquityMonteCarloPaths",
) -> str:
    """Render the Monte Carlo tab: the fan of paths, the outcome distributions."""
    if paths.terminal_returns.size == 0:
        return _empty_section(
            "monte-carlo-paths", "Simulated Equity Paths", "No equity curve data."
        )
    equity = result.equity_curve
    # Read rather than recompute: ``metrics`` owns this definition, and the
    # Overview tab's Total Return row prints the same key. Two formulas here
    # would let the two rows drift apart.
    realized_return = float(result.metrics.get("total_return", 0.0))
    realized_drawdown = float(result.metrics.get("max_drawdown", 0.0))
    strided = (
        ""
        if paths.band_iterations >= mc.iterations
        else (
            f" The percentile bands are taken over {paths.band_iterations:,} of "
            f"them, a uniform stride the memory budget forced at this size."
        )
    )
    summary = (
        f'<p class="empty">{mc.iterations:,} circular block-bootstrap paths of '
        f"{mc.bars:,} bars, block {mc.block}, seed {mc.seed}. "
        f"Each path resamples the realized daily equity returns, so it keeps the "
        f"run's own concurrency, sizing and costs; only the order of the returns "
        f"changes.{strided}</p>"
    )
    return (
        '<section class="panel wide" id="monte-carlo-paths">'
        "<h2>Simulated Equity Paths</h2>"
        + summary
        + _fan_chart_html(equity, paths)
        + "</section>"
        '<section class="panel" id="monte-carlo-returns">'
        "<h2>Terminal Return Distribution</h2>"
        + _distribution_html(
            paths.terminal_returns,
            "tearsheet-mc-returns",
            realized=realized_return,
            label="Terminal Return",
        )
        + "</section>"
        '<section class="panel" id="monte-carlo-drawdowns">'
        "<h2>Max Drawdown Distribution</h2>"
        + _distribution_html(
            paths.drawdowns,
            "tearsheet-mc-drawdowns",
            realized=realized_drawdown,
            label="Max Drawdown",
        )
        + "</section>"
        '<section class="panel" id="monte-carlo-percentiles">'
        '<h2>Outcome Percentiles</h2><div class="table-wrap">'
        '<table class="data-table" id="monte-carlo-percentile-table">'
        "<tr><th>Percentile</th><th>Terminal Return</th><th>Max Drawdown</th></tr>"
        + _monte_carlo_percentile_rows(paths)
        + "</table></div></section>"
        '<section class="panel" id="monte-carlo-summary">'
        '<h2>Monte Carlo Summary</h2><div class="table-wrap">'
        '<table class="data-table" id="monte-carlo-summary-table">'
        f"<tr><th>Iterations</th><td>{mc.iterations:,}</td></tr>"
        f"<tr><th>Bars per path</th><td>{mc.bars:,}</td></tr>"
        f"<tr><th>Block (bars)</th><td>{mc.block}</td></tr>"
        f"<tr><th>Seed</th><td>{mc.seed}</td></tr>"
        f"<tr><th>Paths retained</th><td>{paths.paths.shape[0]:,}</td></tr>"
        f"<tr><th>Realized return</th><td>{realized_return:+.2%}</td></tr>"
        f"<tr><th>Realized max drawdown</th><td>{realized_drawdown:+.2%}</td></tr>"
        f"<tr><th>Probability of profit</th>"
        f"<td>{mc.probability_of_profit:+.2%}</td></tr>"
        f"<tr><th>Risk of ruin</th><td>{mc.risk_of_ruin:+.2%}</td></tr>"
        f"<tr><th>Ruin threshold</th>"
        f"<td>{mc.ruin_threshold:.2%} of starting capital</td></tr>"
        "</table></div></section>"
    )


def render_tearsheet(
    result: BacktestResult,
    output_file: str | Path,
    *,
    title: str = "Backtest Tear Sheet",
    extra_notes: Sequence[str] = (),
    sizing_comparison: tuple[BacktestResult, BacktestResult] | None = None,
    monte_carlo: tuple["EquityMonteCarloResult", "EquityMonteCarloPaths"] | None = None,
) -> Path:
    """Render a static, self-contained HTML tear-sheet and return its path."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = dashboard_frames(result)
    curves = frames["curves"]
    trades = frames["trades"]
    monthly = frames["monthly"]

    sections: list[str] = []
    if sizing_comparison is not None:
        sections.append(_sizing_comparison_html(*sizing_comparison))
    ledger_html = (
        '<p class="empty">No trades.</p>'
        if trades.empty
        else table_html(_trade_ledger_frame(trades), "trade-ledger-table", limit=5000)
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
            + figure_html(perf, "tearsheet-equity-vs-benchmark")
            + "</section>"
        )

        dd = px.area(curves, x="date", y="drawdown", labels={"drawdown": "Drawdown"})
        dd.update_traces(line_color="#b91c1c", fillcolor="rgba(185,28,28,.18)")
        dd.update_yaxes(tickformat=".0%")
        sections.append(
            '<section class="panel wide" id="drawdown-curve"><h2>Drawdown</h2>'
            + figure_html(dd, "tearsheet-drawdown-curve")
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
            + figure_html(hist, "tearsheet-trade-histogram")
            + "</section>"
        )
        winners, losers = _winners_losers_frames(trades)
        sections.append(
            '<section class="panel wide" id="winners-losers"><h2>Top Winners &amp; Losers</h2>'
            '<div class="chart-grid two">'
            '<div class="table-wrap"><h3>Top 10 Winners</h3>'
            + table_html(winners, "top-winners-table")
            + '</div><div class="table-wrap"><h3>Top 10 Losers</h3>'
            + table_html(losers, "top-losers-table")
            + "</div></div></section>"
        )

    notes = [*extra_notes, *result.warnings]
    warnings_html = (
        "".join(f"<li>{html.escape(note)}</li>" for note in notes)
        or "<li>No warnings.</li>"
    )
    cfg = result.config
    monte_carlo_html = (
        "" if monte_carlo is None else _monte_carlo_sections(result, *monte_carlo)
    )
    # The tab only exists on a Monte Carlo run. Its CSS rules stay in the sheet
    # unconditionally; a selector for an absent id is inert.
    monte_carlo_input = (
        ""
        if monte_carlo is None
        else '<input class="tab-radio" type="radio" name="report-tab" id="tab-montecarlo">'
    )
    monte_carlo_label = (
        "" if monte_carlo is None else '<label for="tab-montecarlo">Monte Carlo</label>'
    )
    monte_carlo_panel = (
        ""
        if monte_carlo is None
        else f'<section class="tab-panel" id="montecarlo-panel">{monte_carlo_html}</section>'
    )
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
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 24px 32px 18px;
      background: #0b0f16;
    }
    h1, h2, h3 { margin: 0; font-weight: 700; }
    h1 { font-size: 28px; }
    h2 { font-size: 17px; margin-bottom: 14px; }
    h3 { font-size: 14px; margin-bottom: 8px; }
    .subhead {
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 8px;
      font-size: 13px;
    }
    main {
      padding: 22px 32px 36px;
    }
    .tab-radio { position: absolute; opacity: 0; pointer-events: none; }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .tabs label {
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      cursor: pointer;
      padding: 8px 12px;
      background: var(--panel);
      font-size: 13px;
    }
    #tab-overview:checked ~ .tabs label[for="tab-overview"],
    #tab-montecarlo:checked ~ .tabs label[for="tab-montecarlo"],
    #tab-ledger:checked ~ .tabs label[for="tab-ledger"] {
      color: var(--ink);
      border-color: var(--accent);
      background: #10261c;
    }
    .tab-panel { display: none; }
    #tab-overview:checked ~ #overview-panel,
    #tab-montecarlo:checked ~ #montecarlo-panel,
    #tab-ledger:checked ~ #ledger-panel {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
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
    .metric strong { display: block; margin-top: 5px; font-size: 22px; }
    .panel { padding: 16px; min-width: 0; }
    .wide { grid-column: 1 / -1; }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .table-wrap { overflow: auto; max-height: 520px; }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }
    .data-table th, .data-table td {
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
      text-align: left;
    }
    .data-table th { background: var(--panel-strong); color: var(--ink); }
    .heatmap td { text-align: right; }
    .empty, .warnings { color: var(--muted); font-size: 13px; }
    .warnings { margin: 0; padding-left: 18px; }
    @media (max-width: 900px) {
      header, main { padding-left: 16px; padding-right: 16px; }
      #tab-overview:checked ~ #overview-panel,
      #tab-montecarlo:checked ~ #montecarlo-panel,
      #tab-ledger:checked ~ #ledger-panel,
      .chart-grid { grid-template-columns: 1fr; }
      .wide, .metrics { grid-column: auto; }
    }"""
    page_html = html_page(
        html.escape(title),
        _css,
        f"""\
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
    {monte_carlo_input}
    <nav class="tabs" aria-label="Backtest report tabs">
      <label for="tab-overview">Overview</label>
      {monte_carlo_label}
      <label for="tab-ledger">Trade Ledger</label>
    </nav>
    <section class="tab-panel" id="overview-panel">
      <section class="metrics" id="metrics-summary">{metric_cards(result)}</section>
      {"".join(sections)}
      <section class="panel" id="config"><h2>Config</h2><div class="table-wrap"><table class="data-table" id="config-table">{_config_rows(result)}</table></div></section>
      <section class="panel" id="warnings"><h2>Warnings</h2><ul class="warnings">{warnings_html}</ul></section>
    </section>
    <section class="tab-panel" id="ledger-panel">
      <section class="panel wide" id="trade-ledger"><h2>Trade Ledger</h2><div class="table-wrap ledger-wrap">{ledger_html}</div></section>
    </section>
    {monte_carlo_panel}
  </main>""",
        head_extra=f"<script>{get_plotlyjs()}</script>",
    )
    output_path.write_text(page_html, encoding="utf-8")
    return output_path
