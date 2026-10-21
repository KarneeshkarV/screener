"""Static, self-contained HTML tear-sheet rendering for backtest results."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from screener.backtester.dashboard import (
    dashboard_frames,
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
from screener.report_theme import (
    chart_html,
    diverging_cell_style,
    drawdown_chart,
    histogram_chart,
    line_chart,
    signed_colors,
)

if TYPE_CHECKING:
    from screener.backtester.optimization.monte_carlo import (
        EquityMonteCarloPaths,
        EquityMonteCarloResult,
    )

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


def _heatmap_cell(value: float, max_abs: float) -> str:
    if pd.isna(value):
        return '<td class="hm-empty"></td>'
    style, cls = diverging_cell_style(value, max_abs)
    class_attr = f' class="{cls}"' if cls else ""
    return f'<td{class_attr} style="{style}">{float(value) * 100:+.1f}%</td>'


def _monthly_heatmap_html(monthly: pd.DataFrame) -> str:
    """Render monthly returns as a year x month table with diverging cells."""
    if monthly.empty:
        return '<p class="empty">No monthly returns.</p>'
    frame = monthly.copy()
    frame["year"] = frame["month"].str[:4]
    frame["mon"] = frame["month"].str[5:7].astype(int)
    pivot = frame.pivot(index="year", columns="mon", values="return_pct")
    finite = np.abs(pivot.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    max_abs = float(finite.max()) if finite.size else 0.0
    header = "".join(f"<th>{label}</th>" for label in _MONTH_LABELS)
    rows: list[str] = []
    for year in sorted(pivot.index):
        cells = "".join(
            _heatmap_cell(
                cast(float, pivot.at[year, mon])
                if mon in pivot.columns
                else float("nan"),
                max_abs,
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
    """Horizontal entry-to-exit bar per trade, coloured by the trade's return.

    A transparent offset bar stacked under each visible bar puts it at the entry
    date on a time axis, so the bar spans the holding period.
    """
    if trades.empty:
        return '<p class="empty">No trades.</p>'
    frame = trades.copy().sort_values(["entry_date", "exit_date", "ticker"])
    returns = frame["return_pct"].tolist()
    # The category label carries the return, because an ECharts string tooltip
    # can only show the category name (no JS formatter is allowed in a spec).
    labels = [
        f"{ticker} #{rank}  {format_result_value(ret, 'pct')}"
        for ticker, rank, ret in zip(
            frame["ticker"].astype(str), frame["rank"].astype(str), returns, strict=True
        )
    ]
    entry_ms = [
        int(pd.Timestamp(v).timestamp() * 1000)
        for v in pd.to_datetime(frame["entry_date"])
    ]
    exit_ms = [
        int(pd.Timestamp(v).timestamp() * 1000)
        for v in pd.to_datetime(frame["exit_date"])
    ]
    colors = signed_colors(returns)
    bars = [
        {
            "value": exit_ms[i] - entry_ms[i],
            "itemStyle": {"color": colors[i], "borderRadius": 3},
        }
        for i in range(len(labels))
    ]
    option: dict[str, Any] = {
        "tooltip": {"trigger": "item", "formatter": "{b}"},
        "xAxis": {
            "type": "value",
            "min": min(entry_ms),
            "axisLabel": {"formatter": "fmt:date"},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
            "splitLine": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "name": "Entry",
                "stack": "span",
                "silent": True,
                "barMaxWidth": 14,
                "itemStyle": {"color": "transparent"},
                "data": entry_ms,
            },
            {
                "type": "bar",
                "name": "Trade",
                "stack": "span",
                "barMaxWidth": 14,
                "data": bars,
            },
        ],
    }
    height = min(640, max(240, 28 * len(labels) + 60))
    return chart_html("tearsheet-trade-timeline", option, height=height)


def _sizing_comparison_html(
    equal_slot: BacktestResult,
    reinvested: BacktestResult,
) -> str:
    """Render the equal-slot vs reinvested-slot metric table."""
    header = "".join(
        f"<th>{html.escape(name)}</th>" for name in SIZING_COMPARISON_COLUMNS
    )
    rows = "".join(
        "<tr><th>{}</th>{}</tr>".format(
            html.escape(row[0]),
            "".join(f"<td>{html.escape(cell)}</td>" for cell in row[1:]),
        )
        for row in sizing_comparison_rows(equal_slot.metrics, reinvested.metrics)
    )
    return (
        '<section class="panel" id="sizing-comparison">'
        "<h2>Equal slots vs reinvested slots</h2>"
        '<div class="table-wrap">'
        '<table class="data-table" id="sizing-comparison-table">'
        f"<thead><tr><th>Metric</th>{header}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        '<p class="empty">Equal slots (compounding on by default) size each '
        "entry from realized equity / top. Reinvested slots size from current "
        "marked-to-market equity. Pass --no-compounding to freeze equal slots "
        "at initial_capital / top, which leaves profits as idle cash.</p></section>"
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
# Label and colour token per percentile, looked up by the band's own percentile
# so a change to ``band_percentiles`` cannot silently mislabel a line.
_FAN_BAND_STYLE = {
    5: ("p05", "series-2"),
    50: ("median", "series-1"),
    95: ("p95", "series-3"),
}


def _fan_chart_html(
    equity: pd.Series,
    paths: "EquityMonteCarloPaths",
) -> str:
    """Draw the simulated equity paths, their percentile bands, and the real run.

    The x axis is the bar number, not the date. A resampled path draws its
    returns out of calendar order, so no bar of it belongs to any date; plotting
    against the real index would also repeat the date array once per line.

    The bands come from ``paths.bands``, which the bootstrap took over every
    iteration. Recomputing them here from the retained sample would answer a
    different question from the summary table on the same page, and at
    ``--paths 1`` would collapse all three onto one arbitrary path.

    Paths and bands live in separate ECharts datasets. The paths share one
    thinned row set, so the page carries each drawn x position once rather than
    once per line.
    """
    # Empty bands mean the curve had no bar to resample. Retaining no paths is
    # a different case, and it still has bands to draw.
    if paths.bands.size == 0:
        return '<p class="empty">No bars to simulate.</p>'

    datasets: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
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
            # on the same bar as the realized run.
            columns = np.round(np.linspace(0, bars - 1, points)).astype(int)
            drawn = drawn[:, columns]
        else:
            columns = np.arange(bars)
        rows = np.column_stack([columns.astype(float), np.round(drawn.T)])
        datasets.append({"source": rows.tolist()})
        for row in range(drawn.shape[0]):
            series.append(
                {
                    "type": "line",
                    "datasetIndex": 0,
                    "encode": {"x": 0, "y": row + 1},
                    "showSymbol": False,
                    "silent": True,
                    "tooltip": {"show": False},
                    "lineStyle": {"width": 1, "color": "token:muted/0.15"},
                }
            )

    band_rows = [np.round(band) for band in paths.bands]
    realized = np.round(equity.to_numpy(dtype=float))
    band_index = {pct: i for i, pct in enumerate(paths.band_percentiles)}
    stacked = 5 in band_index and 95 in band_index
    table_columns = [np.arange(paths.bands.shape[1], dtype=float), *band_rows, realized]
    # When both p05 and p95 exist, the p05 line is the base of a stack and the
    # band series stacks the p95 - p05 width on top, so its area fills exactly
    # between the two.
    band_dataset = len(datasets)
    datasets.append({"source": np.column_stack(table_columns).tolist()})

    legend: list[str] = []
    realized_col = 1 + len(band_rows)
    for i, pct in enumerate(paths.band_percentiles):
        label, token = _FAN_BAND_STYLE.get(pct, (f"p{pct:02d}", "muted"))
        name = f"Simulated {label}"
        legend.append(name)
        line: dict[str, Any] = {
            "type": "line",
            "name": name,
            "datasetIndex": band_dataset,
            "encode": {"x": 0, "y": i + 1},
            "showSymbol": False,
            "lineStyle": {"width": 1.5, "type": "dashed", "color": f"token:{token}"},
            "itemStyle": {"color": f"token:{token}"},
        }
        if stacked and pct == 5:
            # ECharts only stacks series that carry their own data, not
            # dataset-encoded ones, so the band base and width are inlined.
            del line["datasetIndex"], line["encode"]
            line["stack"] = "band"
            line["data"] = [[float(x), y] for x, y in enumerate(band_rows[i].tolist())]
        series.append(line)
        if stacked and pct == 5:
            width = band_rows[band_index[95]] - band_rows[band_index[5]]
            series.append(
                {
                    "type": "line",
                    "name": "Simulated band",
                    "stack": "band",
                    "data": [[float(x), y] for x, y in enumerate(width.tolist())],
                    "showSymbol": False,
                    "silent": True,
                    "tooltip": {"show": False},
                    "lineStyle": {"width": 0},
                    "areaStyle": {"color": "token:series-1/0.12"},
                }
            )
    series.append(
        {
            "type": "line",
            "name": "Realized run",
            "datasetIndex": band_dataset,
            "encode": {"x": 0, "y": realized_col},
            "showSymbol": False,
            "lineStyle": {"width": 2, "color": "token:ink"},
            "itemStyle": {"color": "token:ink"},
        }
    )
    option: dict[str, Any] = {
        "legend": {"data": [*legend, "Realized run"]},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "dataset": datasets,
        # A category axis over the bar numbers: ECharts only stacks (the p05 to
        # p95 band) on a category base axis, and numeric x values on one are
        # read as category indices, so the thinned paths still land on their bar.
        "xAxis": {
            "type": "category",
            "name": "Bar",
            "nameLocation": "middle",
            "nameGap": 28,
            "boundaryGap": False,
            "data": list(range(paths.bands.shape[1])),
            "axisLabel": {"interval": max(0, paths.bands.shape[1] // 10 - 1)},
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "name": "Equity",
            "scale": True,
            "axisLabel": {"formatter": "fmt:compact"},
        },
        "series": series,
    }
    return chart_html("tearsheet-mc-paths", option, height=420)


def _distribution_html(
    values: np.ndarray,
    div_id: str,
    *,
    realized: float,
    label: str,
) -> str:
    """Histogram of a per-iteration outcome, with the realized run marked."""
    bins = 60
    option = histogram_chart(
        values, bins=bins, name=label, value_format="pct1", signed=False
    )
    # One vectorized call rather than three: each ``np.percentile`` sorts the
    # whole array, and these three want the same sort.
    p05, median, p95 = (float(v) for v in np.percentile(values, [5, 50, 95]))
    # The category axis holds bin lower edges, and bar ``i`` is centred on index
    # ``i``. A value's fractional bin position places its marker line on the
    # bar it falls in.
    edges = np.histogram(np.asarray(values, dtype=float), bins=bins)[1]
    span = float(edges[-1] - edges[0]) or 1.0

    def bin_position(value: float) -> float:
        return (value - float(edges[0])) / span * bins - 0.5

    # Each marker gets its own (position, distance) pair, so labels for values
    # that sit close together land on separate rows instead of overprinting.
    markers = [
        ("p05", p05, "token:series-2", "insideStartTop", 4),
        ("median", median, "token:series-1", "insideStartTop", 22),
        ("p95", p95, "token:series-3", "insideEndTop", 4),
        ("realized", realized, "token:ink", "insideEndTop", 22),
    ]
    option["series"][0]["markLine"] = {
        "symbol": "none",
        "silent": True,
        "data": [
            {
                "xAxis": bin_position(value),
                "lineStyle": {"color": color, "type": "dashed", "width": 1.5},
                "label": {
                    "formatter": name,
                    "position": position,
                    "distance": distance,
                    "color": color,
                    "rotate": 0,
                    "fontSize": 11,
                    "backgroundColor": "token:surface",
                    "padding": [2, 4],
                    "borderRadius": 3,
                },
            }
            for name, value, color, position, distance in markers
        ],
    }
    return chart_html(div_id, option)


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
        dates = curves["date"].tolist()
        perf = line_chart(
            dates,
            [
                {
                    "name": "Strategy",
                    "values": curves["strategy_return"].tolist(),
                    "area": True,
                    "width": 3,
                },
                {
                    "name": "Benchmark",
                    "values": curves["benchmark_return"].tolist(),
                    "color": "series-2",
                },
            ],
            value_format="pct1",
        )
        sections.append(
            '<section class="panel wide" id="equity-vs-benchmark"><h2>Equity vs Benchmark</h2>'
            + chart_html("tearsheet-equity-vs-benchmark", perf, group="ts")
            + "</section>"
        )

        drawdown = drawdown_chart(dates, curves["drawdown"].tolist())
        sections.append(
            '<section class="panel wide" id="drawdown-curve"><h2>Drawdown</h2>'
            + chart_html("tearsheet-drawdown-curve", drawdown, group="ts")
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
        hist = histogram_chart(
            trades["return_pct"].tolist(),
            bins=24,
            name="Trades",
            value_format="pct1",
        )
        sections.append(
            '<section class="panel" id="trade-histogram"><h2>Trade Return Distribution</h2>'
            + chart_html("tearsheet-trade-histogram", hist)
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
    .tab-radio { position: absolute; opacity: 0; pointer-events: none; }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: var(--gap);
    }
    .tabs label {
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) / 2);
      color: var(--muted);
      cursor: pointer;
      padding: 8px 12px;
      background: var(--surface-2);
      font-size: 13px;
    }
    #tab-overview:checked ~ .tabs label[for="tab-overview"],
    #tab-montecarlo:checked ~ .tabs label[for="tab-montecarlo"],
    #tab-ledger:checked ~ .tabs label[for="tab-ledger"] {
      color: var(--ink);
      border-color: var(--accent);
      background: var(--surface);
    }
    .tab-panel { display: none; }
    #tab-overview:checked ~ #overview-panel,
    #tab-montecarlo:checked ~ #montecarlo-panel,
    #tab-ledger:checked ~ #ledger-panel {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--gap);
    }
    .metrics { grid-column: 1 / -1; }
    @media (max-width: 900px) {
      #tab-overview:checked ~ #overview-panel,
      #tab-montecarlo:checked ~ #montecarlo-panel,
      #tab-ledger:checked ~ #ledger-panel {
        grid-template-columns: 1fr;
      }
      .metrics, .wide { grid-column: auto; }
    }"""
    page_html = html_page(
        html.escape(title),
        _css,
        f"""\
  <header>
    <div class="eyebrow">Backtest tear-sheet</div>
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
        charts=True,
    )
    output_path.write_text(page_html, encoding="utf-8")
    return output_path
