"""The single visual system every HTML report renders with ("Slate").

All self-contained report pages (tear-sheets, the rolling dashboard, the screen
report, optimization reports, sector rotation) share one colour system and one
charting library so they read as one product:

* **Colour** lives only in the CSS custom properties of :data:`THEME_CSS`, with
  a light and a dark step of every token (the page follows the OS colour
  scheme). Page CSS and chart specs refer to roles (``--series-1``, ``--pos``),
  never raw hex, so a theme change is a token change.
* **Charts** are Apache ECharts, vendored under ``screener/assets`` so a report
  opens offline exactly like the old inlined Plotly bundle did. Python builds a
  plain JSON option per chart; a small runtime (:data:`_RUNTIME_JS`) resolves
  theme tokens and named formatters, applies the shared axis/tooltip styling and
  re-renders when the colour scheme flips.

Inside a chart option, two string prefixes are resolved in the browser:

* ``"token:<name>"`` -> the current value of CSS variable ``--<name>``;
  ``"token:<name>/<alpha>"`` adds transparency (``"token:series-1/0.2"``).
* ``"fmt:<name>"`` -> a formatter function from :data:`FORMATTERS` (usable for
  any ``formatter`` / ``valueFormatter`` key).

The categorical order and every step below were checked with the dataviz
palette validator (lightness band, chroma floor, CVD separation, contrast) on
the light (``#fcfcfb``) and dark (``#1a1a19``) chart surfaces.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

# Named JS formatters the runtime understands (see ``fmt:`` above).
FORMATTERS = (
    "pct0",  # fraction -> "12%"
    "pct1",  # fraction -> "12.3%"
    "pct2",  # fraction -> "12.34%"
    "pctv0",  # already-percent value -> "12%"
    "pctv1",  # already-percent value -> "12.3%"
    "num0",  # 12,345
    "num1",  # 12,345.6
    "num2",  # 12,345.67
    "compact",  # 12.3k / 1.2M
    "date",  # YYYY-MM-DD from a time-axis value
)

_LIGHT = {
    "bg": "#f4f4f2",
    "surface": "#fcfcfb",
    "surface-2": "#f0efec",
    "ink": "#0b0b0b",
    "muted": "#52514e",
    "line": "#e2e1dc",
    "line-soft": "#ecebe7",
    "hover": "#f4f3ef",
    "accent": "#2a78d6",
    "series-1": "#2a78d6",
    "series-2": "#eb6834",
    "series-3": "#1baf7a",
    "series-4": "#eda100",
    "series-5": "#e87ba4",
    "series-6": "#008300",
    "series-7": "#6250d6",
    "series-8": "#e34948",
    "pos": "#2a78d6",
    "neg": "#e34948",
    "pos-ink": "#1c5cab",
    "neg-ink": "#b8302f",
    "div-mid": "#f0efec",
    "on-strong": "#ffffff",
    "status-good": "#0ca30c",
    "status-warning": "#fab219",
    "status-critical": "#d03b3b",
}

_DARK = {
    "bg": "#111110",
    "surface": "#1a1a19",
    "surface-2": "#22221f",
    "ink": "#f0efec",
    "muted": "#a8a79f",
    "line": "#2e2e2b",
    "line-soft": "#262624",
    "hover": "#232320",
    "accent": "#3987e5",
    "series-1": "#3987e5",
    "series-2": "#d95926",
    "series-3": "#199e70",
    "series-4": "#c98500",
    "series-5": "#d55181",
    "series-6": "#008300",
    "series-7": "#9085e9",
    "series-8": "#e66767",
    "pos": "#3987e5",
    "neg": "#e66767",
    "pos-ink": "#86b6ef",
    "neg-ink": "#f09090",
    "div-mid": "#383835",
    "on-strong": "#ffffff",
    "status-good": "#0ca30c",
    "status-warning": "#fab219",
    "status-critical": "#d03b3b",
}

#: Every colour token a page or chart may reference (``--<name>``).
TOKENS: tuple[str, ...] = tuple(_LIGHT)


def _token_block(values: Mapping[str, str]) -> str:
    return "\n".join(f"  --{name}: {value};" for name, value in values.items())


THEME_CSS = f"""\
:root {{
  color-scheme: light;
{_token_block(_LIGHT)}
  --font-ui: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --radius: 10px;
  --gap: 14px;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
{_token_block(_DARK).replace("  --", "    --")}
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 var(--font-ui);
  -webkit-font-smoothing: antialiased;
}}
a {{ color: var(--accent); }}
h1, h2, h3 {{ margin: 0; font-weight: 600; letter-spacing: -0.01em; }}
h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }}
h2 {{ font-size: 14px; margin-bottom: 12px; }}
h3 {{ font-size: 13px; margin-bottom: 8px; color: var(--muted); }}
header {{
  max-width: 1440px;
  margin: 0 auto;
  padding: 32px 32px 20px;
  border-bottom: 1px solid var(--line);
}}
header .eyebrow {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}}
.subhead {{
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin-top: 8px;
  font-size: 13px;
}}
main {{ max-width: 1440px; margin: 0 auto; padding: 20px 32px 48px; }}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--gap);
}}
.metric, .panel {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
}}
.metric {{ padding: 14px 16px; }}
.metric span {{ display: block; color: var(--muted); font-size: 12px; }}
.metric strong {{
  display: block;
  margin-top: 4px;
  font-size: 22px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}
.panel {{ padding: 18px; min-width: 0; }}
.panel > .hint {{ margin: -8px 0 12px; color: var(--muted); font-size: 12px; }}
.wide {{ grid-column: 1 / -1; }}
.chart-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--gap);
}}
.echart {{ width: 100%; }}
.table-wrap {{ overflow: auto; max-height: 520px; }}
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
.data-table th {{
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface);
  color: var(--muted);
  font-weight: 500;
  text-align: left;
}}
.data-table th, .data-table td {{
  padding: 7px 9px;
  border-bottom: 1px solid var(--line-soft);
}}
.data-table tbody tr:hover td {{ background: var(--hover); }}
.num-pos {{ color: var(--pos-ink); }}
.num-neg {{ color: var(--neg-ink); }}
.heatmap td {{ text-align: center; border: 2px solid var(--surface); }}
.heatmap td.strong {{ color: var(--on-strong); }}
.empty, .warnings {{ color: var(--muted); font-size: 13px; }}
.warnings {{ margin: 0; padding-left: 18px; }}
.badge {{
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  color: #0b0b0b;
}}
.badge.good {{ background: var(--status-good); color: #ffffff; }}
.badge.warning {{ background: var(--status-warning); }}
.badge.critical {{ background: var(--status-critical); color: #ffffff; }}
.callout {{
  padding: 10px 14px;
  border-radius: var(--radius);
  border: 1px solid color-mix(in oklab, var(--status-warning) 55%, var(--line));
  background: color-mix(in oklab, var(--status-warning) 14%, var(--surface));
}}
pre, code {{ font-family: var(--font-mono); font-size: 12px; }}
pre {{
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
  overflow: auto;
}}
input, select, button {{
  font: inherit;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: calc(var(--radius) / 2);
  padding: 8px 10px;
}}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
@media (max-width: 900px) {{
  header, main {{ padding-left: 16px; padding-right: 16px; }}
  .chart-grid {{ grid-template-columns: 1fr; }}
}}"""

_RUNTIME_JS = r"""
(function () {
  "use strict";
  var FMT = {
    pct0: function (v) { return (v * 100).toFixed(0) + "%"; },
    pct1: function (v) { return (v * 100).toFixed(1) + "%"; },
    pct2: function (v) { return (v * 100).toFixed(2) + "%"; },
    pctv0: function (v) { return Number(v).toFixed(0) + "%"; },
    pctv1: function (v) { return Number(v).toFixed(1) + "%"; },
    num0: function (v) { return Math.round(v).toLocaleString("en-US"); },
    num1: function (v) { return Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 }); },
    num2: function (v) { return Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }); },
    compact: function (v) { return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v); },
    date: function (v) { return new Date(v).toISOString().slice(0, 10); }
  };
  function guard(f) {
    return function (v) {
      if (v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v))) return "-";
      if (typeof v === "object" && "value" in v) v = Array.isArray(v.value) ? v.value[v.value.length - 1] : v.value;
      return f(v);
    };
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim(); }
  function alpha(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.replace(/./g, "$&$&");
    var n = parseInt(h, 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
  }
  function resolve(node) {
    if (typeof node === "string") {
      if (node.indexOf("token:") === 0) {
        var parts = node.slice(6).split("/");
        var value = css(parts[0]);
        return parts.length > 1 ? alpha(value, parts[1]) : value;
      }
      if (node.indexOf("fmt:") === 0 && FMT[node.slice(4)]) return guard(FMT[node.slice(4)]);
      return node;
    }
    if (Array.isArray(node)) return node.map(resolve);
    if (node && typeof node === "object") {
      var out = {};
      for (var k in node) out[k] = resolve(node[k]);
      return out;
    }
    return node;
  }
  function merge(base, extra) {
    if (extra === undefined) return base;
    if (!base || typeof base !== "object" || Array.isArray(base) || !extra || typeof extra !== "object" || Array.isArray(extra)) return extra;
    var out = {};
    var k;
    for (k in base) out[k] = base[k];
    for (k in extra) out[k] = merge(base[k], extra[k]);
    return out;
  }
  function axisDefaults() {
    return {
      axisLine: { lineStyle: { color: css("line") } },
      axisTick: { show: false },
      axisLabel: { color: css("muted"), fontFamily: css("font-ui"), hideOverlap: true },
      splitLine: { lineStyle: { color: css("line-soft") } },
      nameTextStyle: { color: css("muted") }
    };
  }
  function themed(spec) {
    var opt = resolve(spec);
    var base = {
      backgroundColor: "transparent",
      animation: false,
      color: [1, 2, 3, 4, 5, 6, 7, 8].map(function (i) { return css("series-" + i); }),
      textStyle: { fontFamily: css("font-ui"), color: css("muted") },
      grid: { left: 8, right: 24, top: 16, bottom: 8, containLabel: true },
      tooltip: {
        backgroundColor: css("surface"),
        borderColor: css("line"),
        textStyle: { color: css("ink"), fontFamily: css("font-ui"), fontSize: 12 },
        axisPointer: { lineStyle: { color: css("muted"), type: "dashed" }, shadowStyle: { color: alpha(css("muted"), 0.08) } }
      }
    };
    if (opt.legend) {
      base.legend = { textStyle: { color: css("muted") }, icon: "roundRect", itemWidth: 14, itemHeight: 3, top: 0 };
      if (!opt.grid) base.grid.top = 36;
    } else {
      base.legend = { show: false };
    }
    var out = merge(base, opt);
    ["xAxis", "yAxis"].forEach(function (key) {
      if (!out[key]) return;
      var list = Array.isArray(out[key]) ? out[key] : [out[key]];
      var styled = list.map(function (ax) { return merge(axisDefaults(), ax); });
      out[key] = Array.isArray(out[key]) ? styled : styled[0];
    });
    return out;
  }
  var charts = [];
  function renderAll() {
    charts.forEach(function (c) { c.chart.setOption(themed(c.spec), true); });
  }
  function boot() {
    if (!window.echarts) return;
    var groups = {};
    document.querySelectorAll("[data-echart]").forEach(function (el) {
      var src = document.getElementById(el.getAttribute("data-echart"));
      if (!src) return;
      var spec = JSON.parse(src.textContent);
      var chart = echarts.init(el, null, { renderer: "svg" });
      chart.setOption(themed(spec), true);
      charts.push({ chart: chart, spec: spec });
      var group = el.getAttribute("data-echart-group");
      if (group) { chart.group = group; groups[group] = true; }
      if (window.ResizeObserver) new ResizeObserver(function () { chart.resize(); }).observe(el);
    });
    Object.keys(groups).forEach(function (g) { echarts.connect(g); });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(renderAll);
    }
  }
  window.screenerCharts = { charts: charts, render: renderAll, formatters: FMT };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
"""


@cache
def echarts_js() -> str:
    """Return the vendored ECharts bundle (Apache-2.0, see ECHARTS_LICENSE)."""
    return (
        resources.files("screener.assets")
        .joinpath("echarts.min.js")
        .read_text(encoding="utf-8")
    )


def chart_scripts() -> str:
    """``<script>`` tags for the ECharts bundle plus the theme runtime."""
    return f"<script>{echarts_js()}</script>\n<script>{_RUNTIME_JS}</script>"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Make ``value`` strict-JSON safe: NaN/inf -> None, numpy/pandas -> builtins."""
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_clean(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_for_script(option: Mapping[str, Any]) -> str:
    """Serialize for an inline ``<script type="application/json">`` block.

    ``<``, ``>`` and ``&`` are escaped so data (ticker names, notes) can never
    close the script element or be parsed as markup.
    """
    text = json.dumps(_clean(option), separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def chart_html(
    chart_id: str,
    option: Mapping[str, Any],
    *,
    height: int = 320,
    group: str | None = None,
) -> str:
    """Return a sized chart container plus its JSON option.

    ``chart_id`` becomes the container's DOM id (tests and anchors use it).
    Charts sharing a ``group`` get a connected tooltip/crosshair (e.g. equity
    and drawdown over the same dates).
    """
    group_attr = f' data-echart-group="{group}"' if group else ""
    return (
        f'<div class="echart" id="{chart_id}" data-echart="{chart_id}-spec"'
        f'{group_attr} style="height:{int(height)}px"></div>'
        f'<script type="application/json" id="{chart_id}-spec">'
        f"{_json_for_script(option)}</script>"
    )


# ---------------------------------------------------------------------------
# Option builders. Each returns a plain dict; callers may extend it before
# passing it to :func:`chart_html`.
# ---------------------------------------------------------------------------


def _series_values(x: Sequence[Any], values: Iterable[Any]) -> list[list[Any]]:
    return [[_clean(a), _clean(b)] for a, b in zip(x, values, strict=False)]


def _ends_apart(series: Sequence[Mapping[str, Any]], min_gap: float = 0.06) -> bool:
    """Whether the series' last values are far enough apart to label them.

    The gap is relative to the plotted range; closer than ``min_gap`` and the
    end labels would print on top of each other (the legend still names them).
    """
    lasts, everything = [], []
    for spec in series:
        vals = [v for v in (_clean(x) for x in spec["values"]) if v is not None]
        if not vals:
            return False
        lasts.append(vals[-1])
        everything.extend(vals)
    span = max(everything) - min(everything) or 1.0
    lasts.sort()
    return all(b - a >= min_gap * span for a, b in zip(lasts, lasts[1:], strict=False))


def line_chart(
    x: Sequence[Any],
    series: Sequence[Mapping[str, Any]],
    *,
    value_format: str = "num2",
    x_type: str = "time",
    y_scale: bool = True,
    end_labels: bool = True,
) -> dict[str, Any]:
    """Line/area chart over a shared x axis (dates by default).

    Each ``series`` entry: ``name``, ``values``, optional ``color`` (a token
    name, default ``series-<n>``), ``area`` (fill to the baseline), ``width``
    and ``dashed``.
    """
    show_end_labels = end_labels and len(series) > 1 and _ends_apart(series)
    out_series: list[dict[str, Any]] = []
    for i, spec in enumerate(series):
        color = f"token:{spec.get('color', f'series-{i + 1}')}"
        entry: dict[str, Any] = {
            "name": spec["name"],
            "type": "line",
            "showSymbol": False,
            "symbolSize": 8,
            "data": _series_values(x, spec["values"]),
            "lineStyle": {
                "width": spec.get("width", 2),
                "color": color,
                "type": "dashed" if spec.get("dashed") else "solid",
            },
            "itemStyle": {"color": color},
            "emphasis": {"focus": "none", "lineStyle": {"width": spec.get("width", 2)}},
        }
        if spec.get("area"):
            base = color.split("/")[0]
            entry["areaStyle"] = {
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": f"{base}/0.22"},
                        {"offset": 1, "color": f"{base}/0.02"},
                    ],
                }
            }
        if show_end_labels:
            entry["endLabel"] = {
                "show": True,
                "formatter": "{a}",
                "color": "token:ink",
                "fontWeight": 600,
            }
        out_series.append(entry)
    option: dict[str, Any] = {
        "tooltip": {"trigger": "axis", "valueFormatter": f"fmt:{value_format}"},
        "xAxis": {"type": x_type, "boundaryGap": False, "splitLine": {"show": False}},
        "yAxis": {
            "type": "value",
            "scale": y_scale,
            "axisLabel": {"formatter": f"fmt:{value_format}"},
        },
        "series": out_series,
    }
    if len(series) > 1:
        option["legend"] = {"data": [s["name"] for s in series], "left": 0}
        if show_end_labels:
            option["grid"] = {
                "left": 8,
                "right": 88,
                "top": 36,
                "bottom": 8,
                "containLabel": True,
            }
    if x_type == "time":
        option["tooltip"]["axisPointer"] = {"label": {"formatter": "fmt:date"}}
    return option


def drawdown_chart(
    x: Sequence[Any],
    drawdown: Sequence[Any],
    *,
    name: str = "Drawdown",
) -> dict[str, Any]:
    """Underwater curve (fractions <= 0) as a ``neg`` area pinned to ``[deepest, 0]``.

    A flat-zero curve keeps a -1% floor so the axis never collapses or jumps to
    an auto-scaled 0..100%.
    """
    values = [v for v in (_clean(d) for d in drawdown) if v is not None]
    deepest = min([0.0, *values])
    option = line_chart(
        x,
        [{"name": name, "values": drawdown, "color": "neg", "area": True}],
        value_format="pct1",
        end_labels=False,
    )
    option["yAxis"].update({"scale": False, "min": min(deepest, -0.01), "max": 0})
    return option


def signed_colors(values: Iterable[Any]) -> list[str]:
    """Per-bar colour tokens: ``pos`` for >= 0, ``neg`` for < 0."""
    out = []
    for v in values:
        cleaned = _clean(v)
        out.append("token:neg" if cleaned is not None and cleaned < 0 else "token:pos")
    return out


def bar_chart(
    categories: Sequence[Any],
    values: Sequence[Any],
    *,
    name: str = "",
    value_format: str = "num0",
    horizontal: bool = False,
    color: str = "series-1",
    colors: Sequence[str] | None = None,
    labels: bool = False,
) -> dict[str, Any]:
    """Single-series bar chart. ``colors`` (e.g. :func:`signed_colors`) wins over ``color``.

    Horizontal charts draw the first category at the top.
    """
    cats = [_clean(c) for c in categories]
    vals = [_clean(v) for v in values]
    bar_colors = list(colors) if colors is not None else [f"token:{color}"] * len(vals)
    if horizontal:
        cats, vals, bar_colors = cats[::-1], vals[::-1], bar_colors[::-1]
    radius_pos = [0, 4, 4, 0] if horizontal else [4, 4, 0, 0]
    radius_neg = [4, 0, 0, 4] if horizontal else [0, 0, 4, 4]
    data = []
    for v, c in zip(vals, bar_colors, strict=False):
        item: dict[str, Any] = {
            "value": v,
            "itemStyle": {
                "color": c,
                "borderRadius": radius_neg if (v is not None and v < 0) else radius_pos,
            },
        }
        if labels and horizontal:
            item["label"] = {
                "position": "left" if (v is not None and v < 0) else "right"
            }
        data.append(item)
    cat_axis: dict[str, Any] = {
        "type": "category",
        "data": cats,
        "splitLine": {"show": False},
    }
    val_axis: dict[str, Any] = {
        "type": "value",
        "axisLabel": {"formatter": f"fmt:{value_format}"},
    }
    series: dict[str, Any] = {
        "type": "bar",
        "name": name,
        "data": data,
        "barMaxWidth": 28,
        "barCategoryGap": "18%",
    }
    if labels:
        series["label"] = {
            "show": True,
            "position": "right" if horizontal else "top",
            "color": "token:ink",
            "formatter": f"fmt:{value_format}",
        }
    option: dict[str, Any] = {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "valueFormatter": f"fmt:{value_format}",
        },
        "xAxis": val_axis if horizontal else cat_axis,
        "yAxis": cat_axis if horizontal else val_axis,
        "series": [series],
    }
    if labels and horizontal:
        # Value labels sit outside the bar ends; leave room so the longest
        # one is not clipped at the plot edge.
        option["grid"] = {
            "left": 8,
            "right": 56,
            "top": 16,
            "bottom": 8,
            "containLabel": True,
        }
    return option


def histogram_chart(
    values: Iterable[Any],
    *,
    bins: int = 24,
    name: str = "Count",
    value_format: str = "num1",
    signed: bool = True,
) -> dict[str, Any]:
    """Histogram of ``values`` (NaN dropped). ``signed`` colours bins below 0 with ``neg``.

    ``value_format`` formats the bin edges on the x axis.
    """
    arr = np.asarray(
        [v for v in (_clean(x) for x in values) if v is not None], dtype=float
    )
    if arr.size == 0:
        return bar_chart([], [], name=name)
    if float(np.ptp(arr)) == 0.0:
        # A degenerate sample (every value equal) is one bar at that value;
        # numpy would spread it over +-0.5 and label the bar with a made-up edge.
        counts, edges = np.array([arr.size]), np.array([arr[0], arr[0]])
    else:
        counts, edges = np.histogram(arr, bins=max(int(bins), 1))
    data = [
        {
            "value": int(count),
            "itemStyle": {
                "color": "token:neg" if signed and hi <= 0 else "token:series-1",
                "borderRadius": [3, 3, 0, 0],
            },
        }
        for count, hi in zip(counts, edges[1:], strict=False)
    ]
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "category",
            "name": "",
            # Category = the bin's lower edge; the label formatter renders it.
            "data": [float(lo) for lo in edges[:-1]],
            "axisLabel": {"formatter": f"fmt:{value_format}"},
            "splitLine": {"show": False},
        },
        "yAxis": {"type": "value", "minInterval": 1},
        "series": [{"type": "bar", "name": name, "data": data, "barCategoryGap": "6%"}],
    }


def diverging_cell_style(value: Any, max_abs: float) -> tuple[str, str]:
    """Background style + extra class for a diverging heatmap table cell.

    Mixes the ``pos``/``neg`` pole into the neutral ``div-mid`` token by
    magnitude. Returns ``("", "")`` for missing values.
    """
    v = _clean(value)
    if v is None or not max_abs:
        return "", ""
    t = min(1.0, abs(float(v)) / float(max_abs))
    pole = "var(--pos)" if v >= 0 else "var(--neg)"
    pct = round(12 + t * 78)
    style = f"background:color-mix(in oklab, {pole} {pct}%, var(--div-mid))"
    return style, ("strong" if t > 0.55 else "")


__all__ = [
    "FORMATTERS",
    "THEME_CSS",
    "TOKENS",
    "bar_chart",
    "chart_html",
    "chart_scripts",
    "diverging_cell_style",
    "drawdown_chart",
    "echarts_js",
    "histogram_chart",
    "line_chart",
    "signed_colors",
]
