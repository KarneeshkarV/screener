"""Serve screener result files in a browser dashboard.

Supports three result families:
  * /tmp/screener-pine-results     (broad TradingView universes)
  * /tmp/screener-index-results    (fixed S&P 500 / Nifty 500 Pine runs)
  * /tmp/screener-ranked-results   (always-invested ranked portfolios)
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_DATASETS = {
    "broad_pine": "/tmp/screener-pine-results",
    "index_pine": "/tmp/screener-index-results",
    "ranked": "/tmp/screener-ranked-results",
}


PINE_HEADER_RE = re.compile(
    r"^(?P<market>US|INDIA)\s+\|\s+window\s+"
    r"(?P<start>\d{4}-\d{2}-\d{2})\s+.\s+"
    r"(?P<end>\d{4}-\d{2}-\d{2})\s+\|\s+bench="
    r"(?P<benchmark>[^=]+)=(?P<bench_return>[^\s]+)"
)
RANKED_WINDOW_RE = re.compile(
    r"^Window:\s+(?P<start>\d{4}-\d{2}-\d{2})\s+->\s+"
    r"(?P<end>\d{4}-\d{2}-\d{2})\s+\((?P<years>\d+)y\)"
)
RANKED_UNIVERSE_RE = re.compile(r"^Universe:\s+(?P<market>us|india)\s+(?P<count>\d+)\s+tickers")


@dataclass
class ResultRow:
    dataset: str
    file: str
    family: str
    market: str
    years: int | None
    start: str
    end: str
    strategy: str
    benchmark: str | None
    benchmark_return: float | None
    total_return: float | None
    basket: float | None
    cagr: float | None
    median: float | None
    alpha: float | None
    equal_weight_return: float | None
    equal_weight_alpha: float | None
    sharpe: float | None
    win_rate: float | None
    max_drawdown: float | None
    trades: int | None
    tickers: int | None
    avg_holdings: float | None
    days_in: int | None
    exposure: float | None
    turnover: float | None


def _parse_value(raw: str) -> float | None:
    raw = raw.strip()
    if raw in {"-", ""} or raw.lower() == "nan%":
        return None
    if raw.endswith("%"):
        return float(raw[:-1].replace("+", "")) / 100.0
    return float(raw.replace("+", ""))


def _years_from_name(path: Path) -> int | None:
    match = re.search(r"_(\d+)y", path.stem)
    return int(match.group(1)) if match else None


def _parse_pine_file(dataset: str, path: Path) -> list[ResultRow]:
    meta: dict[str, object] = {}
    rows: list[ResultRow] = []
    in_table = False
    for line in path.read_text(errors="replace").splitlines():
        match = PINE_HEADER_RE.match(line)
        if match:
            meta = {
                "market": match.group("market").lower(),
                "start": match.group("start"),
                "end": match.group("end"),
                "benchmark": match.group("benchmark"),
                "benchmark_return": _parse_value(match.group("bench_return")),
                "years": _years_from_name(path),
            }
            continue
        if line.startswith("Strategy"):
            in_table = True
            continue
        if not in_table or not line.strip() or line.startswith("-") or "no results" in line:
            continue
        if line.startswith("Best in this market:"):
            break
        parts = line.split()
        if len(parts) < 12:
            continue
        strategy, tickers, trades = parts[0], int(parts[1]), int(parts[2])
        basket = _parse_value(parts[4])
        median = _parse_value(parts[5])
        alpha = _parse_value(parts[7])
        sharpe = _parse_value(parts[8])
        win_rate = _parse_value(parts[9])
        days_in = None if parts[10] == "-" else int(parts[10])
        exposure = _parse_value(parts[11])
        rows.append(
            ResultRow(
                dataset=dataset,
                file=path.name,
                family="pine",
                market=str(meta.get("market")),
                years=meta.get("years"),  # type: ignore[arg-type]
                start=str(meta.get("start")),
                end=str(meta.get("end")),
                strategy=strategy,
                benchmark=meta.get("benchmark"),  # type: ignore[arg-type]
                benchmark_return=meta.get("benchmark_return"),  # type: ignore[arg-type]
                total_return=basket,
                basket=basket,
                cagr=None,
                median=median,
                alpha=alpha,
                equal_weight_return=None,
                equal_weight_alpha=None,
                sharpe=sharpe,
                win_rate=win_rate,
                max_drawdown=None,
                trades=trades,
                tickers=tickers,
                avg_holdings=None,
                days_in=days_in,
                exposure=exposure,
                turnover=None,
            )
        )
    return rows


def _parse_ranked_file(dataset: str, path: Path) -> list[ResultRow]:
    market = start = end = None
    years = None
    rows: list[ResultRow] = []
    in_table = False
    for line in path.read_text(errors="replace").splitlines():
        u = RANKED_UNIVERSE_RE.match(line)
        if u:
            market = u.group("market")
            continue
        w = RANKED_WINDOW_RE.match(line)
        if w:
            start, end, years = w.group("start"), w.group("end"), int(w.group("years"))
            continue
        if line.startswith("Strategy"):
            in_table = True
            continue
        if not in_table or not line.strip() or line.startswith("-"):
            continue
        if line.startswith("Best alpha:"):
            break
        parts = line.split()
        if len(parts) < 11:
            continue
        rows.append(
            ResultRow(
                dataset=dataset,
                file=path.name,
                family="ranked",
                market=market or "",
                years=years,
                start=start or "",
                end=end or "",
                strategy=parts[0],
                benchmark="benchmark",
                benchmark_return=_parse_value(parts[5]),
                total_return=_parse_value(parts[1]),
                basket=None,
                cagr=_parse_value(parts[2]),
                median=None,
                alpha=_parse_value(parts[6]),
                equal_weight_return=_parse_value(parts[7]),
                equal_weight_alpha=_parse_value(parts[8]),
                sharpe=_parse_value(parts[3]),
                win_rate=None,
                max_drawdown=_parse_value(parts[4]),
                trades=None,
                tickers=None,
                avg_holdings=float(parts[9]),
                days_in=None,
                exposure=None,
                turnover=float(parts[10]),
            )
        )
    return rows


def load_dataset(dataset: str, root: Path) -> list[dict]:
    rows: list[ResultRow] = []
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(errors="replace")
        if "Portfolio: top=" in text:
            rows.extend(_parse_ranked_file(dataset, path))
        elif "bench=" in text and "Strategy" in text:
            rows.extend(_parse_pine_file(dataset, path))
    return [asdict(r) for r in rows]


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strategy Lab</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,.72);
      --panel-strong: rgba(255,255,255,.88);
      --ink: #10221a;
      --muted: #5e6f67;
      --line: rgba(18,34,26,.12);
      --good: #0f8a5f;
      --bad: #d14d41;
      --gold: #d3a74f;
      --blue: #0e6ba8;
      --shadow: 0 22px 60px rgba(31,39,31,.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(14,107,168,.14), transparent 32rem),
        radial-gradient(circle at right center, rgba(211,167,79,.18), transparent 32rem),
        linear-gradient(160deg, #f6f1e6 0%, #eef2eb 58%, #f3e6d8 100%);
      min-height: 100vh;
    }
    main { width: min(1280px, calc(100% - 28px)); margin: 22px auto 40px; }
    .hero, .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(16px);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 26px 28px 24px;
      margin-bottom: 18px;
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 22px;
      align-items: end;
    }
    h1 {
      margin: 0;
      font-size: clamp(42px, 8vw, 88px);
      line-height: .88;
      letter-spacing: -.06em;
    }
    .lede {
      color: var(--muted);
      font-size: 16px;
      max-width: 640px;
      margin: 12px 0 0;
    }
    .hero-stats {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr 1fr;
    }
    .stat {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }
    .stat small { display:block; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 11px; }
    .stat strong { display:block; font-size: 30px; margin-top: 6px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      padding: 16px;
      margin-bottom: 18px;
    }
    label { display:grid; gap:6px; color: var(--muted); font-size:11px; text-transform: uppercase; letter-spacing:.12em; }
    select {
      width: 100%;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 11px 14px;
      background: rgba(255,255,255,.92);
      color: var(--ink);
      font: inherit;
    }
    .grid { display:grid; gap:18px; grid-template-columns: 1.25fr .75fr; }
    .panel { padding: 18px; }
    h2 { margin: 0 0 12px; font-size: 22px; }
    .chart { display:grid; gap:10px; }
    .bar-row { display:grid; gap:12px; grid-template-columns: 180px 1fr 90px; align-items:center; }
    .bar-name { font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar-track {
      position: relative;
      height: 18px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(16,34,26,.08), rgba(16,34,26,.03));
      overflow: hidden;
    }
    .bar-zero { position:absolute; top:0; bottom:0; width:1px; left:50%; background: rgba(16,34,26,.22); }
    .bar { position:absolute; top:0; bottom:0; border-radius:999px; }
    .bar.pos { background: linear-gradient(90deg, #21a377, var(--good)); }
    .bar.neg { background: linear-gradient(90deg, #f08d7f, var(--bad)); }
    .metric { text-align:right; font-variant-numeric: tabular-nums; }
    .cards { display:grid; gap:12px; }
    .mini {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }
    .mini small { display:block; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size:11px; }
    .mini strong { display:block; font-size: 22px; margin-top: 6px; }
    .mini span { display:block; margin-top: 4px; color: var(--muted); font-size: 14px; }
    table {
      width:100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    th:first-child, td:first-child { text-align:left; }
    th { color: var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    .badge {
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:8px 12px;
      border-radius:999px;
      border:1px solid var(--line);
      background: rgba(255,255,255,.85);
      color: var(--muted);
      font-size: 12px;
      margin-right: 8px;
    }
    .kicker { margin-bottom: 8px; }
    @media (max-width: 980px) {
      .hero, .grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr 1fr; }
      .bar-row { grid-template-columns: 120px 1fr 72px; }
    }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <div class="kicker">
        <span class="badge">Strategy Lab</span>
        <span class="badge" id="datasetBadge">Dataset</span>
      </div>
      <h1>Benchmark<br>Workbench</h1>
      <p class="lede">Explore Pine signal systems, fixed-index runs, and always-invested ranking portfolios in one place. Compare alpha, Sharpe, drawdown, win rate, turnover, and exposure without digging through text files.</p>
    </div>
    <div class="hero-stats">
      <div class="stat"><small>Rows Loaded</small><strong id="rowsLoaded">0</strong></div>
      <div class="stat"><small>Current Leader</small><strong id="heroLeader">-</strong></div>
      <div class="stat"><small>Metric</small><strong id="heroMetric">-</strong></div>
      <div class="stat"><small>Window</small><strong id="heroWindow">-</strong></div>
    </div>
  </section>

  <section class="panel controls">
    <label>Dataset<select id="dataset"></select></label>
    <label>Market<select id="market"></select></label>
    <label>Window<select id="years"></select></label>
    <label>Metric<select id="metric"></select></label>
    <label>Sort<select id="sort"></select></label>
  </section>

  <section class="grid">
    <section class="panel">
      <h2 id="chartTitle">Chart</h2>
      <div id="chart" class="chart"></div>
    </section>
    <section class="panel">
      <h2>Highlights</h2>
      <div id="cards" class="cards"></div>
    </section>
  </section>

  <section class="panel" style="margin-top:18px">
    <h2>Result Table</h2>
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Total</th>
          <th>Alpha</th>
          <th>Sharpe</th>
          <th>Bench</th>
          <th>EW Alpha</th>
          <th>Win%</th>
          <th>MaxDD</th>
          <th>Days/Turn</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </section>
</main>
<script>
const DATASETS = {
  broad_pine: {label: "Broad Pine", metrics: ["alpha","basket","sharpe","win_rate","days_in","exposure"]},
  index_pine: {label: "Index Pine", metrics: ["alpha","basket","sharpe","win_rate","days_in","exposure"]},
  ranked: {label: "Ranked Portfolio", metrics: ["alpha","total_return","sharpe","equal_weight_alpha","max_drawdown","turnover"]}
};
let ALL_ROWS = [];
const $ = id => document.getElementById(id);
const pct = v => v == null ? "-" : `${(v * 100).toFixed(1)}%`;
const num = v => v == null ? "-" : Number(v).toFixed(2);
const intv = v => v == null ? "-" : String(Math.round(v));
const metricLabel = {
  alpha: "Alpha vs Benchmark", basket: "Basket Return", total_return: "Portfolio Return",
  sharpe: "Sharpe Ratio", win_rate: "Win Rate", days_in: "Days In Market", exposure: "Exposure",
  equal_weight_alpha: "Alpha vs Equal Weight", max_drawdown: "Max Drawdown", turnover: "Turnover"
};
function metricText(metric, value) {
  if (["sharpe","turnover"].includes(metric)) return num(value);
  if (["days_in"].includes(metric)) return intv(value);
  return pct(value);
}
function uniq(values) { return [...new Set(values.filter(v => v != null && v !== ""))]; }
function fillSelect(id, values, labels) {
  $(id).innerHTML = values.map(v => `<option value="${v}">${labels?.[v] || v}</option>`).join("");
}
function currentFamilyRows() {
  return ALL_ROWS.filter(r => r.dataset === $("dataset").value);
}
function currentRows() {
  const rows = currentFamilyRows();
  const market = $("market").value;
  const years = Number($("years").value);
  const sort = $("sort").value;
  return rows
    .filter(r => r.market === market && Number(r.years) === years)
    .sort((a, b) => (b[sort] ?? -Infinity) - (a[sort] ?? -Infinity));
}
function refreshSelectors() {
  const familyRows = currentFamilyRows();
  fillSelect("market", uniq(familyRows.map(r => r.market)));
  fillSelect("years", uniq(familyRows.map(r => r.years)).sort((a,b)=>a-b));
  const ds = DATASETS[$("dataset").value];
  fillSelect("metric", ds.metrics, metricLabel);
  fillSelect("sort", ds.metrics, metricLabel);
  $("sort").value = $("metric").value;
}
function render() {
  const dataset = $("dataset").value;
  const rows = currentRows();
  const metric = $("metric").value;
  $("datasetBadge").textContent = DATASETS[dataset].label;
  $("rowsLoaded").textContent = currentFamilyRows().length;
  $("chartTitle").textContent = `${$("market").value.toUpperCase()} ${$("years").value}Y • ${metricLabel[metric]}`;
  const leader = rows[0];
  $("heroLeader").textContent = leader?.strategy || "-";
  $("heroMetric").textContent = leader ? metricText(metric, leader[metric]) : "-";
  $("heroWindow").textContent = rows[0] ? `${rows[0].start} to ${rows[0].end}` : "-";

  const vals = rows.map(r => r[metric]).filter(v => v != null);
  const maxAbs = Math.max(0.0001, ...vals.map(v => Math.abs(v)));
  const zeroLeft = ["sharpe","turnover","days_in","win_rate","exposure"].includes(metric) ? 0 : 50;
  $("chart").innerHTML = rows.map(r => {
    const value = r[metric] ?? 0;
    const width = (Math.abs(value) / maxAbs) * (zeroLeft === 0 ? 100 : 50);
    const left = zeroLeft === 0 ? 0 : (value >= 0 ? 50 : 50 - width);
    const cls = value >= 0 ? "pos" : "neg";
    return `<div class="bar-row">
      <div class="bar-name">${r.strategy}</div>
      <div class="bar-track">
        ${zeroLeft === 50 ? '<span class="bar-zero"></span>' : ''}
        <span class="bar ${cls}" style="left:${left}%; width:${width}%"></span>
      </div>
      <div class="metric">${metricText(metric, r[metric])}</div>
    </div>`;
  }).join("");

  const bestAlpha = [...rows].sort((a,b)=>(b.alpha ?? -Infinity) - (a.alpha ?? -Infinity))[0];
  const bestSharpe = [...rows].sort((a,b)=>(b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity))[0];
  const bestReturn = [...rows].sort((a,b)=>((b.total_return ?? b.basket ?? -Infinity) - (a.total_return ?? a.basket ?? -Infinity)))[0];
  $("cards").innerHTML = [
    ["Best Alpha", bestAlpha?.strategy, pct(bestAlpha?.alpha)],
    ["Best Sharpe", bestSharpe?.strategy, num(bestSharpe?.sharpe)],
    ["Best Return", bestReturn?.strategy, metricText(bestReturn?.family === "ranked" ? "total_return" : "basket", bestReturn?.total_return ?? bestReturn?.basket)]
  ].map(([title,name,val]) => `<div class="mini"><small>${title}</small><strong>${name || "-"}</strong><span>${val}</span></div>`).join("");

  $("tbody").innerHTML = rows.map(r => `<tr>
    <td>${r.strategy}</td>
    <td>${metricText(r.family === "ranked" ? "total_return" : "basket", r.total_return ?? r.basket)}</td>
    <td>${pct(r.alpha)}</td>
    <td>${num(r.sharpe)}</td>
    <td>${pct(r.benchmark_return)}</td>
    <td>${pct(r.equal_weight_alpha)}</td>
    <td>${pct(r.win_rate)}</td>
    <td>${pct(r.max_drawdown)}</td>
    <td>${r.family === "ranked" ? num(r.turnover) : intv(r.days_in)}</td>
  </tr>`).join("");
}
fetch("/api/results").then(r => r.json()).then(data => {
  ALL_ROWS = data.rows;
  fillSelect("dataset", Object.keys(DATASETS), Object.fromEntries(Object.entries(DATASETS).map(([k,v]) => [k, v.label])));
  refreshSelectors();
  ["dataset","market","years","metric","sort"].forEach(id => $(id).addEventListener("change", () => {
    if (id === "dataset") refreshSelectors();
    if (id === "metric") $("sort").value = $("metric").value;
    render();
  }));
  render();
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    dataset_map: dict[str, Path]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/results":
            payload = {"rows": []}
            for dataset, root in self.dataset_map.items():
                if root.exists():
                    payload["rows"].extend(load_dataset(dataset, root))
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if parsed.path == "/api/config":
            payload = {name: str(path) for name, path in self.dataset_map.items()}
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve strategy result dashboards.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public", action="store_true", help="Bind to 0.0.0.0 for outside-VM access.")
    parser.add_argument("--results-dir", default=None, help="Optional override: use one directory for broad_pine.")
    args = parser.parse_args()

    dataset_map = {name: Path(path) for name, path in DEFAULT_DATASETS.items()}
    if args.results_dir:
        dataset_map["broad_pine"] = Path(args.results_dir)
    Handler.dataset_map = dataset_map
    host = "0.0.0.0" if args.public else args.host

    httpd = ThreadingHTTPServer((host, args.port), Handler)
    print(f"Serving dashboard at http://{host}:{args.port}")
    for name, path in dataset_map.items():
        print(f"  {name}: {path}")
    if host == "0.0.0.0":
        print(f"External URL: http://<vm-ip>:{args.port}")
        print(f"Local URL:    http://127.0.0.1:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
