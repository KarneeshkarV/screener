#!/usr/bin/env python
"""Rebuild findings/pit_midsmall/index.html from results.csv + run JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pit_config_grid import build_index
from pit_strategy_analysis import analyze

OUT = Path("findings/pit_midsmall/index.html")
STRATEGY_OUT = Path("findings/pit_midsmall/strategy.html")
RUNS = Path("findings/pit_midsmall/runs")
CSV = Path("findings/pit_midsmall/results.csv")

LABEL = {
    "mid": "mid",
    "small": "small",
    "midsmall": "combined",
    "n50": "nifty 50",
    "n500": "nifty 500",
}
PICK = {
    "midsmall": "momentum_12_1_trend",
    "small": "momentum_12_1",
    "mid": "nifty_momentum",
    "n50": "momentum_12_1_trend",
    "n500": "momentum_12_1_trend",
}


def load() -> pd.DataFrame:
    rows = []
    for path in RUNS.glob("*.json"):
        payload = json.loads(path.read_text())
        if payload.get("error"):
            continue
        m = payload.get("metrics") or {}
        rows.append(
            {
                "universe": payload.get("universe"),
                "strategy": payload.get("strategy"),
                "family": payload.get("family"),
                "years": int(payload.get("years") or 0),
                "hold": payload.get("hold"),
                "n_trades": payload.get("n_trades") or 0,
                "sharpe": m.get("sharpe"),
                "cagr": m.get("cagr"),
                "max_drawdown": m.get("max_drawdown"),
                "hit_rate": m.get("hit_rate"),
                "exposure": m.get("exposure"),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (univ, strat), g in df.groupby(["universe", "strategy"]):
        by_y = {int(r.years): r for r in g.itertuples()}

        def get(y: int, col: str):
            r = by_y.get(y)
            return getattr(r, col) if r is not None else None

        recs.append(
            {
                "universe": univ,
                "strategy": strat,
                "family": g["family"].iloc[0] if "family" in g else "",
                "mean_s": g["sharpe"].mean(),
                "min_s": g["sharpe"].min(),
                "s1": get(1, "sharpe"),
                "s2": get(2, "sharpe"),
                "s3": get(3, "sharpe"),
                "s5": get(5, "sharpe"),
                "mean_cagr": g["cagr"].mean(),
                "cagr5": get(5, "cagr"),
                "worst_dd": g["max_drawdown"].min(),
                "dd5": get(5, "max_drawdown"),
                "t5": get(5, "n_trades") or 0,
                "any_zero": bool((g["n_trades"] == 0).any()),
                "exposure": g["exposure"].mean(skipna=True),
                "exp5": get(5, "exposure"),
            }
        )
    return pd.DataFrame(recs)


def fmt(n, d=2) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "-"
    return f"{float(n):.{d}f}"


def pct(n) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "-"
    return f"{float(n)*100:.1f}%"


def core_rows(sumdf: pd.DataFrame) -> list[dict]:
    out = []
    order = ["midsmall", "n500", "n50", "small", "mid"]
    for univ in order:
        block = sumdf[sumdf.universe == univ]
        if block.empty:
            continue
        inv = block[(~block.any_zero) & (block.t5 >= 20)]
        invested = inv[inv.exposure.fillna(0) >= 0.8].copy()
        pool = invested if len(invested) else inv
        if pool.empty:
            continue
        pool = pool.assign(s35=pool[["s3", "s5"]].mean(axis=1))
        pool = pool.sort_values("s35", ascending=False)
        top = pool.iloc[0]
        out.append(top.to_dict())
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if pd.isna(obj):
            return None
        return round(obj, 4)
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if pd.isna(obj):
        return None
    return obj


def build_html(sumdf: pd.DataFrame, df: pd.DataFrame, n_runs: int) -> str:
    rows_json = json.dumps(
        [
            {
                "universe": r.universe,
                "strategy": r.strategy,
                "family": r.family,
                "mean_s": None if pd.isna(r.mean_s) else round(float(r.mean_s), 4),
                "min_s": None if pd.isna(r.min_s) else round(float(r.min_s), 4),
                "s1": None if pd.isna(r.s1) else round(float(r.s1), 4),
                "s2": None if pd.isna(r.s2) else round(float(r.s2), 4),
                "s3": None if pd.isna(r.s3) else round(float(r.s3), 4),
                "s5": None if pd.isna(r.s5) else round(float(r.s5), 4),
                "mean_cagr": None if pd.isna(r.mean_cagr) else round(float(r.mean_cagr), 4),
                "worst_dd": None if pd.isna(r.worst_dd) else round(float(r.worst_dd), 4),
                "t5": int(r.t5) if pd.notna(r.t5) else 0,
                "any_zero": bool(r.any_zero),
                "exposure": None if pd.isna(r.exposure) else round(float(r.exposure), 4),
            }
            for r in sumdf.itertuples()
        ],
        separators=(",", ":"),
    )

    cores = core_rows(sumdf)
    core_html = []
    for r in cores:
        univ = r["universe"]
        pick = r["strategy"] == PICK.get(univ)
        cls = "pick" if pick else ""
        s35 = None
        if r.get("s3") is not None and r.get("s5") is not None:
            s35 = (float(r["s3"]) + float(r["s5"])) / 2
        core_html.append(
            "<tr class='{cls}'>"
            "<td class='l'>{lab}</td>"
            "<td class='l mono'>{strat}</td>"
            "<td class='mono pos'>{s35}</td>"
            "<td class='mono'>{cagr}</td>"
            "<td class='mono'>{dd}</td>"
            "<td class='mono'>{exp}</td>"
            "<td class='mono'>{t5}</td>"
            "</tr>".format(
                cls=cls,
                lab=LABEL.get(univ, univ),
                strat=r["strategy"],
                s35=fmt(s35),
                cagr=pct(r.get("cagr5")),
                dd=pct(r.get("dd5")),
                exp=pct(r.get("exp5") if r.get("exp5") is not None else r.get("exposure")),
                t5=int(r.get("t5") or 0),
            )
        )

    ready = {}
    for path in RUNS.glob("india__*__*__*y.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("equity_curve") and payload.get("trades") is not None:
            key = f"{payload.get('universe')}|{payload.get('strategy')}|{payload.get('years')}"
            ready[key] = True
    return (
        FULL_HTML.replace("__ROWS__", rows_json)
        .replace("__CORE__", "\n".join(core_html))
        .replace("__NRUNS__", str(n_runs))
        .replace("__NUNIV__", str(sumdf["universe"].nunique()))
        .replace("__NSTRAT__", str(sumdf["strategy"].nunique()))
        .replace("__READY__", json.dumps(ready, separators=(",", ":")))
        .replace("__ANALYSIS__", json.dumps(_jsonable(analyze(df)), separators=(",", ":")))
        .replace("__CFG__", json.dumps(_jsonable(build_index(OUT.parent)), separators=(",", ":")))
    )


FULL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>India PIT backtests</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&family=Young+Serif&display=swap" rel="stylesheet">
<style>
:root {
  --paper: #c8d0c4;
  --paper-2: #d7ddd2;
  --ink: #142033;
  --muted: #4a564e;
  --rule: #6e7a70;
  --stamp: #b13228;
  --invest: #1a5c42;
  --cash: #8a5414;
  --bad: #8b2430;
  --card: #e6ebe1;
  --shadow: 0 1px 0 rgba(20,32,51,.08);
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--paper); color: var(--ink); }
body {
  font-family: "Atkinson Hyperlegible", sans-serif;
  font-size: 17px;
  line-height: 1.45;
  min-height: 100vh;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: .09;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  z-index: 0;
}
.wrap { position: relative; z-index: 1; max-width: 1180px; margin: 0 auto; padding: 0 1.25rem 4rem; }
.ruler {
  display: grid;
  grid-template-columns: 13.33% 20% 33.33% 33.34%;
  height: 2.35rem;
  border-bottom: 2px solid var(--ink);
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.ruler span { display: flex; align-items: flex-end; padding: .2rem .45rem; border-right: 1px solid rgba(20,32,51,.25); }
.r-lg { background: #6b8f7a; color: #0d1a12; }
.r-mid { background: #7aa3b5; color: #102028; }
.r-sm { background: #c4a15a; color: #2a220e; }
.r-mc { background: #b7bfb4; color: #5a635c; }
.mast { display: grid; grid-template-columns: 1fr auto; gap: 1.5rem; padding: 1.6rem 0 1.1rem; border-bottom: 1px solid var(--rule); }
h1 { font-family: "Young Serif", serif; font-weight: 400; font-size: clamp(2rem, 5vw, 3.15rem); line-height: 1.05; margin: 0 0 .4rem; letter-spacing: -.02em; }
.sub { color: var(--muted); max-width: 40rem; }
.meta { font-family: "IBM Plex Mono", monospace; font-size: 11px; line-height: 1.6; text-align: right; color: var(--muted); }
.meta b { color: var(--ink); font-weight: 600; }
.thesis { margin: 1.6rem 0 1.4rem; padding: 1.15rem 1.2rem 1.2rem; background: var(--card); border-left: 6px solid var(--stamp); box-shadow: var(--shadow); }
.thesis .kicker { font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--stamp); margin: 0 0 .35rem; }
.thesis h2 { font-family: "Young Serif", serif; font-size: clamp(1.45rem, 3vw, 2rem); font-weight: 400; margin: 0 0 .55rem; line-height: 1.15; }
.thesis p { margin: 0 0 .45rem; }
.thesis p:last-child { margin: 0; }
h3 { font-family: "Young Serif", serif; font-weight: 400; font-size: 1.35rem; margin: 2rem 0 .7rem; }
p.note { color: var(--muted); margin: .2rem 0 0.8rem; }
.table-wrap { overflow-x: auto; background: var(--card); border: 1px solid rgba(20,32,51,.12); }
table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
th, td { padding: .48rem .55rem; text-align: right; border-bottom: 1px solid rgba(20,32,51,.1); white-space: nowrap; }
th { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
td:first-child, th:first-child, td.l, th.l { text-align: left; }
td.mono, .mono { font-family: "IBM Plex Mono", monospace; font-size: 13px; }
tr.pick td { background: #dce8d8; }
tr.cash td { background: #efe4cc; }
.pos { color: var(--invest); }
.neg { color: var(--bad); }
.grid-3 { display: grid; grid-template-columns: repeat(5, 1fr); gap: .55rem; margin: 1rem 0; }
.card { background: var(--card); padding: .85rem .9rem; border: 1px solid rgba(20,32,51,.12); }
.card .lab { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.card .num { font-family: "IBM Plex Mono", monospace; font-size: 1.35rem; margin: .15rem 0; }
.card p { margin: 0; color: var(--muted); font-size: 13.5px; }
.controls { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin: .6rem 0 .7rem; }
.btn { font-family: "IBM Plex Mono", monospace; font-size: 12px; border: 1px solid var(--ink); background: transparent; color: var(--ink); padding: .32rem .6rem; cursor: pointer; }
.btn[aria-pressed="true"] { background: var(--ink); color: var(--paper-2); }
.btn:focus-visible { outline: 2px solid var(--stamp); outline-offset: 2px; }
.check { font-size: 14.5px; color: var(--muted); display: flex; gap: .35rem; align-items: center; margin-left: .4rem; }
#count, #an-count { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--muted); margin-left: auto; }
.avoid { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.avoid ul { margin: .35rem 0 0; padding-left: 1.1rem; }
.windows { display: grid; grid-template-columns: repeat(4, 1fr); gap: .55rem; }
.win { background: var(--card); padding: .7rem .75rem; border-top: 3px solid var(--ink); }
.win.down { border-top-color: var(--bad); }
.win.up { border-top-color: var(--invest); }
.ledger { margin-top: 2.2rem; }
#chart-wrap { background: var(--card); border: 1px solid rgba(20,32,51,.12); padding: .6rem .7rem .3rem; }
#chart { width: 100%; height: 280px; display: block; }
.legend { display: flex; gap: 1rem; font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--muted); padding: .2rem 0 .5rem; }
.legend i { display: inline-block; width: 1.4rem; height: 3px; margin-right: .35rem; vertical-align: middle; }
.leg-eq { background: #1a5c42; }
.leg-bm { background: #b13228; }
#ledger-note { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--muted); margin: .4rem 0; }
select.pick { font-family: "IBM Plex Mono", monospace; font-size: 12px; padding: .28rem .4rem; border: 1px solid var(--ink); background: var(--card); color: var(--ink); max-width: 22rem; }
tr.win td { }
tr.go { cursor: pointer; }
tr.go:hover td { background: #e8ddd0; }
tr.pick.go:hover td { background: #cfdcc9; }
tr.tv { cursor: pointer; }
tr.tv:hover td { background: #e8ddd0; }
tr.tv.pick:hover td { background: #cfdcc9; }
a.tv-link { color: inherit; text-decoration: underline; text-underline-offset: 2px; }
#analysis { margin-top: 2.4rem; }
.role-bar { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: .5rem 0 .8rem; }
.role-bar input[type="search"] {
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  padding: .32rem .5rem;
  border: 1px solid var(--ink);
  background: var(--card);
  color: var(--ink);
  min-width: 12rem;
}
.stamp {
  display: inline-block;
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  letter-spacing: .1em;
  text-transform: uppercase;
  padding: .12rem .4rem;
  border: 1px solid currentColor;
  line-height: 1.3;
}
.stamp.core { color: #0d2a1c; background: #9cbc9a; border-color: #1a5c42; }
.stamp.overlay { color: #3a2408; background: #e0c48a; border-color: #8a5414; }
.stamp.ok { color: var(--ink); background: var(--card); }
.stamp.avoid { color: #3a1014; background: #d9a4a8; border-color: var(--stamp); }
.stamp.broken { color: #4a564e; background: #c5cbc0; }
.stamp.thin { color: #3a3220; background: #d5cbb6; }
.scard {
  background: var(--card);
  border: 1px solid rgba(20,32,51,.12);
  margin: 0 0 .55rem;
}
.scard > summary {
  list-style: none;
  cursor: pointer;
  display: grid;
  grid-template-columns: 5.6rem 1fr;
  gap: .55rem .7rem;
  align-items: start;
  padding: .7rem .85rem;
}
.scard > summary::-webkit-details-marker { display: none; }
.scard[open] > summary { border-bottom: 1px solid rgba(20,32,51,.1); }
.scard .sname { font-family: "IBM Plex Mono", monospace; font-size: 14px; font-weight: 600; }
.scard .smeta { font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--muted); margin-top: .15rem; }
.scard .sline { color: var(--muted); font-size: 15px; margin-top: .28rem; }
.scard .body { padding: .75rem .85rem 1rem; }
.scard .body p { margin: 0 0 .55rem; }
.scard .body p:last-child { margin-bottom: .8rem; }
.mini { width: 100%; }
.mini th, .mini td { font-size: 12.5px; padding: .32rem .4rem; }
.jump { font-family: "IBM Plex Mono", monospace; font-size: 11px; border: 1px solid var(--ink); background: transparent; color: var(--ink); padding: .2rem .45rem; cursor: pointer; text-decoration: none; display: inline-block; } 
footer { margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted); font-size: 14px; }
@media (max-width: 900px) {
  .mast, .avoid, .windows, .grid-3 { grid-template-columns: 1fr 1fr; }
  .meta { text-align: left; }
}
@media (max-width: 640px) {
  .mast, .avoid, .windows, .grid-3 { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="ruler" aria-label="NSE rank bands. Nifty 50 sits inside 1-100. Nifty 500 is 1-500.">
  <span class="r-lg">1-100 Nifty 50 / 100</span>
  <span class="r-mid">101-250 mid</span>
  <span class="r-sm">251-500 small</span>
  <span class="r-mc">501-750 micro (out)</span>
</div>
<div class="wrap">
  <header class="mast">
    <div>
      <h1>Do not pick the highest Sharpe.</h1>
      <p class="sub">India point-in-time membership across Nifty 50, Nifty 500, mid, small, and mid+small. Same 1/2/3/5y windows. Fully invested 12-1 momentum is the core book.</p>
    </div>
    <div class="meta">
      <div><b>16 Aug 2026</b></div>
      <div>windows 5y / 3y / 2y / 1y</div>
      <div>end 2026-08-16</div>
      <div>top 10 · India costs + 10 bps</div>
      <div>__NRUNS__ runs · __NUNIV__ universes · __NSTRAT__ strategies</div>
    </div>
  </header>

  <section class="thesis">
    <p class="kicker">Decision</p>
    <h2>Trade <span class="mono">momentum_12_1_trend</span> on combined mid+small.</h2>
    <p>Use PIT ranks 101-500. Hold 63 sessions. Top 10. India statutory costs.</p>
    <p>Nifty 50 and Nifty 500 use the same windows and the same strategy set. Use them as confirmation, not as a replacement, unless the table below beats combined on both Sharpe and drawdown while staying invested.</p>
    <p>Quality names with high Sharpe often sit in cash. That is an overlay, not a book.</p>
  </section>

  <h3>Best fully invested name per universe</h3>
  <p class="note">3y+5y mean Sharpe. Exposure at least 80% where possible. 5y trades at least 20.</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th class="l">Universe</th><th class="l">Strategy</th>
          <th>3y+5y Sharpe</th><th>5y CAGR</th><th>5y max DD</th><th>5y exp</th><th>5y trades</th>
        </tr>
      </thead>
      <tbody>
__CORE__
      </tbody>
    </table>
  </div>

  <h3>Nifty 50 over the same windows</h3>
  <div class="windows">
    <div class="win down"><div class="lab">1y from 2025-08-16</div><div class="num mono">-2.1%</div></div>
    <div class="win down"><div class="lab">2y from 2024-08-16</div><div class="num mono">-0.7%</div></div>
    <div class="win up"><div class="lab">3y from 2023-08-17</div><div class="num mono">+25.8%</div></div>
    <div class="win up"><div class="lab">5y from 2021-08-17</div><div class="num mono">+46.7%</div></div>
  </div>
  <p class="note">Benchmark is Nifty 50 (^NSEI). 1y and 2y are flat. A name that is strong only in 3y/5y is riding the 2023-2024 bull.</p>

  <h3>PIT universes</h3>
  <div class="grid-3">
    <div class="card"><div class="lab">Nifty 50</div><div class="num mono">50</div><p>archived lists · 10 dates · 108 unique</p></div>
    <div class="card"><div class="lab">Nifty 500</div><div class="num mono">500</div><p>archived lists · 9 dates · 713 unique</p></div>
    <div class="card"><div class="lab">Mid 101-250</div><div class="num mono">150</div><p>cap ranks · 23 dates · 397 unique</p></div>
    <div class="card"><div class="lab">Small 251-500</div><div class="num mono">211-250</div><p>cap ranks · 23 dates · 678 unique</p></div>
    <div class="card"><div class="lab">Combined 101-500</div><div class="num mono">361-400</div><p>cap ranks · 23 dates · 820 unique</p></div>
  </div>

  <h3>All strategies</h3>
  <p class="note">Mean / 1y / 2y / 3y / 5y are the base book (no stop, no take-profit). Best columns are the top hold / stop / take-profit gate on that same window. Click a name for every config and every window. <span id="cfg-status"></span></p>
  <div class="controls">
    <button class="btn" data-u="midsmall" aria-pressed="true">combined</button>
    <button class="btn" data-u="n500" aria-pressed="false">nifty 500</button>
    <button class="btn" data-u="n50" aria-pressed="false">nifty 50</button>
    <button class="btn" data-u="mid" aria-pressed="false">mid</button>
    <button class="btn" data-u="small" aria-pressed="false">small</button>
    <label class="check"><input type="checkbox" id="invested"> fully invested only</label>
    <label class="check"><input type="checkbox" id="robust" checked> robust filter</label>
    <span id="count"></span>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th class="l">Strategy</th>
          <th>Mean</th><th>Min</th>
          <th>1y</th><th>2y</th><th>3y</th><th>5y</th>
          <th>CAGR</th><th>Worst DD</th><th>Exp</th><th>5y n</th>
          <th>Best 5y</th><th>Best 3y</th><th>Best 1y</th><th class="l">Best cfg</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <section id="analysis">
    <h3>Strategy analysis</h3>
    <p class="note">One card per name. Role uses mid / small / combined first. Nifty 50 and 500 are confirmation. <b>core</b> can be a book. <b>overlay</b> sits in cash. <b>ok</b> is invested but weaker. <b>avoid</b> does not pay for the risk. <b>broken</b> has no trades. Open a card for the table. Click the name to load trades.</p>
    <div class="role-bar">
      <button class="btn role-f" data-role="all" aria-pressed="true">all</button>
      <button class="btn role-f" data-role="core" aria-pressed="false">core</button>
      <button class="btn role-f" data-role="overlay" aria-pressed="false">overlay</button>
      <button class="btn role-f" data-role="ok" aria-pressed="false">ok</button>
      <button class="btn role-f" data-role="thin" aria-pressed="false">thin</button>
      <button class="btn role-f" data-role="avoid" aria-pressed="false">avoid</button>
      <button class="btn role-f" data-role="broken" aria-pressed="false">broken</button>
      <input type="search" id="an-q" placeholder="filter name" autocomplete="off">
      <span id="an-count"></span>
    </div>
    <div id="an-list"></div>
  </section>

  <section class="ledger" id="ledger">
    <h3>Trades and equity vs benchmark</h3>
    <p class="note">Pick a universe, a strategy, and a window. Names marked pending do not have a saved ledger yet. The chart is both series rebased to 100. Green row is a winning trade. Click a trade to open the NSE daily chart on TradingView.</p>
    <div class="controls">
      <button class="btn led-u" data-lu="midsmall" aria-pressed="false">combined</button>
      <button class="btn led-u" data-lu="n500" aria-pressed="false">nifty 500</button>
      <button class="btn led-u" data-lu="n50" aria-pressed="false">nifty 50</button>
      <button class="btn led-u" data-lu="mid" aria-pressed="false">mid</button>
      <button class="btn led-u" data-lu="small" aria-pressed="true">small</button>
    </div>
    <div class="controls">
      <select class="pick" id="led-strat"></select>
      <button class="btn led-y" data-ly="5" aria-pressed="true">5y</button>
      <button class="btn led-y" data-ly="3" aria-pressed="false">3y</button>
      <button class="btn led-y" data-ly="2" aria-pressed="false">2y</button>
      <button class="btn led-y" data-ly="1" aria-pressed="false">1y</button>
    </div>
    <p id="ledger-note"></p>
    <div id="chart-wrap">
      <div class="legend"><span><i class="leg-eq"></i>strategy</span><span><i class="leg-bm"></i>Nifty 50</span></div>
      <canvas id="chart" width="1100" height="280"></canvas>
    </div>
    <div class="table-wrap" style="margin-top:.7rem">
      <table>
        <thead>
          <tr>
            <th class="l">Ticker</th>
            <th class="l">Entry</th><th>Entry px</th>
            <th class="l">Exit</th><th>Exit px</th>
            <th>Shares</th><th>PnL</th><th>Return</th>
            <th class="l">Exit reason</th>
          </tr>
        </thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
  </section>

  <footer>
    Research, not financial advice. Prices: yfinance. Costs: India STT/stamp + 10 bps.
    Nifty 50/500 PIT from archived NSE constituent lists, first snapshot backfilled to 2021-01-01.
    Mid/small PIT from reconstructed market-cap rank bands.
    Live FMP is 401: Sloan / Piotroski / FCF yield have 0 trades.
  </footer>
</div>
<script>
const ROWS = __ROWS__;
const READY = __READY__;
const ANALYSIS = __ANALYSIS__;
let CFG = __CFG__;
const ULAB = {midsmall:"combined", n500:"nifty 500", n50:"nifty 50", mid:"mid", small:"small"};
function cfgCell(univ, strat) {
  const cell = CFG && CFG.by && CFG.by[strat] && CFG.by[strat][univ];
  return cell || null;
}
function winCell(univ, strat, y) {
  const slot = cfgCell(univ, strat);
  if (!slot) return null;
  if (slot.win && slot.win[String(y)]) return slot.win[String(y)];
  return Number(y) === 5 ? slot : null;
}
let univ = "midsmall";
const PICK = {midsmall:"momentum_12_1_trend", small:"momentum_12_1", mid:"nifty_momentum", n50:"momentum_12_1_trend", n500:"momentum_12_1_trend"};
function fmt(n, d=2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toFixed(d);
}
function pct(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return (n * 100).toFixed(1) + "%";
}
function cls(n) {
  if (n === null || n === undefined) return "mono";
  return "mono " + (n > 0 ? "pos" : n < 0 ? "neg" : "");
}
function toTv(ticker) {
  const raw = String(ticker || "").trim();
  const m = raw.match(/^(.+)\.(NS|BO)$/i);
  const exch = m ? (m[2].toUpperCase() === "BO" ? "BSE" : "NSE") : "NSE";
  const name = (m ? m[1] : raw).replace(/[&-]/g, "_");
  return exch + ":" + name;
}
function tvUrl(ticker) {
  return "https://in.tradingview.com/chart/?symbol=" + encodeURIComponent(toTv(ticker)) + "&interval=D";
}
function render() {
  const invested = document.getElementById("invested").checked;
  const robust = document.getElementById("robust").checked;
  let rows = ROWS.filter(r => r.universe === univ);
  if (robust) rows = rows.filter(r => !r.any_zero && r.t5 >= 20);
  if (invested) rows = rows.filter(r => r.exposure !== null && r.exposure >= 0.8);
  rows.sort((a,b) => (b.mean_s||0) - (a.mean_s||0));
  document.getElementById("count").textContent = rows.length + " names";
  const st = document.getElementById("cfg-status");
  if (st && CFG) st.textContent = "Config sweep: " + (CFG.done||0) + " / " + (CFG.expected||"?") + " cells (5/3/2/1y).";
  const pick = PICK[univ];
  document.getElementById("tbody").innerHTML = rows.map(r => {
    const cash = r.exposure !== null && r.exposure < 0.5;
    const hi = r.strategy === pick ? "pick" : cash ? "cash" : "";
    const cell = cfgCell(univ, r.strategy);
    const w5 = winCell(univ, r.strategy, 5);
    const w3 = winCell(univ, r.strategy, 3);
    const w1 = winCell(univ, r.strategy, 1);
    const best = (w5 && w5.best) || (cell && cell.best);
    const baseIsBest = best && w5 && w5.base && best.id === w5.base.id;
    const href = "strategy.html?s=" + encodeURIComponent(r.strategy) + "&u=" + univ + "&y=5";
    return `<tr class="${hi} go" data-href="${href}">
      <td class="l mono"><a class="tv-link" href="${href}">${r.strategy}</a></td>
      <td class="${cls(r.mean_s)}">${fmt(r.mean_s)}</td>
      <td class="${cls(r.min_s)}">${fmt(r.min_s)}</td>
      <td class="${cls(r.s1)}">${fmt(r.s1)}</td>
      <td class="${cls(r.s2)}">${fmt(r.s2)}</td>
      <td class="${cls(r.s3)}">${fmt(r.s3)}</td>
      <td class="${cls(r.s5)}">${fmt(r.s5)}</td>
      <td class="mono">${pct(r.mean_cagr)}</td>
      <td class="mono">${pct(r.worst_dd)}</td>
      <td class="mono">${r.exposure==null ? "-" : pct(r.exposure)}</td>
      <td class="mono">${r.t5 ?? "-"}</td>
      <td class="${cls(best && best.sharpe)}">${best ? fmt(best.sharpe) : "…"}</td>
      <td class="${cls(w3 && w3.best && w3.best.sharpe)}">${w3 && w3.best ? fmt(w3.best.sharpe) : "…"}</td>
      <td class="${cls(w1 && w1.best && w1.best.sharpe)}">${w1 && w1.best ? fmt(w1.best.sharpe) : "…"}</td>
      <td class="l mono">${best ? (baseIsBest ? "base" : (best.label||best.tag)) : (cell ? (cell.n||0)+"/"+(cell.expected||"?") : "…")}</td>
    </tr>`;
  }).join("");
}
document.querySelectorAll(".btn[data-u]").forEach(btn => {
  btn.addEventListener("click", () => {
    univ = btn.dataset.u;
    document.querySelectorAll(".btn[data-u]").forEach(b => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
    render();
  });
});
document.getElementById("invested").addEventListener("change", render);
document.getElementById("robust").addEventListener("change", render);
document.getElementById("tbody").addEventListener("click", ev => {
  if (ev.target.closest("a")) return;
  const tr = ev.target.closest("tr[data-href]");
  if (!tr) return;
  location.href = tr.dataset.href;
});
fetch("configs/index.json", { cache: "no-store" }).then(r => r.ok ? r.json() : null).then(d => {
  if (d && d.by) { CFG = d; render(); }
}).catch(() => {});
render();

let ledU = "small";
let ledY = 5;
const stratSel = document.getElementById("led-strat");
function isReady(u, s, y) { return !!READY[u + "|" + s + "|" + y]; }
function fillStrats() {
  const names = [...new Set(ROWS.filter(r => r.universe === ledU).map(r => r.strategy))].sort();
  const keep = stratSel.value;
  stratSel.innerHTML = names.map(n => {
    const ok = [5,3,2,1].some(y => isReady(ledU, n, y));
    return `<option value="${n}">${n}${ok ? "" : "  pending"}</option>`;
  }).join("");
  if (names.includes(keep) && isReady(ledU, keep, ledY)) {
    stratSel.value = keep;
    return;
  }
  const first = names.find(n => isReady(ledU, n, ledY)) || names.find(n => [5,3,2,1].some(y => isReady(ledU, n, y))) || names[0];
  if (first) stratSel.value = first;
}
function drawChart(eq, bm) {
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!eq || eq.length < 2) {
    ctx.fillStyle = "#4a564e";
    ctx.font = "13px IBM Plex Mono, monospace";
    ctx.fillText("No equity curve in this run file yet.", 16, 40);
    return;
  }
  const rebase = (arr) => {
    const z = arr[0].v;
    if (!z) return arr.map(p => ({d:p.d, v:100}));
    return arr.map(p => ({d:p.d, v: 100 * p.v / z}));
  };
  const e = rebase(eq);
  const b = bm && bm.length ? rebase(bm) : [];
  const vals = e.map(p => p.v).concat(b.map(p => p.v));
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const pad = 28;
  const x = i => pad + (w - pad - 10) * i / (e.length - 1);
  const y = v => h - pad - (h - pad - 16) * ((v - mn) / (mx - mn || 1));
  ctx.strokeStyle = "#8a9488";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, 10); ctx.lineTo(pad, h-pad); ctx.lineTo(w-8, h-pad);
  ctx.stroke();
  ctx.fillStyle = "#4a564e";
  ctx.font = "11px IBM Plex Mono, monospace";
  ctx.fillText(mn.toFixed(0), 4, h-pad);
  ctx.fillText(mx.toFixed(0), 4, 16);
  ctx.fillText(e[0].d, pad, h-8);
  ctx.fillText(e[e.length-1].d, w-88, h-8);
  const line = (arr, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    arr.forEach((p,i) => { const X=x(i), Y=y(p.v); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
    ctx.stroke();
  };
  if (b.length) line(b, "#b13228");
  line(e, "#1a5c42");
}
async function loadLedger() {
  const strat = stratSel.value;
  const note = document.getElementById("ledger-note");
  const tb = document.getElementById("trade-body");
  if (!strat) { note.textContent = "No strategy."; return; }
  const path = `runs/india__${ledU}__${strat}__${ledY}y.json?v=3`;
  note.textContent = "loading " + path;
  tb.innerHTML = "";
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error(res.status + " " + path);
    const raw = await res.text();
    const d = JSON.parse(raw.replace(/\b-?Infinity\b/g, "null").replace(/\bNaN\b/g, "null"));
    drawChart(d.equity_curve, d.benchmark_curve);
    const trades = d.trades || [];
    const m = d.metrics || {};
    const nKnown = trades.length || d.n_trades || 0;
    note.textContent = `${d.universe} · ${d.strategy} · ${d.years}y · ${d.start} to ${d.end} · ${nKnown} trades · Sharpe ${Number(m.sharpe||0).toFixed(2)} · CAGR ${(100*(m.cagr||0)).toFixed(1)}%`;
    if (!d.equity_curve || !d.equity_curve.length) {
      tb.innerHTML = `<tr><td class="l" colspan="9">${nKnown} trades exist in the summary. The entry/exit ledger and the chart are still being written. Refresh in a few minutes.</td></tr>`;
      return;
    }
    if (!trades.length) {
      tb.innerHTML = `<tr><td class="l" colspan="9">This run has no completed trades.</td></tr>`;
      return;
    }
    tb.innerHTML = trades.map(t => {
      const win = t.pnl > 0;
      const tv = toTv(t.ticker);
      const href = tvUrl(t.ticker);
      const ed = String(t.entry_date).slice(0,10);
      const xd = String(t.exit_date).slice(0,10);
      return `<tr class="tv${win ? " pick" : ""}" data-href="${href}" title="Open ${tv} daily · ${ed} to ${xd}">
        <td class="l mono"><a class="tv-link" href="${href}" target="_blank" rel="noopener">${t.ticker}</a></td>
        <td class="l mono">${ed}</td>
        <td class="mono">${Number(t.entry_price).toFixed(2)}</td>
        <td class="l mono">${xd}</td>
        <td class="mono">${Number(t.exit_price).toFixed(2)}</td>
        <td class="mono">${Number(t.shares).toFixed(1)}</td>
        <td class="mono ${win?"pos":"neg"}">${Number(t.pnl).toFixed(0)}</td>
        <td class="mono ${win?"pos":"neg"}">${(100*t.return_pct).toFixed(1)}%</td>
        <td class="l">${t.exit_reason||""}</td>
      </tr>`;
    }).join("");
  } catch (err) {
    note.textContent = String(err);
    drawChart([], []);
  }
}
document.querySelectorAll(".led-u").forEach(btn => {
  btn.addEventListener("click", () => {
    ledU = btn.dataset.lu;
    document.querySelectorAll(".led-u").forEach(b => b.setAttribute("aria-pressed", b===btn?"true":"false"));
    fillStrats();
    loadLedger();
  });
});
document.querySelectorAll(".led-y").forEach(btn => {
  btn.addEventListener("click", () => {
    ledY = Number(btn.dataset.ly);
    document.querySelectorAll(".led-y").forEach(b => b.setAttribute("aria-pressed", b===btn?"true":"false"));
    loadLedger();
  });
});
stratSel.addEventListener("change", loadLedger);
document.getElementById("trade-body").addEventListener("click", ev => {
  if (ev.target.closest("a")) return;
  const tr = ev.target.closest("tr[data-href]");
  if (!tr) return;
  window.open(tr.dataset.href, "_blank", "noopener");
});
function openLedger(univ, strat, years) {
  ledU = univ;
  ledY = years || 5;
  document.querySelectorAll(".led-u").forEach(b => b.setAttribute("aria-pressed", b.dataset.lu===ledU?"true":"false"));
  document.querySelectorAll(".led-y").forEach(b => b.setAttribute("aria-pressed", Number(b.dataset.ly)===ledY?"true":"false"));
  fillStrats();
  if (strat) stratSel.value = strat;
  document.getElementById("ledger").scrollIntoView({behavior:"smooth"});
  loadLedger();
}
function renderAnalysis() {
  const role = document.querySelector(".role-f[aria-pressed='true']")?.dataset.role || "all";
  const q = (document.getElementById("an-q").value || "").trim().toLowerCase();
  const box = document.getElementById("an-list");
  const shown = ANALYSIS.filter(a => {
    if (role !== "all" && a.role !== role) return false;
    if (q && !a.name.includes(q) && !(a.what||"").toLowerCase().includes(q) && !(a.verdict||"").toLowerCase().includes(q)) return false;
    return true;
  });
  document.getElementById("an-count").textContent = shown.length + " / " + ANALYSIS.length;
  const order = ["midsmall","n500","n50","mid","small"];
  box.innerHTML = shown.map(a => {
    const best = a.best || {};
    const bestLab = ULAB[best.universe] || best.universe || "-";
    const rows = order.filter(u => a.by && a.by[u]).map(u => {
      const c = a.by[u];
      const mark = best.universe === u ? " class='pick'" : "";
      return `<tr${mark}>
        <td class="l">${ULAB[u]}</td>
        <td class="mono">${fmt(c.mean)}</td>
        <td class="${cls(c.s1)}">${fmt(c.s1)}</td>
        <td class="${cls(c.s2)}">${fmt(c.s2)}</td>
        <td class="${cls(c.s3)}">${fmt(c.s3)}</td>
        <td class="${cls(c.s5)}">${fmt(c.s5)}</td>
        <td class="mono">${pct(c.cagr5)}</td>
        <td class="mono">${pct(c.dd5)}</td>
        <td class="mono">${pct(c.exp5)}</td>
        <td class="mono">${c.t5||0}</td>
      </tr>`;
    }).join("");
    const open = a.role === "core" && !a.clone ? " open" : "";
    const target = best.universe || "midsmall";
    return `<details class="scard" data-role="${a.role}"${open}>
      <summary>
        <span class="stamp ${a.role}">${a.role}</span>
        <div>
          <div class="sname"><a class="tv-link" href="strategy.html?s=${encodeURIComponent(a.name)}&u=${target}&y=5">${a.name}</a></div>
          <div class="smeta">${a.family||""} · hold ${a.hold} · best ${bestLab} · 3y+5y ${fmt(best.s35)} · 5y ${pct(best.cagr5)} · exp ${pct(best.exp5)}</div>
          <div class="sline">${a.what||""}</div>
        </div>
      </summary>
      <div class="body">
        <p>${a.verdict||""}</p>
        <button class="jump" data-jump="${target}" data-strat="${a.name}">open trades on ${bestLab} 5y</button>
        <a class="jump" href="strategy.html?s=${encodeURIComponent(a.name)}&u=${target}&y=5">all configs</a>
        <div class="table-wrap" style="margin-top:.7rem">
          <table class="mini">
            <thead><tr>
              <th class="l">Universe</th><th>Mean</th>
              <th>1y</th><th>2y</th><th>3y</th><th>5y</th>
              <th>5y CAGR</th><th>5y DD</th><th>Exp</th><th>5y n</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </details>`;
  }).join("");
}
document.querySelectorAll(".role-f").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".role-f").forEach(b => b.setAttribute("aria-pressed", b===btn?"true":"false"));
    renderAnalysis();
  });
});
document.getElementById("an-q").addEventListener("input", renderAnalysis);
document.getElementById("an-list").addEventListener("click", ev => {
  const btn = ev.target.closest("[data-jump]");
  if (!btn) return;
  ev.preventDefault();
  ev.stopPropagation();
  openLedger(btn.dataset.jump, btn.dataset.strat, 5);
});
renderAnalysis();
fillStrats();
loadLedger();
</script>
</body>
</html>
"""


def main() -> int:
    df = load()
    if df.empty:
        print("no runs")
        return 1
    sumdf = summarize(df)
    html = build_html(sumdf, df, n_runs=len(df))
    OUT.write_text(html)
    analysis = analyze(df)
    (OUT.parent / "analysis.json").write_text(
        json.dumps(_jsonable(analysis), separators=(",", ":"), allow_nan=False)
    )
    print(
        f"wrote {OUT} runs={len(df)} cells={len(sumdf)} "
        f"universes={sorted(sumdf.universe.unique())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
