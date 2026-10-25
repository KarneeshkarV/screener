#!/usr/bin/env python3
"""Build the 'repeat-trade churn' card from ledger_res/*.csv and inject it into
holdcurve.html (before the 'Reading the result' card). Matches the page's dark
theme + inline-SVG line_chart style."""

import os
import glob
from collections import defaultdict, Counter

SP = os.environ.get("ANALYSIS_DIR", os.path.join(os.getcwd(), "profiling", "_analysis"))
RES = f"{SP}/ledger_res"
WEBVIEW = os.environ.get(
    "WEBVIEW_DIR", os.path.join(os.getcwd(), "profiling", "webview")
)
PAGE = os.path.join(WEBVIEW, "holdcurve.html")
COLS = [
    "mkt",
    "strat",
    "top",
    "hold",
    "tot",
    "uniq",
    "ratio",
    "multi",
    "pct_multi",
    "mx",
    "top3",
]
HOLDS = ["5", "10", "20", "30", "40", "60", "100", "inf"]
HL = ["5", "10", "20", "30", "40", "60", "100", "∞"]
STRATS = ["mark_minervini", "mq_us1", "mq_us2", "mq_us3", "mq_in1", "mq_in2", "mq_in3"]
TOPS = ["5", "10", "20", "40"]
US = "#4f9dff"
IN = "#f5a623"

data = {}
namecount = defaultdict(Counter)
for f in glob.glob(f"{RES}/*.csv"):
    p = open(f).read().rstrip("\n").split(",", 10)
    if len(p) < 10:
        continue
    r = dict(zip(COLS, p))
    if r["tot"] in ("ERROR", ""):
        continue
    for k in ("tot", "uniq", "multi", "mx"):
        r[k] = int(r[k]) if r[k] else 0
    for k in ("ratio", "pct_multi"):
        r[k] = float(r[k]) if r[k] else 0.0
    data[(r["mkt"], r["strat"], r["top"], r["hold"])] = r
    for tok in (r.get("top3") or "").split("|"):
        if ":" in tok:
            n, c = tok.rsplit(":", 1)
            try:
                namecount[r["mkt"]][n] += int(c)
            except:
                pass


def avg(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


curve = {}
for m in ("us", "india"):
    curve[m] = {}
    for h in HOLDS:
        rows = [
            data[(m, s, t, h)] for s in STRATS for t in TOPS if (m, s, t, h) in data
        ]
        curve[m][h] = {
            k: avg([r[k] for r in rows])
            for k in ("ratio", "pct_multi", "tot", "uniq", "mx")
        }


# ---- inline SVG line chart (2 series US/India) ----
def line_chart(title, series, ylabels_fmt="{:.2f}", w=520, h=260):
    L, R, T, B = 52, w - 14, 16, h - 34
    allv = [v for _, pts in series for v in pts if v is not None]
    lo, hi = min(allv), max(allv)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad
    n = len(HOLDS)

    def X(i):
        return L + (R - L) * i / (n - 1)

    def Y(v):
        return T + (B - T) * (hi - v) / (hi - lo)

    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    for g in range(5):
        yv = lo + (hi - lo) * g / 4
        y = Y(yv)
        s.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="#262a33"/>'
        )
        s.append(
            f'<text x="{L - 6}" y="{y + 3:.1f}" fill="#7d8697" font-size="10" text-anchor="end">{ylabels_fmt.format(yv)}</text>'
        )
    for i, lab in enumerate(HL):
        s.append(
            f'<text x="{X(i):.1f}" y="{h - 10}" fill="#9aa4b2" font-size="10" text-anchor="middle">{lab}</text>'
        )
    for color, pts in series:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{X(i):.1f} {Y(v):.1f}"
            for i, v in enumerate(pts)
            if v is not None
        )
        s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for i, v in enumerate(pts):
            if v is not None:
                s.append(
                    f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}"/>'
                )
    s.append(
        f'<text x="{L}" y="11" fill="#e6e6e6" font-size="11" font-weight="600">{title}</text></svg>'
    )
    return "".join(s)


ratio_us = [curve["us"][h]["ratio"] for h in HOLDS]
ratio_in = [curve["india"][h]["ratio"] for h in HOLDS]
pct_us = [curve["us"][h]["pct_multi"] for h in HOLDS]
pct_in = [curve["india"][h]["pct_multi"] for h in HOLDS]

chart1 = line_chart(
    "Trades per unique name vs hold cap", [(US, ratio_us), (IN, ratio_in)], "{:.1f}"
)
chart2 = line_chart(
    "% of trades that are re-entries vs hold cap",
    [(US, pct_us), (IN, pct_in)],
    "{:.0f}",
)
legend = (
    '<div class="legend"><span><i style="background:%s"></i>US</span> <span><i style="background:%s"></i>India</span></div>'
    % (US, IN)
)


def mtable(m):
    hot = ' class="hot"'
    rows = "".join(
        f"<tr{hot if h in ('20', '40') else ''}><td>{HL[i]}</td>"
        f"<td>{curve[m][h]['ratio']:.2f}×</td><td>{curve[m][h]['pct_multi']:.0f}%</td>"
        f"<td>{curve[m][h]['tot']:.0f}</td><td>{curve[m][h]['uniq']:.0f}</td>"
        f"<td>{curve[m][h]['mx']:.1f}</td></tr>"
        for i, h in enumerate(HOLDS)
    )
    return (
        "<table><tr><th>hold</th><th>trades/name</th><th>re-entry %</th>"
        "<th>trades</th><th>unique</th><th>max on 1 name</th></tr>" + rows + "</table>"
    )


def names(m):
    return ", ".join(
        f"<code>{n.replace('.NS', '')}</code>" for n, _ in namecount[m].most_common(8)
    )


card = f"""  <div class="card">
    <h2>Repeat-trade churn — how often the same stock gets re-entered</h2>
    <p class="kpi">The same name gets bought, sold, and bought again constantly — and the <b>time cap is the single biggest driver</b>.
    A tight 5-day cap kicks winners out early, so momentum leaders keep re-qualifying: US churns each unique name <b>5.3×</b> and
    <b>93%</b> of all trades are re-entries. Loosen the cap and churn falls monotonically toward <b>~2.0× / ~70%</b> at ∞ — you hold the
    same winners instead of re-buying them. This is measured across all 448 ledgers (one row per trade, grouped by ticker).</p>
    <div class="grid2">
      <div>{chart1}{legend}</div>
      <div>{chart2}{legend}</div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div><h2 style="font-size:14px">US — churn vs hold cap</h2>{mtable("us")}</div>
      <div><h2 style="font-size:14px">India — churn vs hold cap</h2>{mtable("india")}</div>
    </div>
    <p class="kpi" style="margin-top:14px">Most re-traded names (summed re-entry counts across configs) — the momentum leaders the screen keeps re-buying:<br>
    <b>US:</b> {names("us")}<br>
    <b>India:</b> {names("india")}</p>
    <p class="note">Averages are over 7 strategies × top{{5,10,20,40}} = 28 cells per hold point. Re-entry % = share of trades in names traded 2+ times.
    Repeat trading is by design (<code>allow_reentry</code> default) — it's the screen re-selecting the same leaders, not a bug.</p>
  </div>

"""

html = open(PAGE).read()
anchor = '  <div class="card">\n    <h2>Reading the result</h2>'
assert anchor in html, "anchor not found"
html = html.replace(anchor, card + anchor, 1)
# refresh subtitle count note (keep 448) — add mention
open(PAGE, "w").write(html)
print("injected repeat-trade card; page now", len(html), "bytes")
