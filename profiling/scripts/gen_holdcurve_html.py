import os
#!/usr/bin/env python3
"""Generate a self-contained dark-themed hold-time-curve summary page,
same style as profiling/webview/index.html. Inline SVG charts, no external assets."""
import json, math

SP = os.environ.get("ANALYSIS_DIR", os.path.join(os.getcwd(), "profiling", "_analysis"))
WEBVIEW = os.environ.get("WEBVIEW_DIR", os.path.join(os.getcwd(), "profiling", "webview"))
OUT = os.path.join(WEBVIEW, "holdcurve.html")
D = json.load(open(f"{SP}/holdcurve.json"))
HOLDS = D["holds"]                       # ["5","10","20","30","40","60","100","inf"]
STRATS = D["strat_order"]
MC = D["market_curve"]
SC = D["strat_curve"]
BENCH = D["bench"]
XLAB = ["5","10","20","30","40","60","100","∞"]

def xpos(i, x0, w): return x0 + w*i/(len(HOLDS)-1)

def line_chart(series, ylabel, w=520, h=260, fmt="{:.2f}", zero=True):
    """series: list of (label,color,[vals]) ; vals aligned to HOLDS (None ok)."""
    padL,padR,padT,padB = 52,14,16,34
    x0,y0 = padL, padT
    pw,ph = w-padL-padR, h-padT-padB
    allv=[v for _,_,vals in series for v in vals if v is not None]
    lo,hi = min(allv), max(allv)
    if zero and lo>0: lo=0
    if zero and hi<0: hi=0
    if hi==lo: hi=lo+1
    pad=(hi-lo)*0.08; lo-=pad; hi+=pad
    def yp(v): return y0+ph*(1-(v-lo)/(hi-lo))
    s=[f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    # grid + y labels
    for g in range(5):
        v=lo+(hi-lo)*g/4; y=yp(v)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#262a33"/>')
        s.append(f'<text x="{x0-6}" y="{y+3:.1f}" fill="#7d8697" font-size="10" text-anchor="end">{fmt.format(v)}</text>')
    if lo<0<hi:
        yz=yp(0); s.append(f'<line x1="{x0}" y1="{yz:.1f}" x2="{x0+pw}" y2="{yz:.1f}" stroke="#4a5160" stroke-dasharray="3 3"/>')
    # x labels
    for i,lab in enumerate(XLAB):
        x=xpos(i,x0,pw)
        s.append(f'<text x="{x:.1f}" y="{h-10}" fill="#9aa4b2" font-size="10" text-anchor="middle">{lab}</text>')
    # series
    for lab,col,vals in series:
        pts=[(xpos(i,x0,pw),yp(v)) for i,v in enumerate(vals) if v is not None]
        d=" ".join(f"{'M' if k==0 else 'L'}{x:.1f} {y:.1f}" for k,(x,y) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for x,y in pts: s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{col}"/>')
    s.append(f'<text x="{x0}" y="11" fill="#e6e6e6" font-size="11" font-weight="600">{ylabel}</text>')
    s.append('</svg>')
    return "".join(s)

def legend(series):
    return '<div class="legend">'+" ".join(
        f'<span><i style="background:{c}"></i>{l}</span>' for l,c,_ in series)+'</div>'

US,IN = "#4f9dff","#f5a623"
def mc_vals(m,key): return [MC[m][h][key] for h in HOLDS]

# ---- market-level charts ----
sharpe_series=[("US avg Sharpe",US,mc_vals("us","sharpe")),("India avg Sharpe",IN,mc_vals("india","sharpe"))]
ret_series=[("US avg Return %",US,mc_vals("us","total_return")),("India avg Return %",IN,mc_vals("india","total_return"))]
dd_series=[("US avg MaxDD %",US,mc_vals("us","max_dd")),("India avg MaxDD %",IN,mc_vals("india","max_dd"))]

# ---- per-strategy small multiples (Sharpe vs hold) ----
PAL=["#4f9dff","#f5a623","#5fd08a","#e0679a","#b98bff","#4fd0d0","#d4c04f"]
def strat_multiples(m):
    series=[(s,PAL[i],[SC[m][s][h]["sharpe"] for h in HOLDS]) for i,s in enumerate(STRATS)]
    return line_chart(series,f"{m.upper()} — Sharpe vs hold, per strategy",w=560,h=300,fmt="{:.2f}")+legend(series)

# ---- market summary table ----
def mc_table(m):
    rows=[]
    bestSh=max(HOLDS,key=lambda h:MC[m][h]["sharpe"])
    bestRet=max(HOLDS,key=lambda h:MC[m][h]["total_return"])
    for h,lab in zip(HOLDS,XLAB):
        c=MC[m][h]
        cls=' class="hot"' if h in (bestSh,bestRet) else ''
        star=" ★" if h==bestSh else ""
        rows.append(f"<tr{cls}><td>{lab}{star}</td><td>{c['sharpe']:.3f}</td><td>{c['total_return']:+.1f}%</td>"
                    f"<td>{c['cagr']:+.2f}%</td><td>{c['max_dd']:+.1f}%</td><td>{c['trades']:.0f}</td></tr>")
    return ("<table><thead><tr><th>hold</th><th>avg Sharpe</th><th>avg Return</th>"
            "<th>avg CAGR</th><th>avg MaxDD</th><th>avg trades</th></tr></thead><tbody>"
            +"".join(rows)+"</tbody></table>")

# ---- best-hold-per-strategy table ----
def best_hold(m,s,metric):
    best=None;bh=None
    for h in HOLDS:
        v=SC[m][s][h][metric]
        if v is None: continue
        if best is None or v>best: best=v;bh=h
    return ("∞" if bh=="inf" else bh),best
def bh_table(m):
    rows=[]
    for s in STRATS:
        bh,bv=best_hold(m,s,"sharpe"); rr,rv=best_hold(m,s,"total_return")
        rows.append(f"<tr><td>{s}</td><td>{bh}</td><td>{bv:.3f}</td><td>{rr}</td><td>{rv:+.1f}%</td></tr>")
    return ("<table><thead><tr><th>strategy</th><th>Sharpe-opt hold</th><th>Sharpe</th>"
            "<th>Return-opt hold</th><th>Return</th></tr></thead><tbody>"+"".join(rows)+"</tbody></table>")

html=f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screener — hold-time curve</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0; line-height:1.5; background:#0f1115; color:#e6e6e6; }}
  header {{ padding:20px 28px; border-bottom:1px solid #262a33; background:#151821; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:#9aa4b2; font-size:13px; }}
  main {{ padding:20px 28px; max-width:1180px; }}
  .card {{ border:1px solid #262a33; border-radius:10px; padding:16px 18px; margin:14px 0; background:#151821; }}
  .card h2 {{ margin:0 0 10px; font-size:16px; }}
  a.back {{ color:#4f9dff; text-decoration:none; font-size:13px; }}
  code {{ background:#20242e; padding:1px 5px; border-radius:4px; font-size:13px; }}
  .note {{ color:#f0b429; font-size:13px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media(max-width:820px){{ .grid2{{grid-template-columns:1fr;}} }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:6px; }}
  th,td {{ text-align:right; padding:5px 9px; border-bottom:1px solid #262a33; }}
  th:first-child,td:first-child {{ text-align:left; }}
  th {{ color:#9aa4b2; font-weight:600; }}
  tr.hot td {{ background:#1c2534; }}
  .legend {{ margin-top:8px; font-size:12px; color:#9aa4b2; }}
  .legend span {{ margin-right:14px; white-space:nowrap; }}
  .legend i {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:4px; vertical-align:-1px; }}
  .kpi {{ font-size:14px; }}
  .kpi b {{ color:#5fd08a; }}
  ul {{ margin:6px 0; }}
</style></head><body>
<header>
  <h1>Screener — hold-time curve sweep</h1>
  <div class="sub">5yr · US + India · 7 strategies × top{{5,10,20,40}} × hold{{5,10,20,30,40,60,100,∞}} · equal_slot · stop 0.08 · 448 backtests · 2026-07-25 &nbsp;·&nbsp; <a class="back" href="index.html">← flamegraph analysis</a></div>
</header>
<main>

  <div class="card">
    <h2>The one-line answer</h2>
    <p class="kpi">The hold cap is a real dial, and the two markets want <b>opposite things</b>.
    <b>US</b> is mean-reverting: Sharpe peaks at a <b>~20-day</b> cap ({MC['us']['20']['sharpe']:.3f}) and <em>decays</em> as you let winners run
    ({MC['us']['inf']['sharpe']:.3f} at ∞). <b>India</b> is trend-persistent: Sharpe keeps climbing to a <b>~40-day</b> plateau
    ({MC['india']['40']['sharpe']:.3f}) and raw return maxes at <b>∞</b> ({MC['india']['inf']['total_return']:.1f}%).
    Benchmarks 5yr: US {BENCH['us']:+.1f}%, India {BENCH['india']:+.1f}%.</p>
    <p class="note">★ marks the Sharpe-optimal hold per market. Every point is an average over 28 cells (7 strategies × 4 portfolio sizes); the ∞ point reuses the no-cap equal_slot sweep.</p>
  </div>

  <div class="card">
    <h2>Market-level curves — averaged over 7 strategies × 4 portfolio sizes</h2>
    <div class="grid2">
      <div>{line_chart(sharpe_series,"Sharpe vs hold cap")}{legend(sharpe_series)}</div>
      <div>{line_chart(ret_series,"Total return % vs hold cap")}{legend(ret_series)}</div>
    </div>
    <div style="margin-top:16px">{line_chart(dd_series,"Max drawdown % vs hold cap (less negative = better)")}{legend(dd_series)}</div>
  </div>

  <div class="card">
    <h2>US — market summary</h2>
    {mc_table('us')}
  </div>
  <div class="card">
    <h2>India — market summary</h2>
    {mc_table('india')}
  </div>

  <div class="card">
    <h2>Per-strategy Sharpe curves</h2>
    <div class="grid2">
      <div>{strat_multiples('us')}</div>
      <div>{strat_multiples('india')}</div>
    </div>
  </div>

  <div class="card">
    <h2>Best hold per strategy</h2>
    <div class="grid2">
      <div><h3 style="font-size:13px;color:#9aa4b2;margin:0 0 4px">US</h3>{bh_table('us')}</div>
      <div><h3 style="font-size:13px;color:#9aa4b2;margin:0 0 4px">INDIA</h3>{bh_table('india')}</div>
    </div>
  </div>

  <div class="card">
    <h2>Reading the result</h2>
    <ul>
      <li><b>US = cut winners early.</b> Sharpe rises 5→20 then falls monotonically; drawdown worsens the longer you hold ({MC['us']['20']['max_dd']:+.1f}% at 20 → {MC['us']['inf']['max_dd']:+.1f}% at ∞). A tight ~20-day time-stop is doing risk control the exit rules don't.</li>
      <li><b>India = let winners run.</b> Return climbs almost the whole way to ∞ ({MC['india']['20']['total_return']:+.1f}% at 20 → {MC['india']['inf']['total_return']:+.1f}% at ∞) while drawdown barely moves; the 30–40 band is the risk-adjusted sweet spot, ∞ the return-max.</li>
      <li><b>The knee is sharp, not flat.</b> Both markets gain most of their Sharpe going from a 5-day to a 20–40-day cap; sub-10-day caps churn (US 1168, India 1327 trades) and bleed to costs without improving risk-adjusted return.</li>
      <li><b>Strategy-specific:</b> the momentum-quant US strats (mq_us1/us2/us3) all Sharpe-peak at hold=20 regardless of market; mean-reversion-flavored names tolerate longer holds. mq_in2 is the one weak sheep (Sharpe &lt;0.4 in US) — hold tuning can't fix a broken signal.</li>
    </ul>
  </div>

</main></body></html>"""
open(OUT,"w").write(html)
print(f"wrote {OUT}  ({len(html)} bytes)")
