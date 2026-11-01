// Renders the library-independent parts of every mockup (KPIs, heatmap,
// tables, warnings, palette strip) from window.REPORT and the theme's CSS tokens.
(function () {
  const R = window.REPORT;
  const css = getComputedStyle(document.documentElement);
  const tok = (n) => css.getPropertyValue("--" + n).trim();
  window.tok = tok;
  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1) + "%";
  window.fmtPct = fmtPct;
  const sign = (v) => (v >= 0 ? "num-pos" : "num-neg");

  const THEMES = [["terminal", "Terminal"], ["editorial", "Editorial"], ["slate", "Slate"],
    ["midnight", "Midnight"], ["graphite", "Graphite"], ["forest", "Forest"]];
  const here = location.pathname.split("/").pop().replace(".html", "");
  $("theme-bar").innerHTML = "<span>Theme:</span>" + THEMES.map(([f, n]) =>
    `<a href="${f}.html"${f === here ? ' aria-current="page"' : ""}>${n}</a>`).join("");

  $("title").textContent = R.title;
  $("subtitle").textContent = R.subtitle;
  $("kpis").innerHTML = R.metrics.map((m) => `<div class="kpi" data-tone="${m.tone}">
    <div class="label">${m.label}</div><div class="value">${m.value}</div>
    <div class="delta">${m.delta || (m.tone === "pos" ? "above target" : m.tone === "neg" ? "worst peak-to-trough" : "&nbsp;")}</div></div>`).join("");

  // Diverging heatmap: theme pos/neg poles mixed into a neutral midpoint.
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const all = Object.values(R.monthly).flat().filter((v) => v !== null);
  const max = Math.max(...all.map(Math.abs));
  const cell = (v) => {
    if (v === null) return '<td class="empty"></td>';
    const t = Math.min(1, Math.abs(v) / max), pole = v >= 0 ? "var(--pos)" : "var(--neg)";
    const bg = `color-mix(in oklab, ${pole} ${Math.round(12 + t * 78)}%, var(--div-mid))`;
    return `<td class="${t > 0.55 ? "strong" : ""}" style="background:${bg}" title="${fmtPct(v)}">${fmtPct(v)}</td>`;
  };
  $("heatmap").innerHTML = `<table class="heatmap"><thead><tr><th>Year</th>${months.map((m) => `<th style="text-align:center">${m}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(R.monthly).map(([y, row]) => `<tr><td>${y}</td>${row.map(cell).join("")}</tr>`).join("") + "</tbody></table>";

  const top = [...R.trades].sort((a, b) => b.pnl - a.pnl);
  $("trades").innerHTML = `<table><thead><tr><th>Ticker</th><th>Entry</th><th>Hold</th><th>Exit</th><th>Return</th><th>PnL</th></tr></thead><tbody>` +
    top.map((t) => `<tr><td>${t.ticker}</td><td>${t.entry}</td><td>${t.hold}d</td><td>${t.exit}</td><td class="${sign(t.ret)}">${fmtPct(t.ret)}</td><td class="${sign(t.pnl)}">${t.pnl >= 0 ? "+" : "−"}${Math.abs(t.pnl).toLocaleString()}</td></tr>`).join("") + "</tbody></table>";

  $("warnings").innerHTML = R.warnings.map((w) => `<li><span class="badge ${w.level}">${w.level === "good" ? "✓ OK" : w.level === "warning" ? "! Warning" : "✕ Critical"}</span><span>${w.text}</span></li>`).join("");

  const sw = ["bg", "surface", "ink", "muted", "accent", "series-1", "series-2", "series-3", "pos", "neg", "status-good", "status-warning", "status-critical"];
  $("swatches").innerHTML = sw.map((n) => `<span class="swatch"><b style="background:var(--${n})"></b>--${n} ${tok(n)}</span>`).join("");

  // Histogram bins shared by every library.
  const lo = -20, hi = 22, w = 2, bins = [];
  for (let x = lo; x < hi; x += w) bins.push({ x0: x, x1: x + w, n: R.trades.filter((t) => t.ret >= x && t.ret < x + w).length });
  window.HIST = bins;
})();
