"""Browser UI for comparing multiple rolling backtest strategies."""

from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import click
import pandas as pd
from plotly.offline import get_plotlyjs
from rich.console import Console

from screener.backtester.cli_common import DEFAULT_BENCHMARK, resolve_min_filters
from screener.backtester.data import build_price_fetcher
from screener.backtester.display import trades_dataframe
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.strategies import STRATEGIES, resolve_strategy
from screener.universes import UniverseName, load_current_universe


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _result_payload(name: str, result: BacktestResult) -> dict[str, Any]:
    equity = result.equity_curve.sort_index()
    benchmark = result.benchmark_curve.sort_index()
    first_equity = float(equity.iloc[0]) if not equity.empty else 0.0
    first_benchmark = float(benchmark.iloc[0]) if not benchmark.empty else 0.0
    curves = []
    dates = sorted(set(equity.index).union(set(benchmark.index)))
    for day in dates:
        eq = equity.get(day)
        bench = benchmark.get(day)
        curves.append(
            {
                "date": pd.Timestamp(day).date().isoformat(),
                "strategy": name,
                "strategy_return": (
                    float(eq) / first_equity - 1.0
                    if first_equity and pd.notna(eq)
                    else None
                ),
                "benchmark_return": (
                    float(bench) / first_benchmark - 1.0
                    if first_benchmark and pd.notna(bench)
                    else None
                ),
            }
        )
    trades = trades_dataframe(result)
    trade_rows = [] if trades.empty else trades.to_dict(orient="records")
    return {
        "strategy": name,
        "base_strategy": result.config.strategy_name or name,
        "metrics": result.metrics,
        "curves": curves,
        "trades": trade_rows,
        "warnings": result.warnings,
    }


def _universe_note(name: UniverseName, symbols: tuple[str, ...], source: str, cached_path: object) -> dict[str, Any]:
    return {
        "name": name,
        "symbol_count": len(symbols),
        "source": source,
        "cached_path": str(cached_path),
    }


def compare_payload(
    *,
    market: str,
    strategies: list[str],
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    hold: int,
    top: int,
    initial_capital: float,
    benchmark: str | None = None,
    min_price: float | None = None,
    min_avg_dollar_volume: float | None = None,
    universe: UniverseName | None = None,
    compare_universe: UniverseName | None = None,
    use_universe_cache: bool = True,
) -> dict[str, Any]:
    """Run rolling backtests for selected named strategies and serialize them."""
    if not strategies:
        raise ValueError("Select at least one strategy.")
    universe_note = None
    compare_universe_note = None
    ticker_runs: list[tuple[str, tuple[str, ...]]] = []
    if universe is not None:
        loaded = load_current_universe(
            universe,
            as_of=end_date,
            use_cache=use_universe_cache,
        )
        tickers = loaded.symbols
        universe_note = _universe_note(
            loaded.name, loaded.symbols, loaded.source, loaded.cached_path
        )
        ticker_runs.append((loaded.name, tickers))
    elif not tickers:
        raise ValueError("Enter at least one ticker.")
    else:
        ticker_runs.append(("tickers", tickers))

    if universe is None and compare_universe is not None:
        loaded = load_current_universe(
            compare_universe,
            as_of=end_date,
            use_cache=use_universe_cache,
        )
        compare_universe_note = _universe_note(
            loaded.name, loaded.symbols, loaded.source, loaded.cached_path
        )
        ticker_runs.append((loaded.name, loaded.symbols))

    bench = benchmark or DEFAULT_BENCHMARK.get(market, "SPY")
    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )
    fetcher = build_price_fetcher(auto_adjust=True)
    results: list[dict[str, Any]] = []
    for name in strategies:
        strategy = resolve_strategy(name)
        for run_label, run_tickers in ticker_runs:
            cfg = BacktestConfig(
                market=market,
                as_of=end_date,
                hold=int(hold),
                top=int(top),
                entry_expr=strategy.entry,
                exit_expr=strategy.exit,
                stop_loss=None,
                take_profit=None,
                trailing_stop=None,
                slippage_bps=0.0,
                commission_bps=0.0,
                initial_capital=float(initial_capital),
                benchmark=bench,
                strategy_name=name,
                tickers=run_tickers,
                min_price=resolved_min_price,
                min_avg_dollar_volume=resolved_min_adv,
            )
            result = run_rolling_backtest(
                cfg,
                fetcher,
                start_date=start_date,
                end_date=end_date,
            )
            results.append(_result_payload(f"{name} · {run_label}", result))
    return {
        "request": {
            "market": market,
            "strategies": strategies,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "hold": hold,
            "top": top,
            "initial_capital": initial_capital,
            "benchmark": bench,
            "universe": universe,
            "universe_note": universe_note,
            "compare_universe": compare_universe,
            "compare_universe_note": compare_universe_note,
        },
        "results": results,
    }


def _lab_html() -> str:
    strategy_options = "\n".join(
        f'<label><input type="checkbox" name="strategy" value="{html.escape(name)}" '
        f'{"checked" if i < 3 else ""}> {html.escape(name)}</label>'
        for i, name in enumerate(STRATEGIES)
    )
    today = date.today()
    start = today - timedelta(days=365)
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Strategy Lab</title>
  <script>{plotly_js}</script>
  <style>
    :root {{
      --ink: #20231f;
      --muted: #687068;
      --paper: #f4f1e8;
      --panel: #fffdf8;
      --line: #d9d1c1;
      --accent: #0f766e;
      --bad: #b91c1c;
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
      padding: 20px 28px;
      border-bottom: 1px solid var(--line);
      background: #e8e2d4;
    }}
    h1 {{ margin: 0; font-size: 26px; }}
    main {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
      padding: 18px 28px 32px;
    }}
    aside, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      min-width: 0;
    }}
    .controls {{ display: grid; gap: 12px; align-content: start; }}
    label {{ display: grid; gap: 5px; font-size: 13px; color: var(--muted); }}
    input, select, button {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }}
    .strategy-list {{
      display: grid;
      gap: 7px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fbfaf6;
    }}
    .strategy-list label {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
    }}
    button {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      cursor: pointer;
      font-weight: 700;
    }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .workspace {{ display: grid; gap: 16px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 11px;
      background: #fbfaf6;
    }}
    .metric span {{ color: var(--muted); font-size: 12px; display: block; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 18px; }}
    .status {{ color: var(--muted); font-size: 13px; }}
    .error {{ color: var(--bad); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; }}
    th {{ background: #e8e2d4; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 360px; overflow: auto; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Backtest Strategy Lab</h1>
  </header>
  <main>
    <aside class="controls">
      <label>Market
        <select id="market"><option value="us">US</option><option value="india">India</option></select>
      </label>
      <label>Universe
        <select id="universe">
          <option value="manual">Manual tickers</option>
          <option value="sp500">S&P 500</option>
          <option value="nifty50">Nifty 50</option>
        </select>
      </label>
      <label>Tickers
        <input id="tickers" value="AAPL,MSFT" placeholder="AAPL,MSFT,NVDA">
      </label>
      <label>Compare Tickers Against
        <select id="compare-universe">
          <option value="none">None</option>
          <option value="sp500">S&P 500</option>
          <option value="nifty50">Nifty 50</option>
        </select>
      </label>
      <label>Start
        <input id="start" type="date" value="{start.isoformat()}">
      </label>
      <label>End
        <input id="end" type="date" value="{today.isoformat()}">
      </label>
      <label>Hold days
        <input id="hold" type="number" min="1" value="20">
      </label>
      <label>Top slots
        <input id="top" type="number" min="1" value="5">
      </label>
      <label>Initial capital
        <input id="capital" type="number" min="1" value="100000">
      </label>
      <div class="strategy-list">{strategy_options}</div>
      <button id="run">Run Comparison</button>
      <div id="status" class="status">Choose strategies and run.</div>
    </aside>
    <div class="workspace">
      <section><div id="curve"></div></section>
      <section><h2>Summary</h2><div id="metrics" class="metrics"></div></section>
      <section><h2>Trades</h2><div class="table-wrap"><table id="trades"></table></div></section>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const runBtn = document.getElementById("run");
    const universeEl = document.getElementById("universe");
    const compareUniverseEl = document.getElementById("compare-universe");
    const tickersEl = document.getElementById("tickers");
    const pct = value => value == null || Number.isNaN(value) ? "" : `${{(value * 100).toFixed(2)}}%`;
    const num = value => value == null || Number.isNaN(value) ? "" : Number(value).toFixed(3);

    function selectedStrategies() {{
      return [...document.querySelectorAll('input[name="strategy"]:checked')].map(el => el.value);
    }}

    function syncUniverseMode() {{
      const manual = universeEl.value === "manual";
      tickersEl.disabled = !manual;
      tickersEl.style.opacity = manual ? "1" : ".55";
      compareUniverseEl.disabled = !manual;
      compareUniverseEl.style.opacity = manual ? "1" : ".55";
    }}

    universeEl.addEventListener("change", syncUniverseMode);
    syncUniverseMode();

    function renderMetrics(results) {{
      const keys = [
        ["total_return", "Total Return", pct],
        ["benchmark_return", "Benchmark", pct],
        ["max_drawdown", "Max DD", pct],
        ["sharpe", "Sharpe", num],
        ["trade_count", "Trades", value => value ?? ""],
        ["exposure", "Exposure", pct],
        ["hit_rate", "Hit Rate", pct],
      ];
      document.getElementById("metrics").innerHTML = results.flatMap(result =>
        keys.map(([key, label, fmt]) =>
          `<article class="metric"><span>${{result.strategy}} · ${{label}}</span><strong>${{fmt(result.metrics[key])}}</strong></article>`
        )
      ).join("");
    }}

    function renderCurve(results) {{
      const traces = results.map(result => ({{
        x: result.curves.map(row => row.date),
        y: result.curves.map(row => row.strategy_return),
        name: result.strategy,
        mode: "lines",
        type: "scatter"
      }}));
      if (results[0]) {{
        traces.push({{
          x: results[0].curves.map(row => row.date),
          y: results[0].curves.map(row => row.benchmark_return),
          name: "Benchmark",
          mode: "lines",
          type: "scatter",
          line: {{dash: "dot", color: "#52525b"}}
        }});
      }}
      Plotly.newPlot("curve", traces, {{
        title: "Strategy Return Comparison",
        paper_bgcolor: "#fffdf8",
        plot_bgcolor: "#fbfaf6",
        yaxis: {{tickformat: ".0%"}},
        margin: {{l: 56, r: 24, t: 44, b: 42}},
        hovermode: "x unified"
      }}, {{displaylogo: false, responsive: true}});
    }}

    function renderTrades(results) {{
      const rows = results.flatMap(result => result.trades.map(trade => ({{strategy: result.strategy, ...trade}})));
      const cols = ["strategy", "ticker", "signal_date", "entry_date", "exit_date", "exit_reason", "return_pct", "pnl"];
      const head = `<thead><tr>${{cols.map(c => `<th>${{c}}</th>`).join("")}}</tr></thead>`;
      const body = `<tbody>${{rows.map(row => `<tr>${{cols.map(c => `<td>${{c === "return_pct" ? pct(row[c]) : row[c] ?? ""}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
      document.getElementById("trades").innerHTML = rows.length ? head + body : "<tbody><tr><td>No trades.</td></tr></tbody>";
    }}

    runBtn.addEventListener("click", async () => {{
      runBtn.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "Running backtests...";
      try {{
        const payload = {{
          market: document.getElementById("market").value,
          universe: universeEl.value,
          compare_universe: compareUniverseEl.value,
          strategies: selectedStrategies(),
          tickers: tickersEl.value,
          start: document.getElementById("start").value,
          end: document.getElementById("end").value,
          hold: Number(document.getElementById("hold").value),
          top: Number(document.getElementById("top").value),
          initial_capital: Number(document.getElementById("capital").value)
        }};
        const response = await fetch("/api/run", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(payload)
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Backtest failed.");
        renderCurve(data.results);
        renderMetrics(data.results);
        renderTrades(data.results);
        const universeText = data.request.universe_note
          ? ` across ${{data.request.universe_note.symbol_count}} symbols`
          : data.request.compare_universe_note
          ? ` plus ${{data.request.compare_universe_note.symbol_count}} comparison symbols`
          : "";
        statusEl.textContent = `Rendered ${{data.results.length}} strategy runs${{universeText}}.`;
      }} catch (err) {{
        statusEl.className = "status error";
        statusEl.textContent = err.message;
      }} finally {{
        runBtn.disabled = false;
      }}
    }});
  </script>
</body>
</html>
"""


class LabHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send(HTTPStatus.OK, _lab_html().encode(), "text/html; charset=utf-8")
            return
        if self.path == "/api/strategies":
            body = json.dumps({"strategies": list(STRATEGIES)}).encode()
            self._send(HTTPStatus.OK, body, "application/json")
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size)
            payload = json.loads(raw.decode() or "{}")
            tickers = tuple(
                item.strip()
                for item in str(payload.get("tickers", "")).split(",")
                if item.strip()
            )
            universe_raw = str(payload.get("universe", "manual"))
            universe = None if universe_raw == "manual" else universe_raw
            if universe not in {None, "sp500", "nifty50"}:
                raise ValueError(f"Unknown universe: {universe_raw}")
            compare_universe_raw = str(payload.get("compare_universe", "none"))
            compare_universe = (
                None if compare_universe_raw == "none" else compare_universe_raw
            )
            if compare_universe not in {None, "sp500", "nifty50"}:
                raise ValueError(f"Unknown comparison universe: {compare_universe_raw}")
            data = compare_payload(
                market=str(payload.get("market", "us")),
                strategies=list(payload.get("strategies", [])),
                tickers=tickers,
                start_date=datetime.strptime(str(payload["start"]), "%Y-%m-%d").date(),
                end_date=datetime.strptime(str(payload["end"]), "%Y-%m-%d").date(),
                hold=int(payload.get("hold", 20)),
                top=int(payload.get("top", 5)),
                initial_capital=float(payload.get("initial_capital", 100_000.0)),
                universe=universe,
                compare_universe=compare_universe,
            )
            body = json.dumps(data, default=_json_default).encode()
            self._send(HTTPStatus.OK, body, "application/json")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self._send(HTTPStatus.BAD_REQUEST, body, "application/json")

    def log_message(self, format: str, *args: Any) -> None:
        return


@click.command(name="backtest-lab")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8766, show_default=True)
def backtest_lab(host: str, port: int) -> None:
    """Launch a browser UI for comparing rolling backtest strategies."""
    console = Console()
    server = ThreadingHTTPServer((host, int(port)), LabHandler)
    console.print(f"[green]Backtest lab:[/green] http://{host}:{port}/")
    console.print("[dim]Press Ctrl+C to stop the lab server.[/dim]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["backtest_lab", "compare_payload"]
