# Backtest analysis pipeline

Reproduction scripts for the findings rendered in [`../webview/`](../webview/).
All sweeps run the screener CLI (`uv run screener backtest-rolling`) over 5 years,
US + India, 7 strategies (`mark_minervini`, `mq_us1..3`, `mq_in1..3`) × top{5,10,20,40}.

## Layout

| script | what it does |
|---|---|
| `run_sweep_nohold.sh` | No-hold-cap sweep — exit on strategy criteria only. Produces the ∞ (no-cap) point. |
| `run_holdcurve.sh` | Hold-time curve — same grid across hold caps {5,10,20,30,40,60,100}. |
| `run_ledger.sh` | Re-runs the grid with `--csv` to capture the trade ledger and measure repeat trading. |
| `agg_holdcurve.py` | Aggregates the hold + no-hold results into `holdcurve.json`. |
| `gen_holdcurve_html.py` | Renders `../webview/holdcurve.html` (inline-SVG charts, no external assets). |
| `gen_repeat_section.py` | Appends the "repeat-trade churn" card to `holdcurve.html` from the ledger metrics. |

## Configuration

Scripts are path-portable via environment variables (with sane defaults):

- `SCREENER_REPO` — repo checkout to run the CLI from (default: `git rev-parse --show-toplevel`).
- `ANALYSIS_DIR` — scratch dir for raw ledgers + per-combo metric CSVs (default: `./profiling/_analysis`, git-ignored).
- `WEBVIEW_DIR` — where the HTML pages are written (default: `./profiling/webview`).

## Run order

```bash
export ANALYSIS_DIR=./profiling/_analysis
bash profiling/scripts/run_sweep_nohold.sh   # ∞ point
bash profiling/scripts/run_holdcurve.sh      # finite hold caps  (448 backtests)
python profiling/scripts/agg_holdcurve.py    # -> holdcurve.json
python profiling/scripts/gen_holdcurve_html.py
bash profiling/scripts/run_ledger.sh         # ledger sweep      (448 backtests, --csv)
python profiling/scripts/gen_repeat_section.py
```

The sweeps are race-safe, resumable (skip combos with a valid result row), and cap
concurrency at 4 with a 900s per-run timeout.

## Headline findings

- **Hold-time curve.** US Sharpe peaks at a ~20-day cap (0.99) and decays to 0.69 at ∞
  (mean-reverting; cutting winners early helps, drawdown worsens with longer holds).
  India Sharpe peaks at ~40 days (1.24) with return maxing at ∞ (+100.9%) — trend-persistent,
  let winners run. `mq_in2` is a broken signal (Sharpe <0.4 in US).
- **Sizing is not a lever.** All sizers clamp to the equal-slot ceiling, so the curve
  collapses to `equal_slot` — Sharpe-neutral.
- **Repeat-trade churn.** The same name is re-entered constantly, driven by the time cap:
  a 5-day cap re-buys each unique name ~5.3× (93% of trades are re-entries) vs ~2.0× / ~70%
  at ∞. This is by design (`allow_reentry`) — the screen re-selecting the same momentum
  leaders (US: NVDA/AVGO/ANET…; India: BSE/BEL/INDHOTEL…), not a bug.
- **Profiling.** See `../flamegraph_gpu_analysis_2026_07_25.md` and the flamegraphs /
  snakeviz profile in `../webview/` for the bottleneck ranking + GPU-offload assessment.

Benchmarks (5yr): US +79.27%, India +50.19%.
