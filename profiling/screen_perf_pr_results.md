# Screen CLI performance PR results

**Branch:** `perf/screen-cli-startup`
**Harness:** `profiling/scripts/bench_screen.py`
**Environment:** Turso configured via `.env`; pandas 3.0.5; numpy 2.4.6;
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
Means of N=3 timed runs after one warmup.

## Problem

Default `screener screen` spent ~1.1s on warm CSV:

- ~0.55-0.65s eager CLI imports
- ~0.35-0.50s synchronous Turso usage x2
- ~0.18s always-on Plotly HTML report on the table path

Workflow scan itself was ~10ms warm.

## Commits

1. `perf(usage): reuse Turso client and avoid double-connect per command`
2. `perf(usage): non-blocking Turso usage recording`
3. `perf(cli): lazy-import subcommands to cut startup`
4. `perf(screen): make HTML report opt-in via --report`
5. `perf(screen): avoid importing plotly unless rendering a report`
6. `perf(usage): default flush join budget to 50ms` (residual wall tune)
7. `docs(profiling): screen bench harness and per-commit measurements`

## Measurement table (mean_s)

| metric | baseline | after C1 | after C2 | after C3 | after C4 | after C5 | final | delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1_cli_warm_csv | 1.080 | 1.003 | 0.780 | 0.620 | 0.623 | 0.533 | **0.507** | **-53.1%** |
| M2_cli_warm_table | 1.357 | 1.530 | 1.057 | 0.893 | 0.703 | 0.610 | **0.573** | **-57.8%** |
| M3_cli_help | 0.627 | 0.630 | 0.637 | 0.143 | 0.143 | 0.140 | **0.140** | **-77.7%** |
| M4_workflow_warm_csv | 0.009 | 0.011 | 0.009 | 0.009 | 0.009 | 0.009 | **0.009** | ~flat |
| M5_workflow_warm_full | 0.137 | 0.136 | 0.140 | 0.130 | 0.049 | 0.049 | **0.045** | **-67.2%** |
| M6_import_cli | 0.633 | 0.647 | 0.637 | 0.140 | 0.140 | 0.147 | **0.140** | **-77.9%** |
| M7_usage_pair | 0.415 | 0.232 | 0.100 | 0.100 | 0.100 | 0.100 | **0.050** | **-87.9%** |
| M8_cli_warm_csv_no_turso | 0.640 | 0.667 | 0.657 | 0.510 | 0.513 | 0.420 | **0.430** | **-32.8%** |

Full per-commit samples: `profiling/_analysis/screen_bench_log.md`.

## What each commit moved

| Commit | Primary metrics | Effect |
| --- | --- | --- |
| C1 client reuse | M7, M1 | M7 0.415 → 0.232 (one connect + DDL cache) |
| C2 non-blocking | M1, M7 | M1 near no-Turso; M7 ≈ flush budget |
| C3 lazy CLI | M3, M6, M1 | Help/import ~0.14s; no eager backtester/plotly |
| C4 opt-in report | M2, M5 | M5 ~0.05s (history only); table path near CSV |
| C5 lazy plotly | M1, M4 path | CSV workflow never imports plotly |
| flush 50ms | M1, M7 | Residual Turso wait cut in half |

## Turso / `.env` notes

- Baseline and most rows used Turso credentials from repo `.env`.
- **M8** runs from `/tmp` with `TURSO_*` unset so usage no-ops.
- After C2, M1 is close to M8; remaining gap is mostly the short flush join plus noise.
- `SCREENER_USAGE=0` (or `off`/`false`/`no`) skips all usage I/O.
- `SCREENER_USAGE_FLUSH_MS` overrides the best-effort flush join (default 50).

## UX

- HTML report only with `--report` and/or `--open-report` (no auto temp HTML).
- Usage still recorded when Turso is configured (non-blocking, best-effort).
- Default warm `screen -c ema -n 50` result ordering unchanged when cache is warm.

## Success criteria

| Criterion | Result |
| --- | --- |
| M1 ≥50% better with Turso | 1.080 → 0.507 (**53.1%**) |
| M3 ≥40% better | 0.627 → 0.140 (**77.7%**) |
| M5 no Plotly by default | 0.045 ≈ M4 + history |
| Warm CSV returns 50 rows | verified |
| No fetch_limit change | not changed |
