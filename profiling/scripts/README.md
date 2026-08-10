# Profiling scripts

## Screen CLI bench (`bench_screen.py`)

Measures shell and in-process costs for `screener screen` (metrics M1-M8).

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python profiling/scripts/bench_screen.py --label baseline --runs 3
```

Outputs:

- JSON: `profiling/_analysis/screen_bench_<label>.json` (baseline uses `screen_bench_baseline.json`)
- Markdown log append: `profiling/_analysis/screen_bench_log.md`

| Metric | Isolates |
| --- | --- |
| M1_cli_warm_csv | Full shell warm CSV |
| M2_cli_warm_table | Full shell warm table |
| M3_cli_help | Import + CLI load (`--help`) |
| M4_workflow_warm_csv | In-process workflow CSV |
| M5_workflow_warm_full | In-process workflow table path |
| M6_import_cli | `import screener.cli` wall |
| M7_usage_pair | Turso usage + invocation pair |
| M8_cli_warm_csv_no_turso | Shell CSV without Turso (cwd `/tmp`, env cleared) |

Turso is detected from `TURSO_*` env or repo `.env`. Set `SCREENER_USAGE=0` to skip usage I/O entirely.

See `profiling/screen_perf_pr_results.md` for the per-commit measurement write-up.
