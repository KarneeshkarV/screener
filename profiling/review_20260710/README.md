# Backtest performance profiling — 2026-07-10

Deterministic, offline profiling material for
[`findings/backtest_performance_review_2026_07_10.md`](../../findings/backtest_performance_review_2026_07_10.md).

All commands run from the repository root with Python 3.11 through `uv`.

## Reproduction

```bash
uv sync --all-groups --extra vectorbt

# Core engine baselines and phase timers
uv run python profiling/harness.py --path rolling --tickers 300 --years 3 --top 10 --repeat 3
uv run python profiling/harness.py --path historical --tickers 300 --years 3 --top 10 --repeat 3
uv run python profiling/review_20260710/phase_bench.py --path rolling
uv run python profiling/review_20260710/phase_bench.py --path historical

# Output-equivalent optimization ceilings
uv run python profiling/review_20260710/hotspot_poc.py

# Repeated-data and optimization paths
uv run python profiling/review_20260710/cache_bench.py
uv run python profiling/review_20260710/grid_bench.py

# Other backtest engines and report construction
uv run python profiling/review_20260710/earnings_harness.py
uv run python profiling/review_20260710/vbt_harness.py --chunk-size 16
uv run python profiling/review_20260710/report_bench.py
```

## Flame graphs

- [`rolling_cpu_flame_20hz.svg`](rolling_cpu_flame_20hz.svg) — complete sampled
  rolling engine run.
- [`historical_cpu_flame_20hz.svg`](historical_cpu_flame_20hz.svg) — complete
  sampled historical engine run.
- [`earnings_cpu_flame_20hz.svg`](earnings_cpu_flame_20hz.svg) — 2,400-event
  earnings backtest.
- [`bot_performance_cpu_flame.svg`](bot_performance_cpu_flame.svg) — worst-case
  paper-performance per-symbol aggregation from the bot checkout.

The `py-spy` graphs use a deliberately low 20 Hz sampling rate because 200 Hz
materially slowed this pandas-heavy workload. Treat their widths as corroborating
call-stack evidence, not as absolute timings. Exact call counts and cumulative
times in the report come from `cProfile`; raw `.prof` files and multi-megabyte
instrumented flame graphs are intentionally ignored because they are
Python-version-specific and reproducible with the harnesses above.

## Workload

The primary core workload is seeded synthetic OHLCV data with 300 symbols,
1,306 bars per symbol (warm-up included), a three-year/756-session rolling
window, ten portfolio slots, SMA crossover entry/exit expressions, filters,
stops, commission, and fixed slippage. It produced 709 rolling trades and 14
historical trades on the reviewed commit.

The harness excludes provider/network latency unless the benchmark specifically
targets the parquet cache. It is a deterministic engine benchmark, not a claim
about live-provider response times.
