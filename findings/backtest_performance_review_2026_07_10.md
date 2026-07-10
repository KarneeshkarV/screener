# Backtest and performance bottleneck review

**Date:** 2026-07-10 (Asia/Kolkata)

**Screener ref:** `4acc6b149248f1ad47e5b7481b5acd91f2c084b5` (`main`)

**Review branch:** `review/backtest-performance-20260710`

**Bot ref (read-only review):** `3ff26d8db9adea1eae264c2b38e8b5ccaa1928e4`

## Executive conclusion

The current engines are not suffering from one slow inner loop. The end-to-end
cost is split among data reloading, signal preparation, simulation, and result
assembly, and repeated workflows multiply all four.

The priority order is:

1. **Fix volume-impact slippage correctness.** The engine computes ADV and
   volatility, but fill calls never carry position shares. `VolumeImpactSlippage`
   therefore receives `shares=0` and returns zero impact. An extreme `k=1000`
   produced exactly the same trades and equity as zero slippage. For the shipped
   built-in models, the associated liquidity work is currently dead computation.
2. **Fetch/load a price panel once per user operation.** A full cache hit for
   301 symbols took a median **5.69 s** from parquet versus **0.25 s** from the
   identical in-memory panel. Grid search, walk-forward, and Strategy Lab reload
   that same data for every run.
3. **Replace per-trade pandas exposure masks.** Exposure alone took about
   **1.10 s** uninstrumented for 709 trades × 756 sessions. An exact
   difference-array equivalent took **0.016 s** (71× faster) and preserved the
   current inclusive entry/exit semantics.
4. **Split preparation from execution for repeated runs.** The default grid has
   144 combinations. It currently repeats data access, Pine evaluation, filter
   calculation, candidate-matrix construction, equity assembly, and all metrics
   for each combination even when only stop/target/hold settings change.
5. **Bound vectorbt by actual peak memory, not only tiled-price bytes.** A
   100-symbol × 1,306-bar × 93-combination sweep peaked at **1.39 GB** despite a
   nominal 1 GiB chunk target. A chunk size of 16 still peaked at **1.07 GB**.
6. **Remove repeated full-frame filtering from the earnings engines.** A
   2,400-event offline run took **13.80 s**; the instrumented profile recorded
   12,001 boolean DataFrame filters and spent **10.00 s** cumulatively in
   `DataFrame.__getitem__`.
7. **Fix the bot performance overview's query fan-out and quadratic symbol
   aggregation.** With 5,000 trades across 5,000 symbols, CPU-only metrics took
   **6.08 s**; the same 5,000 trades across ten symbols took **0.10 s**. The
   command also repeats portfolio/trade queries and schema DDL across fresh Turso
   connections.

The existing July 4 hot-path work remains effective. A same-machine,
same-interpreter comparison found no material rolling regression between commit
`1f900b7` and current `main` (7.11 vs 7.22 s/run); current historical was faster
(2.58 vs 2.87 s/run). The older commit message's 4.92/1.62-second figures came
from a different environment and are not a valid regression baseline here.

## Scope and method

Reviewed paths:

- historical and rolling backtest orchestration;
- Pine expression evaluation, candidate selection, fills, portfolio accounting,
  equity reconstruction, metrics, and reporting;
- grid search, walk-forward optimization, Monte Carlo, and Strategy Lab;
- vectorbt parameter sweep and its indicator cache/chunking;
- earnings-drift and PEAD backtests;
- yfinance/FMP cache and provider boundaries;
- the bot's `/paper_metrics` performance overview, read-only in the separate bot
  checkout.

The primary profile used the repository's deterministic
[`profiling/harness.py`](../profiling/harness.py): 300 synthetic symbols, 1,306
bars per symbol including warm-up, a three-year/756-session rolling window, ten
slots, SMA crossover entry/exit expressions, price/ADV filters, stop/target,
commission, and fixed slippage. It generated 709 rolling trades and 14
historical trades. Provider-network time is deliberately absent from core CPU
profiles; the parquet benchmark measures cached I/O separately.

Environment:

| Item | Value |
|---|---:|
| Python | 3.11.15 |
| pandas | 2.3.3 |
| NumPy | 2.4.4 |
| vectorbt / vectorbt-rust | 1.0.0 / 1.0.0 |
| Visible CPUs | 4 |
| RAM | 7.8 GiB |
| Final test baseline (vectorbt installed) | 1,983 passed, 18 skipped in 105.14 s |

Profilers and artifacts:

- [`rolling flame graph`](../profiling/review_20260710/rolling_cpu_flame_20hz.svg)
- [`historical flame graph`](../profiling/review_20260710/historical_cpu_flame_20hz.svg)
- [`earnings flame graph`](../profiling/review_20260710/earnings_cpu_flame_20hz.svg)
- [`bot performance flame graph`](../profiling/review_20260710/bot_performance_cpu_flame.svg)
- [reproduction commands and harnesses](../profiling/review_20260710/README.md)

`py-spy` used 20 Hz sampling. Even that slowed the pandas workloads, so sampled
graphs are used for stack shape only. Absolute times and call counts come from
unprofiled timers, phase wrappers, `/usr/bin/time -v`, and `cProfile`. Phase
timers are inclusive; their percentages intentionally overlap.

## Baselines and scaling

### Controlled current-ref baseline

| Path | Workload | Time/run | Trades | Peak RSS including data/imports |
|---|---|---:|---:|---:|
| Rolling | 300 symbols, 3 years, top 10 | 7.22 s (3-run controlled comparison) | 709 | 202,004 KiB |
| Historical | 300 symbols, 3-year source panel, top 10 | 2.58 s (3-run controlled comparison) | 14 | 162,928 KiB |
| Rolling | 600 symbols, 3 years, top 10 | 10.50 s | 709 | 249,624 KiB |
| Earnings | 200 symbols, 2,400 events, 800 bars | 13.80 s | 2,400 | not isolated |
| vectorbt warm sweep | 100 symbols, 1,306 bars, 93 combos | 8.69 s | n/a | 1,392,872 KiB |

Single-run numbers varied by roughly 5–15% under shared-host scheduling. The
priority conclusions rely on repeated comparisons or large deltas, not on
sub-second differences.

### Universe scaling

| Symbols | Rolling time | Rolling trades | Historical time | Historical trades |
|---:|---:|---:|---:|---:|
| 50 | 4.04 s | 628 | 0.55 s | 0 |
| 100 | 5.16 s | 677 | 1.05 s | 3 |
| 300 | 7.82 s | 709 | 2.51 s | 14 |
| 600 | 10.50 s | 709 | 4.42 s | 21 |

Historical is predominantly universe/Pine bound and scales close to linearly.
Rolling has a sizeable trade/result fixed component plus a roughly linear
universe preparation component.

### Rolling horizon and slot scaling

| Years | Time | Trades |
|---:|---:|---:|
| 1 | 4.74 s | 255 |
| 2 | 6.03 s | 478 |
| 3 | 7.96 s | 709 |
| 5 | 10.45 s | 1,214 |

| Slots (`top`) | Time | Trades |
|---:|---:|---:|
| 1 | 3.47 s | 74 |
| 5 | 6.33 s | 349 |
| 10 | 8.21 s | 709 |
| 25 | 13.73 s | 1,761 |

The top/slot result shows why trade-driven work deserves priority: increasing
slots with a fixed universe and horizon nearly quadrupled runtime.

## Detailed findings

### F-01 — Volume-impact slippage is a no-op while its inputs are expensive

**Severity: critical correctness; high performance waste.**

[`FillModel._apply_slip`](../screener/backtester/fills.py#L94) accepts shares,
but [`entry_price`](../screener/backtester/fills.py#L116) and
[`exit_price`](../screener/backtester/fills.py#L140) do not pass them onward.
The default stays `0.0`. `VolumeImpactSlippage` explicitly returns zero when
shares are non-positive. Actual shares are only computed later in
[`Portfolio.open`](../screener/backtester/portfolio.py#L67).

Proof: on a 50-symbol/one-year rolling run, `VolumeImpactSlippage(k=1000.0)` and
zero fixed slippage produced an exactly equal trade ledger, equity curve, and
total return (`0.21683696243395456`). This is too strong to be a rounding effect.

At the same time, [`_make_slot_state`](../screener/backtester/core.py#L267)
unconditionally computes ADV and return volatility. In the instrumented rolling
profile, `_cached_trailing_liquidity` accumulated 1.68 s (9.7% of profiled time).
Those values cannot affect any fill from the shipped built-in models because
shares never reach the volume-impact component. A third-party custom model could
still choose to use ADV or volatility independently of shares.

**Improve:** make allocation and fill pricing share-aware. Entry has a circular
dependency (fill affects shares and shares affect impact), so use a documented
iteration: estimate shares at reference/fixed-cost price, apply impact, recompute
shares, and iterate to a small tolerance/cap. Exits and partial exits already have
an open `Position`; pass its actual closing shares. Add an engine-level test that
non-zero volume impact changes both entry and exit fills. Until that is correct,
skip ADV/sigma for known built-ins that cannot use it; after the fix, compute it
only when the selected model declares that it needs liquidity inputs. Preserve a
capability/fallback path for third-party custom models.

### F-02 — Full cache hits deserialize every ticker for every run

**Severity: P0 end-to-end bottleneck.**

[`YFinancePriceFetcher.fetch`](../screener/backtester/data.py#L524) calls
`pd.read_parquet` serially through `_load_cached` for every ticker. There is no
run/session-level memory panel. On 301 files × 1,306 rows:

| Source | Time/fetch |
|---|---:|
| In-memory stub | 0.248 s |
| Full parquet cache hit, run 1 | 5.539 s |
| Full parquet cache hit, run 2 | 5.692 s |
| Full parquet cache hit, run 3 | 6.674 s |

The warm median is 23× the in-memory path. This cost is outside the engine-only
flame graphs and can dominate historical runs.

It is multiplied in:

- grid search: every parameter combination calls a complete backtest;
- walk-forward: every train combination and every test window does the same;
- Strategy Lab: every selected strategy and comparison universe reruns through
  the same fetcher, which still reloads parquet;
- earnings batches: new fetcher instances and retained panels repeat cache work.

**Improve:** introduce an immutable `PricePanel`/`PanelPriceFetcher` scoped to a
CLI operation. Load/fetch once for the union of required symbols and the widest
date window, then return zero-copy or shallow date slices. For process workers,
initialize the panel once per worker or use Arrow/shared-memory backing; submit
only parameter dictionaries. Parallelize independent parquet reads only as a
fallback—avoiding the reads is the larger win.

Record provider, adjustment regime, interval, requested dates, symbol set, and
data freshness with the prepared panel. Optimization-result cache keys currently
do not contain provider/data/code fingerprints, so faster caching must not make
stale results harder to detect.

### F-03 — Exposure uses one pandas masked update per trade

**Severity: P0 CPU/algorithmic bottleneck; low implementation risk.**

[`metrics._exposure`](../screener/backtester/metrics.py#L106) creates a full
boolean index mask and performs `.loc[mask] += 1` for every trade. Complexity is
effectively O(trades × equity points) with high pandas construction/indexing
overhead.

Evidence:

- instrumented rolling profile: 2.873 s of a 17.256 s profile (16.6%);
- uninstrumented isolated call: 1.105 s for 709 trades × 756 sessions;
- vectorized difference-array PoC: 0.0155 s, **71× faster**;
- result: exactly equal exposure (`0.9985449735449736`).

**Improve:** vector-search all entry positions with `side="left"`, all exit
positions with `side="right"`, `np.add.at` +1/-1 into a length N+1 difference
array, then cumulative-sum once. This preserves current inclusive exit-day
behavior exactly. Reuse the same interval/event primitive for other occupancy
metrics rather than building additional masks.

### F-04 — Rolling preparation repeats pandas expression and filter work

**Severity: P1 core CPU; magnified by optimization workflows.**

The complete instrumented rolling profile (17.256 s) showed:

| Inclusive stack | Cumulative time | Share of profile |
|---|---:|---:|
| Day loop | 6.917 s | 40.1% |
| `_prepare_simulation` | 6.070 s | 35.2% |
| `_make_slot_state` | 4.665 s | 27.0% |
| `_assemble_results` | 4.233 s | 24.5% |
| Pine `evaluate` | 3.523 s | 20.4% |
| `compute_metrics` | 2.913 s | 16.9% |
| Entry-signal precompute | 2.522 s | 14.6% |
| Trailing liquidity | 1.682 s | 9.7% |
| Filter-signal precompute | 1.506 s | 8.7% |
| Frame-cache construction | 1.068 s | 6.2% |
| Candidate matrices | 0.917 s | 5.3% |
| Equity curve | 0.789 s | 4.6% |

The Pine evaluator repeatedly converts named columns with `astype(float)` and
rebuilds rolling subexpressions. The benchmark's `sma(close, 50)` appears in both
entry and exit ASTs. A semantic call-cache PoC found 214 hits among 1,328 call
nodes and was output-exact; measured whole-run benefit varied from 2% to 10%
because a Python dictionary wrapper also touched every call.

**Improve:**

1. Normalize a ticker frame's numeric columns once (`copy=False` when already
   float) and provide an evaluation context with cached name/indicator arrays.
2. Key rolling primitives by semantic expression `(function, source, length)`,
   not AST object identity or source-column metadata, and share the context
   between entry and exit evaluation for one frame.
3. Build filter and ranking arrays from the same prepared numeric columns.
   `_precompute_filter_signals` and `_build_rolling_candidate_matrices` currently
   repeat close/volume conversion and retain several pandas intermediates.
4. Keep the existing causal/full-frame evaluation guarantee and compare complete
   results bit-for-bit. Do not cache arbitrary strategy hooks whose output can
   depend on configuration or external data.

### F-05 — Historical selection is predominantly per-ticker Pine/pandas work

**Severity: P1 for large universes.**

Historical `cProfile` recorded 5.852 s instrumented:

| Inclusive stack | Cumulative time | Share |
|---|---:|---:|
| `select_candidates` | 3.382 s | 57.8% |
| Pine `evaluate` | 2.606 s | 44.5% |
| Entry filters | 1.215 s | 20.8% |
| Event simulation | 1.151 s | 19.7% |
| In-memory fetch/slicing | 0.589 s | 10.1% |

Historical already evaluates an entry AST once per ticker and reuses it for
reserve rotation. That previous optimization should remain. The next gain is
wide-panel/common-subexpression evaluation, not another whole-AST memo.

**Improve:** reuse the same prepared numeric frame and indicator cache proposed
for rolling. For compatible built-in Pine operations, evaluate one wide
date×ticker panel per unique indicator/window. Retain a per-ticker fallback for
heterogeneous columns, custom fundamentals, and strategy preparation hooks.

### F-06 — Grid and walk-forward multiply invariant work

**Severity: P0 workflow bottleneck.**

The CLI's default parameter grid is 4 stop values × 4 targets × 3 trailing stops
× 3 holds = **144 rolling backtests**, with `--workers 1` by default. On a small
50-symbol/one-year, eight-combination benchmark:

| Workers | Total | Effective time/combo | Speedup vs 1 |
|---:|---:|---:|---:|
| 1 | 13.42 s | 1.68 s | 1.00× |
| 2 | 7.31 s | 0.91 s | 1.83× |
| 4 | 5.93 s | 0.74 s | 2.26× |

Process parallelism helps but does not remove duplicated work. On the primary
300×3-year workload, a rough current-cache estimate is (7.2 s engine + 5.7 s
parquet) × 144 ≈ **31 minutes** before report overhead or live misses. This is an
extrapolation, not a live-provider benchmark.

Cold parallel grids are worse: each worker can issue identical provider requests
and write the same cache files. The full fetcher is also included in every task
argument. Cache persistence rewrites the entire growing JSON file after every
completed result, making large cache saves cumulative rather than append/batch
oriented.

Walk-forward calls that grid for every train window, then performs another full
rolling backtest for its test window. It does not preload the union date range or
share invariant preparation between overlapping windows.

**Improve:** split the engine API into:

```text
provider/cache -> immutable panel -> prepared signals/filters -> simulation -> metrics
```

Group parameter combinations by preparation-affecting fields. Stop, target,
trailing stop, costs, and often hold can reuse price data and entry/exit signal
arrays. Use a worker initializer to load a prepared panel once, send only compact
configuration deltas, flush the result cache in bounded batches, and include
data/code identity in its key. Walk-forward should fetch the widest union window
once and slice it in memory while preserving point-in-time membership and
fundamental lags.

### F-07 — vectorbt chunk sizing understates real memory

**Severity: P1 reliability/scale bottleneck.**

[`run_parameter_sweep`](../screener/backtester/vbt/sweep.py#L156) chooses a chunk
from only `close + price` tiled bytes and targets 1 GiB. It does not include:

- complete all-combination entry and exit panels built before chunking;
- vectorbt portfolio state, records, grouped value/cash/return arrays;
- MultiIndex/category factorization and result objects;
- Python, pandas, Numba, Rust extension, and source-panel memory.

Measured on 100 symbols × 1,306 bars × 93 combinations:

| Mode | Warm sweep | Peak RSS |
|---|---:|---:|
| Auto chunk (all 93) | 8.69 s | 1,392,872 KiB |
| Chunk size 16 | 9.03 s | 1,065,732 KiB |

The first process also spent 29.43 s importing/JIT-compiling the tiny warm-up;
subsequent process warm-up fell to 7–9 s through Numba caches.

The chunk-16 `cProfile` spent 9.622/10.901 s inside portfolio chunk metrics;
`Portfolio.from_signals` accounted for 2.871 s and the Rust signal simulator
2.202 s. Signal-panel construction was 0.926 s. Much of the remainder was grouped
value/returns work and repeated MultiIndex factorization.

**Improve:**

1. Stream entry/exit panels by combination chunk instead of materializing all
   combinations first; keep only the reusable `SignalCache` primitives.
2. Derive chunk size from an explicit memory budget/available memory with a
   conservative multiplier calibrated against RSS. Default well below 1 GiB.
3. For hard memory isolation, run chunks in recyclable worker processes so native
   allocations are returned when a worker exits.
4. Reduce per-chunk grouping setup; prebuild/reuse compact integer group codes if
   vectorbt accepts them. Compute required return/value arrays once and derive the
   requested metrics from that shared result.
5. Surface cold-start/JIT and warm-run times separately in CLI output. A fast
   sweep after a 30-second silent warm-up otherwise looks hung.

### F-08 — Earnings backtests scan whole frames about five times per event

**Severity: P1 CPU; P1/P2 provider latency depending market.**

For each event, `run_earnings_backtest`:

1. filters `bars.index <= earnings_date` in `_find_entry_exit`;
2. filters equality for entry and exit bars;
3. filters the prefix again in `price_momentum`;
4. filters the same prefix again in `volume_surge`.

On 2,400 events × 800 bars:

- unprofiled wall time: 13.80 s;
- instrumented wall time: 24.38 s;
- 12,001 boolean DataFrame gets: 10.00 s cumulative;
- `volume_surge`: 6.75 s;
- `price_momentum`: 5.22 s;
- `_find_entry_exit`: 3.72 s;
- `iterrows`: 1.01 s.

PEAD repeats per-event boolean filtering to find post-event bars and then calls
`get_indexer`, despite having a sorted `DatetimeIndex`.

The earnings "batch for RAM" loop still accumulates every returned frame into
one `price_data` dictionary, so batching limits request size but not retained
memory. India openscreener collection sleeps 0.1 seconds after every completed
future. At 500 symbols that imposes at least 50 seconds, while not actually
throttling submissions because all futures were already queued.

**Improve:** sort events by ticker; for each ticker, convert the index and OHLCV
columns once, vector-search all event positions with `searchsorted`, and compute
5/20-day returns and rolling volume baselines once as arrays. Access entry/exit
prices positionally. Process a price batch's events and discard the frames before
loading the next batch. Put rate limiting at request execution/submission, not in
the result-consumption loop. Apply the same indexed-position primitive to PEAD.

### F-09 — Result assembly and self-contained reporting are visible costs

**Severity: P2/P3, depending CLI use.**

Rolling result assembly took 4.233 s instrumented, dominated by metrics (2.913 s)
and then equity reconstruction (0.789 s). Fixing exposure removes most measured
metrics cost. Equity reconstruction is already substantially vectorized and is
not the first rewrite target.

For a 709-trade result:

| Output | Build time | Size |
|---|---:|---:|
| Tear sheet | 1.476 s | 5,351,126 bytes |
| Dashboard | 1.682 s | 5,195,480 bytes |
| Strategy Lab JSON payload | 0.088 s | 374,498 bytes |

Both HTML outputs inline the full Plotly runtime. The normal backtest CLI also
creates a temporary tear sheet by default unless CSV output is selected, so this
cost is part of perceived command latency, not merely an explicit export.

**Improve:** add a clear `--no-report`/text-only path, and offer a compact report
mode that references a versioned local/CDN Plotly asset when self-containment is
not required. Cache the JS string within a process. Strategy Lab should paginate
or virtualize large trade tables; its JavaScript currently flattens and renders
every trade row. The larger Lab win is still sharing price/preparation across
strategies.

### F-10 — Bot `/paper_metrics` fans out queries and has O(trades × symbols) CPU

**Severity: P0 in the bot repository for remote Turso/many symbols.**

This finding is against bot commit `3ff26d8`; no bot files are changed by this
review.

The command first calls `get_all_portfolios_status()` and filters by requested
name afterward. For each selected portfolio it then calls `compute_metrics()`
and `equity_curve()` separately. Both refetch the portfolio and complete trade
history. `PaperStore` opens a fresh client per operation, while each low-level
read reruns five schema/PRAGMA statements before its SELECT.

For P portfolios:

- named request: `1 + 2P + 4` fresh store connections before benchmark data;
- all portfolios: `1 + 6P` connections;
- with P=10, that is 25 connections / at least 150 SQL statements for one named
  overview, or 61 connections / at least 366 SQL statements for all, plus price
  and benchmark fetches.

CPU aggregation builds a unique ticker set, then rescans every trade for every
ticker. Synthetic local-store results:

| Trades | Unique symbols | Metrics CPU | Equity curve CPU |
|---:|---:|---:|---:|
| 1,000 | 10 | 0.013 s | 0.001 s |
| 1,000 | 1,000 | 0.191 s | 0.002 s |
| 5,000 | 10 | 0.099 s | 0.004 s |
| 5,000 | 5,000 | 6.078 s | 0.005 s |

The instrumented worst case made 25.03 million `dict.get` calls.

**Improve in the bot repo:** add one `performance_overview(name=None)` service
operation that selects the requested portfolio(s) first, uses one client/transaction,
fetches positions/trades once, computes metrics and curve from that shared data,
and groups per ticker in one `defaultdict` pass. Run schema migration once at
startup/version upgrade. Batch status/position queries, group benchmark requests
by market and earliest start date, and reuse returned benchmark bars.

### F-11 — Smaller secondary opportunities

**Severity: P3 unless a specific workload proves otherwise.**

- Candidate ranking sorts and materializes every eligible row whenever any slot
  is free, although it normally consumes at most the number of free slots. Use
  top-k/partial selection with deterministic tie behavior and an adaptive buffer
  for candidates whose entry fill fails.
- `_active_or_pending_tickers` is rebuilt before ranking and again while filling
  slots. Maintain a small active set for high-slot/high-churn runs.
- `Portfolio.get_position` scans open keys for a ticker. Maintain a direct
  ticker→FIFO-key index if pyramiding/high slot counts become material.
- Dividend equity reconstruction scans remaining dividend dates on every
  calendar point. A sorted dividend-event iterator makes this linear.
- Monte Carlo is a Python loop over iterations, but it was only 0.55 s for 709
  trades × 5,000 iterations and 1.33 s for 5,000 × 5,000. Chunked 2D NumPy could
  improve it, but it is below the items above and full vectorization can consume
  too much memory.

## Output-equivalent optimization ceiling

The local PoC applied three substitutions only during measurement:

- skip currently unused liquidity statistics;
- semantic Pine call-result cache;
- exact difference-array exposure.

Across repeated 300×3-year runs, the combined path was **1.41–1.44× faster** and
all trades, equity, benchmark, metrics, selections, and warnings were exact. The
individual liquidity result varied from 1.11× to 1.24×; semantic indicator memo
varied from 1.02× to 1.10× because its Python wrapper imposed lookup overhead.
This is a measured ceiling for these three changes, not a promise that their
production implementations will add linearly.

Combining prepared-panel reuse with that PoC suggests a cached single rolling
operation could move from roughly 13–14 seconds (5.7 s parquet + 7–8 s engine)
toward roughly 6 seconds before report rendering. That ~2× estimate is an
inference from independent measurements and must be confirmed end-to-end after
implementation.

## Recommended delivery sequence

### Phase 1 — Correctness and low-risk wins

1. Add an engine integration test proving volume-impact fills change with
   non-zero position size; fix the share-aware fill lifecycle.
2. Replace exposure masks with the exact difference-array algorithm and retain
   existing golden metric tests.
3. Gate ADV/sigma work on whether a now-correct slippage model needs it.
4. Add benchmark assertions based on operation counts/output identity, not strict
   wall-clock thresholds in normal CI.

### Phase 2 — Prepared data boundary

1. Introduce immutable fetched/prepared panel types.
2. Make grid, walk-forward, and Strategy Lab fetch the union panel once.
3. Split signal/filter preparation from execution/result assembly.
4. Initialize process workers with shared/read-only data; send parameter deltas.
5. Version optimization cache entries with data/provider/code identity and batch
   persistence.

### Phase 3 — Engine-specific work

1. Add Pine evaluation contexts and common-subexpression/indicator caching.
2. Stream vectorbt combination panels under an actual memory budget.
3. Convert earnings/PEAD event loops to positional arrays and true batch
   processing.
4. Add text-only/compact report modes and Lab trade virtualization.

### Separate bot PR

1. Move schema initialization out of every read.
2. Add a single batched performance-overview query/service.
3. Group per-symbol metrics in one pass and reuse trades for the equity curve.
4. Batch current prices and benchmark bars by market.

## Validation requirements for implementation PRs

- Run `uv run pytest` and retain all correctness/golden/reference tests.
- Compare `BacktestResult` field-by-field and curves with exact equality for pure
  performance refactors.
- Treat the volume-impact fix as an intentional numerical behavior change and
  add hand-computed entry/exit witnesses.
- Reconcile fixed-cost/stop regimes with existing vectorbt reference tests.
- Benchmark cold cache, warm parquet, prepared in-memory, and live-provider
  phases separately.
- Report peak RSS as well as wall time for vectorbt and process pools.
- Test grid cancellation/cache recovery and concurrent cold-cache behavior.
- For earnings, verify point-in-time boundaries after replacing boolean prefixes
  with positional search.

## What should not be rewritten first

The prior optimizations are visible and valuable: entry/exit evaluation is
already cached per ticker at the whole-AST level, frame primitives use NumPy in
the per-bar loop, candidate scans have array mirrors, and equity reconstruction
pre-aligns closes. The current-ref comparison did not reveal a material
post-July-4 regression. Replacing the day loop wholesale, moving immediately to
Rust, or micro-optimizing Pydantic models before fixing repeated I/O, exposure,
and workflow preparation would target smaller costs while leaving the main
latency and memory multipliers intact.
