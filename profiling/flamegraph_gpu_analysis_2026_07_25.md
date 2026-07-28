# Flamegraph-driven bottleneck analysis and GPU offload feasibility

> **⚠️ Superseded — this describes the engine *before* PRs #114–#117.**
> Its central finding (F-04, whole-panel Pine evaluation) has since shipped, and Pine
> evaluation is no longer the top bottleneck. See
> [`flamegraph_analysis_2026_07_28.md`](flamegraph_analysis_2026_07_28.md) for the
> re-run and the current ranking.
>
> The artifacts in `profiling/webview/` (`rolling.prof`, both flamegraphs, both
> `pstats_*.txt`) were **regenerated on 2026-07-28** and now reflect the post-#117
> engine, so they no longer match the numbers below. The profile this document was
> written against is preserved as `profiling/webview/rolling_pre_vectorization.prof`.
> Note also that the run below was taken under heavy CPU contention (§1), so its
> absolute seconds are inflated and not comparable to the 07-28 figures.

**Date:** 2026-07-25 · **Engine:** rolling backtester · **Source of truth:** `profiling/webview/rolling_pre_vectorization.prof`

## 1. Environment and method

Profiled the deterministic offline harness
`profiling/harness.py --path rolling --tickers 300 --years 3 --top 10 --repeat 2`
through cProfile into `profiling/webview/rolling.prof`: **7,171,660 function calls
(7,021,106 primitive) in 26.974 s instrumented** for 2 repeats. The harness runs an
SMA-crossover strategy (`crossover(close, sma(close,50)) and close > sma(close,20)`
entry, `crossunder(close, sma(close,50))` exit, stop 0.08, top-10) over 300 seeded
synthetic tickers (~1,306 bars each incl. warm-up, 756-session window) against a stub
fetcher — so the profile is pure engine CPU with **zero network/parquet I/O**. Two
flamegraphs back the ranking: `flamegraph_cprofile.svg` (call-count-weighted, the
authoritative artifact) and `flamegraph_pyspy.svg` (6,427 samples; corroborating but
noisier — 6.8% of its samples are `importlib` process start-up, so it is used only to
confirm relative shape). **Contention caveat:** a 280-run backtest sweep ran
concurrently on this 4-core box, so every wall-clock number is inflated. All ranking
claims below therefore rest on cProfile **call counts and cumulative-time share**
(relative, contention-invariant), not absolute seconds. Tools: `cProfile`/`pstats`
(with targeted `print_callers`/`print_callees`/`print_stats` cuts on `rolling.prof`),
both flamegraph SVGs, and one isolated `perf_counter` micro-benchmark (§2.3). This VM
has **no GPU** (no `nvidia-smi`, no `/dev/nvidia*`) — §3 assumes one is available per
the requested framing.

**Prior-work delta (verified against source, not just the July-10 review's claims).**
Two July-10 P0 findings have shipped and no longer appear in the top-40:

- **F-01** (share-aware slippage): `fills.py:124 entry_price` now computes `shares`
  from budget and passes them into `_apply_slip` (fills.py:152–160); `_make_slot_state`
  gates the expensive ADV/σ work behind `if fills.needs_liquidity_inputs`
  (`core.py:324`). The former dead `_cached_trailing_liquidity` cost is gone from the
  profile.
- **F-03** (exposure): `metrics.py:118 _exposure` is now `searchsorted` +
  `np.add.at` + one `cumsum` (no per-trade `.loc[mask] += 1`). It is absent from the
  top-40 — confirmed fixed.

**F-04** (Pine common-subexpression / whole-panel evaluation) has **not** shipped —
`pine.py` still has no cache/context abstraction (a `grep` for `cache|memo|Context`
matches only an unrelated comment) — and it remains the single largest bottleneck here.

## 2. Where the time goes

Cumulative-time shares of the 26.974 s profile (verified via `pstats`; inclusive stacks
overlap by construction). The distinction between **(a)** per-node/per-call Python +
pandas object overhead and **(b)** actual vectorizable float math is the crux of §3, so
it is called out per item.

| Rank | Function (verified file:line) | ncalls | cumtime | % | Nature |
|---|---|---:|---:|---:|---|
| 1 | Pine `evaluate` `pine.py:479` | 1,028 | 8.874 s | 32.9% | mostly (a) |
| — | ↳ `_precompute_entry_signals` `core.py:810` | 600 calls → 6.831 s | | 25.3% | prep phase |
| — | ↳ exit-signal eval in `_make_slot_state` `core.py:310` | 428 calls → 2.043 s | | 7.6% | memoized/ticker |
| 2 | `Series.__init__` `series.py:392` | 17,096 | 4.957 s | 18.4% | (a) |
| 3 | `generic.py:6485 astype` | 10,632 | 4.155 s | 15.4% | (a) |
| 4 | `_make_slot_state` `core.py:310` | 1,418 | 5.511 s | 20.4% | per-trade, mixed |
| 5 | `_construct_result` `series.py:6222` | 7,586 | 2.951 s | 10.9% | (a) wrapping (b) |
| 6 | `rolling.mean` `window/rolling.py:2216` | 2,232 | 2.233 s | 8.3% | **(b) — the real SMA** |
| 7 | `after_exits`→`DataFrame.__init__` (§2.3 bug) | 1,418 | 2.097 s | 7.8%\* | (a), avoidable |
| 8 | `frame()`/`_build_frame_cache` `core.py:151/110` | 1,418 / 428 | 2.047 s | 7.6% | (a) one-time/ticker |
| 9 | `_build_rolling_candidate_matrices` `rolling_candidates.py:101` | 2 | 1.867 s | 6.9% | whole-panel (a)+(b) |
| 10 | `build_equity_curve` `portfolio.py:327` | 2 | 1.667 s | 6.2% | already vectorized |
| 11 | `isinstance` (builtin) | 1,440,736 | 1.582 s | 5.9% | (a) dispatch |
| 12 | `pandas_dtype` `dtypes/common.py:1606` | 39,904 | 1.470 s | 5.5% | (a) dtype resolution |
| 13 | `entry_price`→`_resolve_entry_fill` `fills.py:124/37` | 1,418 | 0.756 s | 2.8% | (a) `bars.iloc`, §2.4 |

\* Item 7's 7.8% is the cProfile **cumtime attribution**, which over-states the
recoverable wall-clock — see §2.3.

### 2.1 Pine expression evaluation — 32.9%, the largest single cost

`pine.py:479 evaluate` (1,028 calls, 8.874 s cum) recursively walks the AST via
`pine.py:383 _eval` (7,112/1,028 calls) into `pine.py:438 _eval_call` (2,656/1,628,
6.784 s cum). `print_callers` confirms it is invoked from exactly two places:
`_precompute_entry_signals` (600 calls, 6.831 s — once per ticker × 2 runs) and
`_make_slot_state` (428 calls, 2.043 s — the **exit-signal cache miss** path, one eval
per *unique ticker*, correctly memoized in `_RunCaches.exit_signals`, **not** per trade).

So this is **not a per-bar Python loop** — each `evaluate()` is a legitimate
whole-history vectorized pandas op over ≤1,200 bars. The cost is **per-ticker
call-count amplification of pandas' fixed per-call overhead**. Inside `evaluate`:

- `window/rolling.py:2216 mean` (the actual `sma()` compute) — 2,232 calls, 2.233 s
  cum. **This is the only genuine float-math bucket, and it is 8.3%.**
- `pine.py:368 _series_from_name` — 3,256 calls; each unconditionally does
  `bars[name].astype(float)` (source lines 372/373/377/379), driving **3,256 of the
  10,632 `astype` calls (1.320 s cum)**. On the synthetic panel these columns are
  already `float64`, so this is a redundant full-copy + dtype-machinery pass.
- `series.py:6222 _construct_result` — 7,586 calls, 2.951 s cum — the Series object
  built after every `+ - * / > >= <= == != & |` op (reached via `_arith_method`,
  `_cmp_method` 3,264, `_logical_method` 2,840 = Pine's `and`). The underlying NumPy
  compare is nanoseconds; the 2.951 s is the **wrapping**, i.e. bucket (a).
- `pine.py:350 _crossover`/`:359 _crossunder` — two `.shift(1)` each plus a
  compare-mask; cheap per call but run once per ticker (up to 300×) rather than once
  panel-wide.

py-spy corroborates independently: `evaluate` (pine.py:491) is **15.39%** of samples at
one call site plus **7.14%** at another; `_eval` frames add 8.71% + 8.01%.

### 2.2 Generic pandas object-construction tax — spread across the profile

`Series.__init__` (17,096 calls, 4.957 s cum, 0.791 s tot). `print_callers`: 7,586 via
`_construct_result` (§2.1 arithmetic), 2,232 via `rolling._apply_series` (rolling
output wrapping), 2,018 via `dict_to_mgr`/`_init_dict` (DataFrame-from-dict), 1,200 via
`_precompute_filter_signals` (the `pd.Series(True, index=...)` seed). Around it:
`isinstance` (**1.44 M calls**, 1.582 s — pandas ABC `_instancecheck` 155,316 plus the
AST walker's own type chain), `pandas_dtype` (39,904, 1.470 s), `__setattr__` (57,500,
1.036 s), `sanitize_array` (17,950, 0.984 s), `__finalize__` (32,524, 0.726 s),
`is_extension_array_dtype` (10,632, 1.312 s). None of this is arithmetic — it is
Python-object and pandas-metadata bookkeeping that scales with **call count**, not array
size. There is even ~0.8 s in `warnings` context-manager churn (`simplefilter`/
`_add_filter`, ~36 k calls) from pandas' internal `catch_warnings` — pure overhead.

### 2.3 A concrete, fixable bug — eager default-argument `pd.DataFrame()`

`rolling_simulation.py:387`: `bars = self.bars_by_tv.get(ticker, pd.DataFrame())`.
`dict.get(key, default)` evaluates `default` **unconditionally**, so an empty DataFrame
is built and discarded on every candidate popped in the slot-refill loop — even though
`ticker` is essentially always present. `print_callees` on `after_exits` confirms **1,418
`frame.py:698 DataFrame.__init__` calls, 2.097 s cumtime**, and py-spy shows the
`__init__` frame independently.

**Right-sizing the payoff (correction to the prior draft).** The 2.097 s / 7.8% is
cProfile *cumtime*, which inflates high-call-count paths via per-subcall profiler tax. An
isolated `perf_counter` micro-benchmark on this (contended) box measured a bare
`pd.DataFrame()` at **477 µs**, so the true recoverable work is ≈ 1,418 × 477 µs ≈
**0.68 s contended, and less uncontended** — realistically **~2–3 % of an
un-instrumented run**, not 7.8%. It is still a genuine, zero-risk free win (behavior is
unchanged: the next line already treats a missing ticker the same as an empty frame), but
"7.8 % of runtime" overstates the wall-clock recoverable amount. Fix:
`bars = self.bars_by_tv.get(ticker)` then `if bars is None or bars.empty: continue`.

### 2.4 Per-trade slot setup — `_make_slot_state` (core.py:310)

1,418 calls (once per opened trade — 709 trades × 2 runs), 5.511 s cum (3.9 ms/call),
genuinely per-trade. `print_callees` breakdown:

- `core.py:151 frame()` — 2.047 s cum, essentially all from the **428 cache misses**
  that call `core.py:110 _build_frame_cache` (2.036 s) to convert OHLCV to NumPy once per
  ticker. Legitimate one-time-per-ticker cost, already an optimization.
- `pine.py:479 evaluate` (exit signals) — 2.043 s over 428 misses (§2.1).
- `fills.py:124 entry_price` → `fills.py:37 _resolve_entry_fill` — **0.756 s / 1,418
  calls**. Verified: line 48 still does `bars.iloc[entry_idx]["open"]` — a pandas
  scalar-row `Series` build per trade — instead of the `_FrameCache.open_arr` NumPy
  array `_make_slot_state` already holds. This is the **one per-trade hot path missed**
  when F-03's array-cache work landed on the exit side.

### 2.5 Per-bar exit scan — already vectorized, no longer a bottleneck

`core.py:484 _check_exit_at_bar` (13,680 calls, 0.221 s tot) and `core.py:860
_close_slot_at_day` (15,098 calls, 0.274 s tot, 1.266 s cum) read
`frame_cache.open_arr/high_arr/low_arr/close_arr` (plain `np.ndarray`) with
`np.searchsorted` day lookup rather than `bars.iloc[i]`/`get_loc`. Together ~0.5 s tot
for ~29 k calls — this is exactly the F-03 per-bar rewrite the July-10 review called for,
and it shows: no longer worth further optimization.

### 2.6 Whole-panel construction

`rolling_candidates.py:101 _build_rolling_candidate_matrices` (2 calls, 1.867 s cum)
builds `pd.DataFrame(entry_signals_by_tv)` from 300 per-ticker boolean Series (a
300-column union-index reindex) then applies membership/regime/blackout masks — a
once-per-run whole-panel batch op, cost dominated by aligning 300 heterogeneous Series.
`portfolio.py:327 build_equity_curve` (2 calls, 1.667 s) is confirmed, as July-10 found,
already event-list/vectorized — not a per-day Python calendar loop.

### 2.7 The split: object overhead vs. real math (decisive for §3)

Using **tottime** (function-body time, no double-counting), the clearly-numeric
primitives — `ufunc.reduce`, `numeric.roll`, `_clip`, `ndarray.copy`, `rolling.calc`,
`ndarray.searchsorted` — sum to **0.642 s = 2.4 % of the 26.97 s of tottime.** Even
generously counting the *entire* `rolling.mean` cumulative path (2.233 s) as "math", real
vectorizable float work is **< 10 %** of the profile. The remaining **~90 %** is
Python-level control flow plus pandas object/metadata construction and dispatch — and
even the small math bucket is fragmented into ~300 tiny per-ticker calls (2,232
`rolling.mean` over ≤1,200-row Series) rather than one batched panel op. **This is a
latency-bound Python-overhead workload, not a throughput-bound array-math one.**

## 3. GPU offload feasibility (assume a GPU is available)

The current VM has no GPU; the rest of this section assumes one is present and asks
whether offloading helps.

### 3.1 Per hot path

- **Pine evaluator (§2.1, 32.9 %).** *Not GPU-amenable as written.* It runs one
  whole-array op per AST node **per ticker** on ≤1,200-element arrays. A one-for-one
  cuDF/cuPy swap replaces a ~10–50 µs pandas C call with cuDF Python/Cython/RMM dispatch
  (~50–200 µs) + kernel launch (~5–20 µs floor) + a host↔device copy for a few-KB array
  — the launch/dispatch tax alone likely exceeds today's per-call cost, paid
  2,232 + 7,586 + 10,632 times. A naive per-ticker GPU port would very likely be
  **slower**.
- **Per-bar exit scan (§2.5).** *Not GPU-amenable, and already fast.* Each bar's
  stop/target/trail/time check depends on carried-forward per-slot state (`peak`,
  `stop_ref`) — an inherently sequential scan. The only parallel width is the ≤10
  concurrently open slots: three orders of magnitude short of hiding kernel-launch
  latency.
- **Per-trade slot setup / `entry_price` (§2.4).** *Not a GPU target* — per-trade,
  small, branchy, with a data dependency (tomorrow's refill depends on which slots freed
  today).
- **Candidate matrix (§2.6).** The one true whole-panel batch op (300 × ~756 ≈ 227 k
  cells, single-digit MB). Closest cuDF/cuPy candidate today, but far below the
  ~1–10 M-row regime where RAPIDS beats pandas; PCIe transfer + launch for a panel this
  small is comparable to the compute itself.
- **vectorbt's own engine** (used elsewhere per July-10 F-07) is already the
  "batch-everything-into-one-array-op" answer — but it JITs to CPU via Numba/Rust and has
  **no CUDA backend**. Adopting its pattern is a *CPU* win, not a GPU path.
- **numba.cuda** — feasible for a *batched* workload (the 180-combo grid search, the
  280-run sweep already on this box, or a 5,000-path Monte Carlo), where many independent
  single-path simulations run across CUDA threads (SIMT). That is fundamentally different
  from accelerating *one* backtest's per-bar loop: it needs the *outer* loop (params,
  paths, or a much larger universe) to supply the parallel width, not the inner per-bar
  logic.

### 3.2 What a GPU-worthwhile redesign requires

The precondition is **the same rewrite that makes it fast on CPU**: replace the
per-ticker interpreted AST walk with one **whole-panel vectorized evaluation** — batch
`sma`/`crossover`/`crossunder` across all 300 tickers as a single
`DataFrame.rolling(50).mean()` etc., collapsing ~2,200+ tiny calls into a handful (July-10
F-04, done for real, not as a memo wrapper). Only after that does arithmetic intensity
per call rise enough to discuss a GPU port — and even then the harness's own 300 × 756
panel (~227 k cells, single-digit MB) is small enough that a CPU-native batched
NumPy/pandas op finishes in low-single-digit ms, likely faster than a GPU round trip.

### 3.3 Rough speedup range and reasoning

- **Naive GPU port of the current per-ticker loop:** ~**0.3–0.7×** (slower) — launch/
  dispatch overhead × thousands of tiny calls, plus PCIe round trips.
- **CPU-side batched whole-panel rewrite (no GPU):** ~**3–8×** on the Pine share
  (collapsing ~300× per-ticker call overhead into panel-wide calls). Since Pine is
  ~33 % of the profile, plausibly **1.5–3× whole-engine**, above the July-10 measured
  1.4× ceiling (which excluded this rewrite).
- **That batched panel then on GPU at 300-symbol / 3-year size:** ~**flat, plausibly
  still a net loss** — MB-scale data and tens-of-MFLOP compute sit below cuDF/RAPIDS
  break-even.
- **GPU at much larger scale** (10 k+ symbols, minute/tick history → 10⁷–10⁸ elements,
  or hundreds of param combos / Monte-Carlo paths as one SIMT launch): plausibly
  **5–50×** over a CPU batched implementation — but that is a *different, larger*
  workload than the one profiled.

### 3.4 Verdict

**GPU is not worth it for this backtester as currently architected.** The profile is
latency-bound Python/pandas object overhead inside per-ticker / per-trade / per-bar
control-flow-heavy loops — not throughput-bound array math (real float work is
< 10 %, §2.7) — and the one existing batch op (candidate matrix) is MB-scale, far below
GPU break-even. A naive port would likely be *slower*. **Precondition for revisiting:**
first do the whole-panel vectorized Pine rewrite (§4.1) — the same change needed for a
real *CPU* speedup — then reconsider GPU only for a workload with genuine parallel width
(much larger universes, or batching many independent backtests — grid search,
walk-forward, Monte Carlo, the 280-run sweep pattern — into one kernel via numba.cuda /
cuDF), never for one backtest's per-bar engine loop.

## 4. CPU-side wins that dominate GPU

Ranked by payoff in this profile; cross-referenced to July-10.

### 4.1 Batch Pine evaluation across the whole ticker panel — implements F-04 (biggest)

Rewrite `sma`/`ema`/`rsi`/`crossover`/`crossunder` to operate on a `DataFrame` with one
column per ticker (~756 × 300) instead of once per ticker inside the AST walk. This
collapses the 2,232 `rolling.mean`, 3,256 `_series_from_name`, and 7,586 arithmetic
dispatches (§2.1) — each paying full Series/dtype/`__finalize__` overhead on a ≤1,200-row
array — into a handful of panel-wide calls. **Single largest opportunity: Pine is 32.9 %
of cumtime and most of it is per-call overhead, not the `rolling.mean` math (§2.7).**
July-10's F-04 PoC (an AST-node result cache) only measured 2–10 % because it didn't
reduce per-ticker array size or call *shape*; a real panel-vectorized rewrite cuts call
count by the ticker-count factor. Normalizing the panel to float once up front also
removes the redundant `.astype(float)` re-casts in `_series_from_name` (§2.1).

### 4.2 Route `entry_price`/`_resolve_entry_fill` through the frame cache (§2.4)

`fills.py:37–64 _resolve_entry_fill` still does `bars.iloc[entry_idx]["open"]` even
though `_make_slot_state` already holds a `_FrameCache` with `open_arr`/`close_arr`/
`low_arr`. Thread the cache in (as the exit path already does). **~0.756 s / 1,418
calls**; mechanical, low-risk, closes the last `bars.iloc[]` on the trade-open path.

### 4.3 Fix the eager-default `pd.DataFrame()` (§2.3)

`rolling_simulation.py:387` → `self.bars_by_tv.get(ticker)` with the existing
`bars is None or bars.empty` guard. **Recoverable ~0.7 s (contended); zero-risk,
one-line, no behavior change.** Highest ROI-per-line, though smaller in absolute terms
than the 7.8 % cumtime figure implies (§2.3).

### 4.4 F-01 / F-03 already landed — no action

Share-aware slippage (F-01) and the exposure difference-array (F-03) are verified fixed
in source and absent from the top-40. Noted so the next review does not re-flag them.

### 4.5 Diffuse pandas-construction tax shrinks with 4.1

The `isinstance` (1.44 M), `pandas_dtype` (39,904), `__setattr__` (57,500),
`sanitize_array` (17,950), `__finalize__` (32,524), and `warnings` churn (§2.2) are
reached almost entirely through the Pine evaluator and per-ticker construction paths
targeted by 4.1–4.2. Not worth chasing independently — they fall roughly proportionally
once call counts drop. Re-profile after 4.1.

## 5. Bottom line

- The profile is dominated by **Python/pandas object-construction and dispatch
  overhead** — real vectorizable float math is only **~2–10 %** (§2.7). It is
  **latency-bound**, not throughput-bound.
- **Pine evaluation is #1 at 32.9 % cumtime** (`pine.py:479 evaluate`), overwhelmingly
  per-ticker call overhead rather than the `rolling.mean` SMA it computes.
- **GPU offload is not worth pursuing for this engine as architected** — the hot paths
  are sequential per-bar/per-trade state machines (no parallel width) or per-ticker calls
  on arrays too small to clear PCIe/kernel-launch overhead; a naive port would likely be
  *slower*. The precondition for GPU ever mattering is the whole-panel vectorized rewrite
  — which is itself the CPU win.
- **Highest-ROI CPU fix: batch Pine evaluation across the panel (F-04, §4.1)** —
  plausibly 1.5–3× end-to-end. Then two cheap mechanical wins: route `entry_price`
  through the frame cache (§4.2, ~0.76 s) and drop the eager `pd.DataFrame()` default
  (§4.3, ~0.7 s, one line).
- **Corrections to the prior draft:** (a) its headline "7.8 % of runtime" for the
  `pd.DataFrame()` bug is cProfile cumtime and over-states the recoverable wall-clock —
  the real figure is ~0.7 s / ~2–3 % (§2.3); (b) it labeled `_construct_result` as
  "2.485 s cum" — its own cumtime is **2.951 s** (2.485 s is the Series-`__init__`-under-
  it slice). F-01/F-03-shipped and F-04-not-shipped findings are confirmed correct.
