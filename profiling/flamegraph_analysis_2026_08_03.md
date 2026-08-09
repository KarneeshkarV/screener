# Flamegraph re-run after the rolling-backtest acceleration (2026-08-03)

> **Superseded 2026-08-09.** [`flamegraph_analysis_2026_08_09.md`](flamegraph_analysis_2026_08_09.md)
> re-measures the engine after rebasing onto current `main` (`58ce9ba`) with
> `pandas==2.3.3`/`numpy==2.4.4` pinned. Its 08-03 numbers were taken on a
> 16-core host before PR #96 landed; the new document profiles on a 2-core box
> with the deps the code was built against. This document stays as the record
> of the #114–#119 state.

**Engine:** rolling backtester · **Source of truth:** `profiling/webview/rolling.prof`
**Supersedes:** [`flamegraph_analysis_2026_07_28.md`](flamegraph_analysis_2026_07_28.md),
which profiled the engine after PRs #114–#117. Since then #118 (price/ADV filters
across the ticker panel) and #119 (rolling-backtest and parameter-sweep
acceleration) landed, so the whole profile is re-measured again. The pre-#114
tree is still preserved as `profiling/webview/rolling_pre_vectorization.prof`.

## 1. Method, and why the numbers differ from the 07-28 run

Same harness, same arguments, same seed:

```
profiling/harness.py --path rolling --tickers 300 --years 3 --top 10 --repeat 2
```

300 seeded synthetic tickers (1,306 bars each, 756-session window), SMA-crossover
entry/exit, stop 0.08, top-10, stub fetcher — pure engine CPU, zero I/O.

Two things changed about the *measurement*, independent of the code:

- **The box is different.** The 07-28 numbers came from a 4-core host under load;
  this run is on a 16-core host with `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
  **Absolute seconds are therefore not comparable across documents.**
- To keep the delta valid, both arms were re-profiled *now*, on this box,
  interleaved. `before` is `98f8ea8` (the merge-base, pre-#114), `after` is
  current `main` at `572d47b` (includes #114–#119). The old `rolling.prof`
  remains as the source for the 07-28 write-up; the new one replaces it.

Wall clock, uninstrumented, 3 interleaved A/B trials:

| arm | trial 1 | trial 2 | trial 3 | mean |
|---|---:|---:|---:|---:|
| before (`98f8ea8`) | 0.590 s | 0.616 s | 0.623 s | **0.61 s/run** |
| after (`main` @ `572d47b`) | 0.317 s | 0.318 s | 0.317 s | **0.32 s/run** |

**≈1.92× faster, and both arms produce 709 trades and 0 warnings** — the
speedups from #114–#119 are free of behavioural drift on this harness.

Instrumented totals: **7,171,738 calls / 3.070 s → 2,948,939 calls / 1.291 s**
(−59% calls, −58% instrumented time). `isinstance` went 1,440,739 → 451,016.

> **Caveat on generality.** The harness panel is 300 tickers that *all share one
> index*, the best possible case for the exact-index grouping in the panel
> evaluator. Real universes are more ragged (IPOs, halts, mixed calendars). The
> before→after deltas still hold directionally on real cached parquets, but treat
> the absolute seconds here as the ideal-case floor.

## 2. What shipped since the 07-28 write-up, and what it bought

| PR | change | measured effect |
|---|---|---|
| #118 | price/ADV filters evaluated across the ticker panel | `_filter_signals_for_group` `core.py:921` — 2 calls / 0.115 s; filters stop paying per-ticker pandas dispatch |
| #119 | whole-run preparation split + fingerprint caching in `rolling_simulation.py`; `evaluate_panel_many` in `pine.py` | preparation phase 0.789 s inclusive; `_precompute_filter_signals` 2.318 s → 0.127 s (the old §4.1 target, closed) |
| #119 | numpy-array preallocation in `_build_rolling_candidate_matrices` | 1.098 s → 0.143 s (~7.7×) |

The old doc's §4.1 recommendation — apply the same exact-index panel grouping to
`_precompute_filter_signals` that `evaluate_panel` already used — was shipped as
`_filter_signals_for_group`. That target is closed.

## 3. Where the time goes now

Cumulative-time share of the 1.291 s instrumented profile, own code only.
Inclusive stacks overlap by construction: `_prepare_simulation` is the caller of
the panel evaluator, so its time contains `evaluate_panel_many`.

| Rank | Function | ncalls | cumtime | % |
|---|---:|---:|---:|---:|
| 1 | `_prepare_simulation` `rolling_simulation.py:159` | 2 | 0.789 s | 61.2% |
| 2 | `run_day_loop` `day_loop.py:143` | 2 | 0.327 s | 25.4% |
| 3 | `evaluate_panel_many` `pine.py:814` | 2 | 0.239 s | 18.5% |
| 4 | `after_exits` `rolling_simulation.py:458` | 1,512 | 0.205 s | 15.9% |
| 5 | `_assemble_results` `rolling_simulation.py:554` | 2 | 0.158 s | 12.2% |
| 6 | `_build_rolling_candidate_matrices` `rolling_candidates.py:101` | 2 | 0.143 s | 11.1% |
| 7 | `_make_slot_state` `core.py:351` | 1,418 | 0.136 s | 10.6% |
| 8 | `_precompute_filter_signals` `core.py:878` | 2 | 0.127 s | 9.9% |
| 9 | `process_exits_for_day` `day_loop.py:89` | 1,512 | 0.120 s | 9.3% |
| 10 | `_filter_signals_for_group` `core.py:921` | 2 | 0.115 s | 8.9% |
| 11 | `_close_slot_at_day` `core.py:964` | 15,098 | 0.112 s | 8.6% |
| 12 | `build_equity_curve` `portfolio.py:327` | 2 | 0.111 s | 8.6% |

The pendulum swung again: `_prepare_simulation` — the one-off set-up that runs
`evaluate_panel_many`, candidate-matrix building and the filter signals — is now
61.2% of the engine, and inside it nothing is a single dominant leaf. The old
"top target" (`_precompute_filter_signals`) fell from 21.4% to 9.9% of the
profile.

py-spy corroborates but is now less informative: 285 samples @250 Hz (down from
2,647 — the run is ~8× shorter), and **importlib start-up is 48.2% of samples**
(up from 15.8%): the fixed ~0.9 s import cost did not change, engine time shrank
around it. Discount it when reading shares.

## 4. Remaining targets, sized

### 4.1 `_prepare_simulation` — 61.2%, the whole preparation phase

`prepare_rolling_backtest` (`rolling_simulation.py:620`) now fingerprints its
inputs and caches prepared state per config, which is what makes parameter
sweeps fast (the #119 headline). On a single fresh run, though, the *preparation
itself* is the biggest cost, split across:
`evaluate_panel_many` (0.239 s), `_build_rolling_candidate_matrices` (0.143 s),
`_precompute_filter_signals` (0.127 s), `_filter_signals_for_group` (0.115 s)
and `_build_frame_cache` (0.092 s). No single 20% leaf — the win is additive,
not one clean swap, and the preparation is already amortised across sweeps by
the fingerprint cache.

### 4.2 `_assemble_results` — 12.2%, uninvestigated

Unchanged since 07-28 and never profiled in detail. `rolling_simulation.py:554`
uses `searchsorted` per trade to build the calendar, then hands off to
`build_equity_curve` (item 12, 0.111 s), so the two overlap. Worth a targeted
`print_callees` pass before assuming there is a win here.

### 4.3 `_close_slot_at_day` — 15,098 calls

Still the highest call count in own code and 0.028 s tottime. Per-bar, per-slot;
a candidate for the same panel treatment the exit scan already got. At 8.6%
cumulative this is now a smaller relative target than it was at 07-28.

### 4.4 Not worth it

`Portfolio._active_keys`' O(n) scan still does not appear near the top-40 at
this universe size. Leave it.

## 5. GPU verdict — even more firmly negative

The 07-28 verdict already said no; the #118/#119 work removed the remaining
per-ticker overhead a GPU would have been hiding, so the case only got stronger:

- Genuine float math (`rolling.mean`) is now **0.067 s of 1.291 s — 5.2%** (was
  7.6%). Everything else is Python control flow, pandas dispatch, and object
  construction, none of which a GPU addresses.
- Working-set size is unchanged and MB-scale (300 × 1,306 × 5 float64 ≈ 16 MB),
  well below the cuDF break-even.
- Engine time halved again, so the fixed ~0.9 s interpreter/import cost is an
  even larger share of any single run, further eroding device-side ceilings.

The one GPU-shaped opportunity is unchanged: parameter sweeps, grids and Monte
Carlo are embarrassingly parallel across independent backtests — a scheduling
problem, not a kernel problem. And at **0.32 s/run** a single roll now costs
almost nothing; the fingerprint cache in #119 already makes sweeps fast, so CPU
parallelism is the cheaper and simpler first move.
