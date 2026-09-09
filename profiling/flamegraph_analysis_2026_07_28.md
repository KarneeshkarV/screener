# Flamegraph re-run after the vectorization work (2026-07-28)

**Engine:** rolling backtester · **Source of truth:** `profiling/webview/rolling.prof`
**Supersedes:** [`flamegraph_gpu_analysis_2026_07_25.md`](flamegraph_gpu_analysis_2026_07_25.md),
which profiled the engine *before* PRs #114–#117 landed. The pre-vectorization
profile is preserved as `profiling/webview/rolling_pre_vectorization.prof` so the
older document's claims stay checkable.

## 1. Method, and why the numbers differ from the 07-25 run

Same harness, same arguments, same seed:

```
profiling/harness.py --path rolling --tickers 300 --years 3 --top 10 --repeat 2
```

300 seeded synthetic tickers (1,306 bars each, 756-session window), SMA-crossover
entry/exit, stop 0.08, top-10, stub fetcher — pure engine CPU, zero I/O.

Two things changed about the *measurement*, independent of the code:

- **The 07-25 run was taken under heavy contention** (a 280-run backtest sweep was
  running on the same 4-core box), which is why it reported 26.974 s instrumented.
  This run was taken with the box near-idle and `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
  **Absolute seconds are therefore not comparable across the two documents.**
- To get a valid delta, the pre-vectorization tree (`98f8ea8`, the merge-base of this
  PR) was re-profiled *now*, on the same box, in the same conditions. Every
  before → after number below comes from that pair, not from the 07-25 write-up.

Wall clock, uninstrumented, 3 interleaved A/B trials (before/after alternating so
residual background load hits both arms equally):

| arm | trial 1 | trial 2 | trial 3 | mean |
|---|---:|---:|---:|---:|
| before (`98f8ea8`) | 4.147 s | 3.695 s | 4.591 s | **4.14 s/run** |
| after (`main` @ `d2382eb`) | 2.457 s | 2.121 s | 2.634 s | **2.40 s/run** |

**≈1.73× faster, and both arms produce 709 trades and 0 warnings** — the speedup is
free of behavioural drift on this harness, consistent with the byte-identical
14-strategy equivalence checks run against real cached parquets when each PR landed.

Instrumented totals: **7,171,842 calls / 16.577 s → 4,332,144 calls / 10.839 s**
(−40% calls, −35% instrumented time). `isinstance` alone went 1,440,739 → 719,961.

> **Caveat on generality.** The harness panel is 300 tickers that *all share one
> index*, which is the best possible case for the exact-index grouping in
> `evaluate_panel`. Real universes are raggeder (IPOs, halts, mixed calendars), and
> the same change measured ~1.55× on real cached parquets. Treat 1.73× as the
> ceiling, ~1.5× as the realistic figure.

## 2. What shipped, and what it actually bought

| PR | change | measured effect |
|---|---|---|
| #114 | whole-panel Pine evaluation (**F-04**) | `evaluate` 1,028 calls / 4.70 s → `evaluate_panel` 4 calls / 1.40 s. `_series_from_name` 3,256 → 12 calls; `_crossover` 600 → 2 |
| #115 | drop eager `pd.DataFrame()` defaults (**§2.3 of the old doc**) | `after_exits` 5.83 s → 1.61 s; `DataFrame.__init__` 2,042 → 44 calls; `dict_to_mgr` 2,032 → 14 |
| #116 | index dtype in the panel group key | correctness only (tz-aware/naive indexes silently relabelled); no perf delta |
| #117 | `_build_frame_cache` via numpy | 1.36 s → 0.71 s |
| #114 | `arrays=` threading into `fills.entry_price` (**§2.4**) | 0.53 s → 0.05 s (~10×) |

Downstream of those, the pandas object tax the old doc called out in §2.2 fell across
the board: `Series.__init__` 17,096 → 4,714 calls (2.99 → 0.90 s), `astype`
10,632 → 4,640 (2.32 → 1.18 s), `pandas_dtype` 39,904 → 14,720, `_construct_result`
7,586 → 2,446.

**The old §2.1 headline no longer holds.** Pine evaluation was 32.9% of the profile
and the single largest cost; it is now 12.9% and third. F-04 is closed.

## 3. Where the time goes now

Cumulative-time share of the 10.839 s instrumented profile, own code only. Inclusive
stacks overlap by construction — `_precompute_entry_signals` and
`prewarm_exit_signals` are the two callers of `evaluate_panel`, so their times
contain it.

| Rank | Function | ncalls | cumtime | % | before | status |
|---|---|---:|---:|---:|---:|---|
| 1 | `_precompute_filter_signals` `core.py:858` | 2 | 2.318 s | 21.4% | 2.04 s | **untouched — now the top target** |
| 2 | `_assemble_results` `rolling_simulation.py:448` | 2 | 1.808 s | 16.7% | 1.72 s | untouched |
| 3 | `after_exits` `rolling_simulation.py:352` | 1,512 | 1.613 s | 14.9% | 5.83 s | improved 3.6× |
| 4 | `evaluate_panel` `pine.py:710` | 4 | 1.402 s | 12.9% | 4.70 s¹ | improved 3.4× |
| 5 | `build_equity_curve` `portfolio.py:327` | 2 | 1.117 s | 10.3% | 1.06 s | untouched |
| 6 | `_build_rolling_candidate_matrices` `rolling_candidates.py:101` | 2 | 1.098 s | 10.1% | 1.03 s | untouched |
| 7 | `_make_slot_state` `core.py:337` | 1,418 | 1.078 s | 9.9% | 3.72 s | improved 3.5× |
| 8 | `_precompute_entry_signals` `core.py:838` | 2 | 1.061 s | 9.8% | 3.54 s | improved 3.3× |
| 9 | `process_exits_for_day` `day_loop.py:89` | 1,512 | 1.014 s | 9.4% | 0.94 s | untouched |
| 10 | `_close_slot_at_day` `core.py:892` | 15,098 | 0.945 s | 8.7% | 0.88 s | untouched |

¹ as `evaluate`, the per-ticker entry point that `evaluate_panel` replaced.

The shape of the profile has inverted: **everything still expensive is code the
vectorization work never touched.** Items 1, 2, 5, 6, 9 and 10 are unchanged in
absolute terms (their small +/- deltas are run-to-run noise), and now dominate purely
because what surrounded them got cheaper.

py-spy corroborates (2,647 samples @250 Hz, down from 6,427 — the run is shorter):
`run_rolling_backtest` 36.2% + 13.9% across two call sites, `_prepare_simulation`
11.2% + 9.2%, `run_day_loop` 9.0%, `_precompute_entry_signals` 8.0%,
`_assemble_results` 6.9%, `after_exits` 5.8%. Note that **importlib start-up is now
15.8% of py-spy samples** (it was 6.8%): the fixed ~0.9 s import cost did not change,
but engine time halved around it. Discount it as before.

## 4. Remaining targets, sized

### 4.1 `_precompute_filter_signals` — 21.4%, and the only clean win left

`core.py:872` is still a per-ticker Python loop doing, 300 times, exactly what the
Pine evaluator used to do 300 times: `astype(float)` two columns, seed a
`pd.Series(True, index=...)`, run a rolling mean, and `&=` the masks. It never went
through the Pine path, so #114 passed it by entirely.

A column-wise rewrite using the same exact-index grouping as `evaluate_panel` was
prototyped and measured on the harness panel:

```
current  median 387.3 ms
panel    median 193.3 ms      → 2.00×,  0 / 300 tickers mismatched (bit-exact)
```

**2.0×, not the 4× estimated earlier** — that earlier figure came from a looser
prototype that skipped the per-ticker Series unpacking the real signature requires.
At 2 calls per run this is ~0.19 s off a 2.40 s run, so **~8% end-to-end**, which is
still the largest single item on the board and is a near-mechanical port of logic
already proven in `pine.py`.

### 4.2 `_assemble_results` — 16.7%, uninvestigated

Second-largest and never profiled in detail. `rolling_simulation.py:466–471` already
uses `searchsorted` per trade to build the calendar, then hands off to
`build_equity_curve` (item 5, 1.117 s), so the two overlap. Worth a targeted
`print_callees` pass before assuming there is a win here.

### 4.3 `_close_slot_at_day` — 15,098 calls

The highest call count in own code and 0.197 s *tottime* (the largest own-code
tottime in the profile). Per-bar, per-slot; a candidate for the same treatment the
exit scan already got.

### 4.4 Not worth it

`Portfolio._active_keys`' O(n) scan, flagged earlier, does not appear anywhere near
the top-40 at this universe size. Leave it.

## 5. GPU verdict — unchanged, and now more firmly negative

The old §3 concluded a GPU port was not worth it as architected, and recommended the
CPU-side whole-panel rewrite first. That rewrite has now shipped, and it removed the
overhead a GPU would have been hiding rather than fixing:

- Genuine float math (`rolling.mean`) is **0.83 s of 10.839 s — 7.6%**. Everything
  else is Python control flow, pandas dispatch, and object construction, none of which
  a GPU addresses.
- Working-set size is unchanged and MB-scale (300 × 1,306 × 5 float64 ≈ 16 MB), well
  below the cuDF break-even.
- Halving engine time made the fixed ~0.9 s interpreter/import cost a *larger* share
  of any single run, which further erodes the ceiling on device-side wins.

The conclusion the old document reached for the *outer* loop still stands and is the
only GPU-shaped opportunity here: parameter sweeps, grids and Monte Carlo are
embarrassingly parallel across independent backtests. That is a scheduling problem,
not a kernel problem — and at 2.40 s/run, 4-way CPU process parallelism is the
cheaper first move.
