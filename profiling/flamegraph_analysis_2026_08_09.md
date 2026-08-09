# Flamegraph re-run after rebasing onto current main (2026-08-09)

**Engine:** rolling backtester · **Source of truth:** `profiling/webview/rolling.prof`
**Supersedes:** [`flamegraph_analysis_2026_08_03.md`](flamegraph_analysis_2026_08_03.md).
This re-run measures the branch from PR #109 (`analysis/backtest-findings`) after
rebasing it onto current `main` (`58ce9ba`), and answers one question the 08-03
write-up could not: **does the rebase itself regress the engine?**

Answer up front: **no.** On identical dependency versions the rebased code is
statistically indistinguishable from the pre-rebase base, and a wall-clock
regression that *did* appear was traced, via a commit bisect, to a dependency
bump pulled in by PR #96 (`pandas 2.3.3 → 3.0.5`, `numpy 2.4.4 → 2.4.6`). PR #96
was a code-removal commit whose lockfile regeneration silently upgraded the
math stack. This PR pins `pandas==2.3.3` and `numpy==2.4.4` so the measured
numbers are reproducible and the ~27% wall-clock penalty is not shipped.

## 1. Method

Same harness, same arguments, same seed as the 08-03 run:

```
profiling/harness.py --path rolling --tickers 300 --years 3 --top 10 --repeat 2
```

300 seeded synthetic tickers (1,306 bars each, 756-session window), SMA-crossover
entry/exit, stop 0.08, top-10, stub fetcher — pure engine CPU, zero I/O.

**Box:** a 2-core GCP e2-class VM (idle, load ~0), `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
All three arms were synced with the **same** pinned dependency set
(`numpy==2.4.4`, `pandas==2.3.3`) and the **same** harness file (md5-identical),
each in its own worktree with the editable-install `.pth` trap verified
(`screener.__file__` points inside the worktree). Wall-clock trials were
interleaved: before → prerebase → after, repeated.

**Arms:**
- `before` — `98f8ea8`, the merge-base pre-#114 (the "before" arm of the 07-28
  and 08-03 write-ups; preserved as `rolling_pre_vectorization.prof`).
- `prerebase` — `572d47b`, the base the PR sat on when the 08-03 numbers were
  taken. This is "after" in that document.
- `after` — `58ce9ba`, current `main`, which the PR now sits on after rebasing.

The 08-03 numbers came from a 16-core host; absolute seconds are therefore not
comparable across documents. What matters here is the **within-batch,
interleaved** comparison on one box.

## 2. Wall clock: before vs prerebase vs after (identical deps)

5 interleaved trials, per-run means:

| arm | trial 1 | trial 2 | trial 3 | trial 4 | trial 5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| before (`98f8ea8`) | 2.261 | 2.225 | 2.160 | 2.315 | 2.213 | **2.235 s** |
| prerebase (`572d47b`) | 1.793 | 1.804 | 1.801 | 1.788 | 1.898 | **1.817 s** |
| after (`58ce9ba`) | 1.807 | 1.776 | 1.787 | 1.787 | 1.777 | **1.787 s** |

- before → after: **2.235 → 1.787 s/run, ≈1.25×** — the #114–#119 vectorization
  work, now measured head-to-head against its own merge-base on one box.
- prerebase → after: **1.817 → 1.787 s/run** — the rebase is flat (≈1.7%
  faster, within trial noise). **The rebase did not regress the engine.**

Instrumented totals (cprofile, engine-only, repeat=1):

| arm | calls | instrumented |
|---|---:|---:|
| before (`98f8ea8`) | 3,992,079 | 4.739 s |
| after (`58ce9ba`) | 2,454,463 | 2.984 s |

(−39% calls, −37% instrumented time; both arms produce **709 trades, 0 warnings**).

## 3. The apparent regression, and the commit bisect

The PR's original 08-03 numbers (`572d47b` at 0.32 s/run) predate PR #96. When
the rebase pulled in `main@58ce9ba`, a naive re-measure on the *as-shipped*
lockfile showed `after` at **2.276 s/run vs prerebase 2.049 s/run** — a ~11%
regression. That is what motivated this investigation.

A 3-trial commit bisect of the four commits between `572d47b` and `58ce9ba`
pointed at `9fe7ddf` (PR #96, "retire vbt-sweep and backtest-lab"):

| commit | PR | mean s/run |
|---|---:|---:|
| `ef29af0` | #123 provider seam | 2.065 |
| `42e07b1` | #108 minervini | 2.029 |
| `9fe7ddf` | **#96 retire vbt-sweep** | **2.242** |
| `58ce9ba` | #124 diff stargy | 2.215 |

PR #96's diff is almost entirely deletions (8,104 LOC removed) and cosmetics —
but its lockfile regeneration dropped the optional `vectorbt` extra and in doing
so bumped `numpy 2.4.4 → 2.4.6` and `pandas 2.3.3 → 3.0.5`. The 2×2 matrix
(code × deps) isolates the effect:

| code | pandas 2.3.3 | pandas 3.0.5 |
|---|---:|---:|
| prerebase (`572d47b`) | 2.049 s | 2.036 s |
| after (`58ce9ba`) | **1.810 s** | 2.276 s |

The after code is fastest on the deps it was built against (1.810 s, and 1.787 s
in the fuller interleaved set). The pandas 3.0.5 bump alone costs **~27% wall
clock** on this engine, and it hits the new code harder than the old — the new
panel code leans on `DatetimeIndex` iteration that got slower in pandas 3.

## 4. Where the time goes now (after, pinned deps)

Cumulative-time share of the 2.984 s instrumented profile, own code only:

| Rank | Function | ncalls | cumtime | % |
|---|---|---:|---:|---:|
| 1 | `prepare_rolling_backtest` `rolling_simulation.py:335` | 1 | 2.026 s | 67.9% |
| 2 | `build_signal_panel` `signal_panel.py:154` | 1 | 1.716 s | 57.5% |
| 3 | `run_prepared_rolling_backtest` `rolling_simulation.py:410` | 1 | 0.954 s | 32.0% |
| 4 | `evaluate_panel_many` `pine.py:815` | 1 | 0.761 s | 25.5% |
| 5 | `panel_index_key` `pine.py:721` | 600 | 0.734 s | 24.6% |
| 6 | `run_day_loop` `day_loop.py:246` | 1 | 0.701 s | 23.5% |
| 7 | `_precompute_filter_signals` `core.py:831` | 1 | 0.638 s | 21.4% |
| 8 | `_group_key` `pine.py:752` | 300 | 0.411 s | 13.8% |
| 9 | `_close_slot_at_day` `day_loop.py:73` | 7,549 | 0.340 s | 11.4% |
| 10 | `after_exits` `rolling_simulation.py:172` | 756 | 0.343 s | 11.5% |

The pendulum has swung to a clean single target: **`panel_index_key`
(`pine.py:721`) is the #1 own-code leaf.** It is called 600 times and each call
re-hashes the full `DatetimeIndex` (the `tuple(index)` fallback path drives
610,800 `pandas arrays/datetimes.__iter__` calls, 0.622 s of tottime — the single
largest tottime entry in the whole profile). 300 tickers share one identical
index; the key is recomputed per ticker per evaluation when it could be computed
once and memoized on identity. `_group_key` (13.8%) is its caller-side twin and
would fall with it.

## 5. GPU verdict — unchanged

Float math proper (`rolling.mean`) is a rounding error of the profile; the cost
is pandas dispatch, index hashing and object construction. Nothing here is GPU-
shaped, and at ~1.8 s/run on a 2-core VM (≈0.3 s/run on a 16-core host) the
cheap win is memoizing `panel_index_key`, not buying hardware.

## 6. What this PR changes

- `pyproject.toml` / `uv.lock` — pin `pandas==2.3.3`, `numpy==2.4.4`. This
  restores the measured engine speed and makes the profiling artifacts
  reproducible; without it, the lockfile silently ships the ~27% pandas-3
  penalty discovered here. Tests: 1,608 passed / 17 skipped on the pinned set.
- `profiling/webview/` — regenerated `rolling.prof`, `flamegraph_pyspy.svg`,
  `flamegraph_cprofile.svg`, `pstats_cumulative.txt`, `pstats_tottime.txt`,
  and `rolling_pre_vectorization.prof` (the `before` arm re-profiled on this box
  so the before/after delta is reproducible under identical conditions).
- `profiling/flamegraph_analysis_2026_08_09.md` — this document.
- `profiling/scripts/README.md` and `profiling/webview/index.html` — point at
  the new write-up and document the pinned-versions rule.
