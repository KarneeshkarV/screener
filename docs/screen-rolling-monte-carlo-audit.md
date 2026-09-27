# Screen, rolling, and Monte Carlo audit

## Scope and method

This audit used baseline commit `8d24273` and the deterministic 48-cell matrix from `scripts/backtest_delta.py`.
The baseline full suite had 2,835 passing tests, 17 skipped tests, and 91.72% coverage.
Each accuracy defect was reproduced through its public API or CLI path before its fix.
Focused regression tests were run red before implementation and green after implementation.
The three pre-existing untracked documents were not changed.

## Confirmed findings and fixes

### Dynamic screen universe policy

A screen resolved a dynamic universe to its symbol list and then discarded its size, ADV lookback, and rebalance schedule.
The screen now carries the complete dynamic policy and configured benchmark from universe resolution through the price and signal panels.
The screen window includes the active weekly, monthly, or quarterly period boundary so the dynamic algorithm can select membership on the same rebalance anchor as a rolling backtest.
Dynamic universes ignore the static `--max-universe` cap so liquidity selection sees the complete base universe.
Price warmup also covers the dynamic ADV lookback.
Static and point-in-time universe behavior remains unchanged.
The offline dynamic-screen reproduction now selects `AAA`, which matches the same-day rolling selection.

### Rolling refill candidate exhaustion

The rolling day loop materialized only a fixed top slice plus a small overfetch.
A lower-ranked affordable name could be lost when too many higher-ranked names could not buy one share.
The loop now consumes stable candidate batches until all free slots are filled or the ranked field is exhausted.
Ranks and setup scores still use the complete eligible field.
The normal path still materializes only one small batch.
The offline reproduction with nine unaffordable names now opens `CHEAP` at rank 10.

### Ruin threshold validation

The trade bootstrap did not validate `ruin_threshold`.
Both bootstrap APIs now use one validation function that requires a finite fraction in `(0, 1]`.
The `optimize validate` command performs this check before it reads the trade ledger and returns a Click usage error that names `--ruin-threshold`.
Regression coverage includes zero, a negative value, 1.5, NaN, and infinity at API and CLI seams.

### Summary Monte Carlo chart work

The summary API called the paths API with `keep_paths=0`, but it still allocated the percentile-band buffer and calculated chart bands.
Both public APIs now share one simulation core with an explicit chart-data collection choice.
The summary API skips retained-path and band work.
The paths API keeps its band contract when `keep_paths=0`.
The random stream and summary result stay equal to the paths variant for the same seed.
An independent offline benchmark used 5,000 iterations, 2,520 returns, block 20, seed 42, one warmup, and three timed repeats.
Median summary time changed from 0.451 seconds to 0.168 seconds, a 62.8% reduction.
The complete summary result was exactly unchanged.
The chart path measured 0.445 seconds before and 0.457 seconds after, so no chart-path speed gain is claimed.

### Portfolio dividend lookup

Non-full price adjustment masked a complete ticker frame once per trade to find dividend events.
The curve builder now caches each ticker's positive, nonmissing dividend series and uses index search bounds for each trade.
It falls back to the prior boolean-mask behavior for a public unsorted frame.
Trade accumulation order is unchanged.
Tests cover duplicate trade rows, partial-exit tranches, strict entry exclusion, inclusive exit credit, intraday timestamps, absent dividends, NaN, zero, and infinity behavior.
An independent end-to-end rolling benchmark used 100 tickers, 800 bars, 650 simulated days, 20 slots, a five-bar hold, non-adjusted prices, and 2,180 trades.
After one warmup and five timed repeats, median time changed from 0.928 seconds to 0.549 seconds, a 40.9% reduction.
Every equity value and metric was exactly unchanged.

### Screen display-row clipping

The candidate frame built display rows for the complete candidate field before applying a small result limit.
It now clips candidate rank order before display-row construction when the requested order is the default or `setup_score`.
It still builds the complete frame when a display column must be sorted.
Candidate setup scores remain the full-field percentiles calculated by the candidate layer.
Missing-bar filtering and the existing zero and negative limit behavior remain unchanged.
A focused 3,000-candidate frame benchmark changed from 0.198 seconds to 0.009 seconds with identical output, a 95.5% reduction for this function.
The supplied whole-screen benchmark was noisy in this environment and measured 2.592 seconds before its internal comparison and 2.693 seconds after it, so no whole-screen timing gain is claimed from that run.

## Verification

The two supplied offline CLI reproductions pass.
The deterministic backtest delta matrix is identical in all 48 cells with no skipped cells.
The focused six-file regression set passes with 310 tests.
Ruff lint passes.
Ruff format check passes.
Mypy strict checking passes for all 240 source files.
The final full suite passes with 2,862 tests, 17 skips, and 91.81% coverage.
Offline CLI regression tests also cover daily lagged ADV, monthly and quarterly membership anchors, dynamic-universe cap bypass, and multi-slot refill across candidate batches.

## Remaining caveats

Dynamic membership uses the first available master-calendar bar in each rebalance period, as the existing rolling algorithm specifies.
A data set that begins after the true period anchor cannot reconstruct a membership decision from unavailable bars, so the screen explicitly extends its simulation window to include the active period boundary.
The portfolio dividend fast path assumes a monotonic frame index only when that condition is true and deliberately falls back otherwise.
Performance timings depend on machine load, so the tests assert contracts and equality, not elapsed time.
