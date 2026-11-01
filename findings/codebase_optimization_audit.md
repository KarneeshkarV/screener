# Wider codebase optimization audit

## Scope and limits

The inventory covers all 406 tracked Python files present at the start of this pass, including 240 production modules.
All tracked Python files were parsed and checked for repeated row operations, datetime assumptions, forward references, and cache writes.
This is a repository-wide scan and targeted runtime review, not a claim that every line received a complete manual audit.
Changes are limited to confirmed defects and measured workloads.
Correct files are left unchanged.
Existing untracked blog research files are outside this change.
No generated files, dependencies, or strategy rules were changed.

## Runtime findings and fixes

The offline 48-cell matrix identified trailing liquidity calculations as the largest measured engine cost.
The old implementation sliced pandas indexes, built filtered Series, and copied volume data for each completed bar.
The new implementation caches numeric arrays, calculates volume means from those arrays, and retains pandas standard deviation to preserve optional bottleneck behavior and exact fill inputs.
Both engines use this shared calculation.
Cache buffers no longer retain redundant volume and return Series.

Factor IC previously used a pandas callback per date.
The new implementation masks paired observations first, ranks complete matrices, and computes row correlations in one batch.
Quantile calculations now batch the rank-based qcut formula, including pandas' boundary-rounding rule and column-order tie breaking.
Turnover uses Boolean membership matrices for unique labels and preserves string-set behavior when labels collide.

Operator labels previously created one Series per row.
The new implementation uses column masks and preserves object-typed None values for unlabelled rows.
Missing nullable futures membership no longer raises an ambiguous-Boolean error.

Screen history writes now use tuples rather than one Series per row.
Missing symbol values no longer become the literal ticker `nan`.
The original row ranks and database transaction semantics remain unchanged.

## Correctness findings and fixes

Shared JSON and parquet cache writes used one fixed `.tmp` filename.
A deterministic two-writer test showed that one writer could remove the other writer's temporary file.
A failed replacement also left temporary files behind.
Writers now receive unique temporary files, replace the target atomically, and clean temporary files after success or failure.
The panel-snapshot writer uses the same atomic helper and retains its existing process lock.

The frame-cache binary-search shortcut assumed that a unique datetime index was sorted and had no missing timestamps.
The shortcut now requires both properties.
Other indexes use the existing pandas lookup path.

## Local measurements

The repeatable benchmark compares this pass with commit `2776bb3`, the first part of PR #171.
This means the measurements below do not count the earlier drawdown-metric improvement twice.
Factor workloads use 1,000 dates and 300 names with fixed random missing cells.
Operator and history workloads use 10,000 rows.
Matrix measurements include subprocess startup and all 48 offline cases.

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Factor IC | 453.37 ms | 42.97 ms | 10.55x |
| Quantile returns | 873.11 ms | 68.81 ms | 12.69x |
| Quantile turnover | 1,074.95 ms | 21.24 ms | 50.62x |
| Operator labels | 195.50 ms | 3.92 ms | 49.87x |
| Screen history save | 431.25 ms | 110.52 ms | 3.90x |
| Complete backtest matrix | 5.190 s | 3.693 s | 1.41x |

The matrix took about 29% less time in this benchmark run.
Timings are local medians and depend on hardware and concurrent load.
No speed claim is made for network-bound provider requests or every possible strategy.

Run the benchmark with:

```bash
uv run python scripts/benchmark_codebase.py --reference 2776bb3
```

## Verification

All 48 backtest cases remain identical to the recorded baseline.
Liquidity tests compare exact results with the original pandas calculation, including missing prices, zero prices, missing volumes, short windows, and both bottleneck settings.
Factor tests compare batched IC against pairwise ranks and batched quantiles against qcut on tied and missing scores.
Operator labels match the scalar reference on randomized rows.
Concurrent cache tests force both writers to reach atomic replacement together.
History tests verify missing-symbol handling and preserved ranks.
The full offline suite passed with 2,952 tests and 17 skips.
Coverage was 91.84%, above the 90% floor.
Repository lint, formatting, and strict type checks passed.

## Areas left unchanged

CLI startup, provider composition, strategy definitions, indicators, options, earnings research, scoring, and unusual-volume detection were included in the scan and repository checks.
This pass did not establish a measured optimization or confirmed defect in those paths.
Changing them without such evidence would increase regression risk without a proven benefit.
Live provider latency and UI rendering were not benchmarked.
