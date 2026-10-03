# Backtest performance and accuracy audit

## Scope

This audit checks the historical price cache and drawdown-duration metric in `screener/`.
Accuracy means preservation of price timestamps, frame types, and metric results, not improved investment returns.
The audit does not certify every strategy or provider.
Existing untracked blog research files are outside this change.

## Confirmed cache defects

The fast Arrow-to-pandas cache reader rebuilt time-zone-aware indexes from their raw UTC buffers without restoring their time zones.
A daily frame saved at India midnight on 2024-01-02 reloaded at 2024-01-01 midnight.
This differs from the canonical daily rule, which preserves the local calendar date.
Intraday data must instead use naive UTC timestamps.

The same reader changed nullable numeric extension types into NumPy types and numeric column labels into strings.
It also treated any index name beginning with `__index_level_` as unnamed, even when that was the user's explicit name.

The fix retains the fast path only for string column labels, ordinary numeric types, and a naive datetime index.
Other schemas use Arrow's metadata-aware `to_pandas()` conversion before canonical timestamp normalization.
The fast path now restores the index name from pandas metadata rather than guessing from the stored field name.
Regression tests compare the result against `to_pandas()` and test daily and intraday cache round trips.

## Confirmed performance finding

The drawdown-duration metric read a pandas timestamp for every bar, even when most bars belonged to one underwater interval.
The fix identifies running peaks with NumPy and subtracts timestamps only at drawdown boundaries.
Equal highs reset the peak timestamp, completed drawdowns include their recovery bar, and an open drawdown ends at the last bar.
Non-finite equity, unordered timestamps, and missing timestamps keep the existing scalar behavior.

A fixed 100,000-bar, 15-minute curve produced these local median timings across five repetitions:

| Implementation | Median |
| --- | ---: |
| Original scalar calculation | 320.670 ms |
| Vectorized calculation | 0.765 ms |

This is a 419.1x speedup for this metric only.
It is not a measured speedup for the complete backtest.
Timings depend on system load and hardware.

Run the repeatable offline benchmark with:

```bash
uv run python scripts/benchmark_drawdown_duration.py
```

The benchmark checks equality before it reports timing.
Tests also compare against an independent scalar reference on irregular curves, including daylight-saving transitions.

## Verification

The pre-change offline suite passed with 2,878 tests and 17 skips.
The deterministic backtest comparison produced identical results in all 48 cells after these fixes.
The matrix covers both engines, daily and 15-minute bars, both sizing rules, both slippage models, and three fee schedules.
It checks compatibility on canonical input, while the new cache tests cover the corrected non-canonical input.
The post-change offline suite passed with 2,894 tests and 17 skips.
Coverage was 91.85%, above the 90% floor.
Repository lint, format checks, and strict type checks passed.
