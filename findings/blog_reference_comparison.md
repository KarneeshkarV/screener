# Matched momentum reference comparisons

## Study design

The reference strategies are the repository's `momentum_12_1` and `mark_minervini` expression strategies.
The requested 5, 3, 2, and 1 calendar-year windows end on 2025-12-31.
Their start dates are 2021-01-01, 2023-01-01, 2024-01-01, and 2025-01-01.
The original 2020-2025 results remain a separate six-year window.
Each requested window contains 21 fixed strategies across US stocks, US ETFs, India stocks, and India ETFs, for 84 cells per window and 336 new cells total.
A four-cell Minervini batch completes the original six-year reference comparison.
No lookback, ranking weight, or source parameter is optimized for these windows.

## Reference conventions

`momentum_12_1` uses the existing repository entry and exit rules and a 62-session hold limit.
`mark_minervini` uses the full repository trend template, including its SMA200 slope and cross-sectional relative-strength rank.
Its exit is a completed-close crossing below SMA50, with the existing market-specific execution delay.
Minervini has no added stop, target, or binding time exit in this comparison.
Both references share the stock slots, ETF slot, fixed baskets, frozen price files, historical membership windows, and cost assumptions of the blog tests.
This controls portfolio settings, not the inherent difference between the two strategies' exit rules.
ETF references rank only the disclosed ETF basket, not the historical ETF market.

## Execution and data limits

US expression exits fill at the next close and India expression exits at the next open.
Monthly rotation closes old positions at month end, one session before the next entry open.
That rotation schedule is an explicitly disclosed adaptation, not synchronized publisher next-open rotation.
Stock ROC-HV adaptations omit the source's market MA200 filter.
Membership snapshots have gaps and some historic symbols have no price coverage.
Free adjusted daily bars and surviving ETF baskets do not certify survivorship-free execution.
Early history-end sales remain provisional in each window.
The dashboard compares reference results even when they are provisional, but marks their status and flags such deltas with an asterisk.
A larger in-sample CAGR is not evidence of future performance.

## Reproduction

Run `uv run python -m scripts.run_blog_comparison` after the original frozen data is available.
Run `uv run python -m scripts.verify_blog_backtests` to check trade P&L and fill dates.
Run `uv run python -m scripts.build_blog_dashboard` to rebuild the machine-local report.
Run `uv run python -m scripts.export_blog_comparison` to publish compact metrics and the standalone HTML under `findings/blog_momentum_results/`.
Raw frozen bars and trade exports stay machine-local and are not included in the PR.
The published HTML embeds all chart data and works without network access.
Its theme follows the operating system by default and a user's explicit light/dark choice is saved locally.

## Final verification

All 420 cells completed, including 336 requested-window cells and 84 original six-year cells.
There are 98 provisional cells across all windows.
All 420 trade ledgers reconcile with final equity and have no zero-volume entry or exit fills.
Rotation checks reject same-day expression exits.
The full offline suite passed with 2,952 tests passed and 17 skipped, using `--no-cov`.
Repository-wide Ruff lint and format checks passed, along with core mypy and research-script mypy.
The first PR CI run failed on one formatter difference in the export script; that was fixed and the next run passed.
Desktop and mobile browser checks passed for both themes, window filters, unique test IDs, reference tables, and year-specific heatmaps.
