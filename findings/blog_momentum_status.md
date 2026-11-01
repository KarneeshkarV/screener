# Blog momentum backtest status

## Completed batch

The approved scope is daily equities and ETFs in India and the US.
The main window is January 1, 2020 through December 31, 2025.
The batch contains 48 tests across rules associated with Ernie Chan, Alvarez Quant Trading, Unger Academy, and Robot Wealth.
Of these, 29 completed without early history-end forced sales and 19 are provisional because they include those sales.
A completed label does not certify full point-in-time membership, exact source replication, or deployment readiness.
All tests use fixed rules rather than optimized parameters.
No order was placed and no paid data was purchased.

## Read these files

- `findings/blog_momentum.md` contains the Chan rules, adaptations, data limits, and corrected 20-cell results.
- `findings/blog_momentum_additional_tests.md` contains the 28 additional-source results and their adaptation rules.
- `findings/alvarez_momentum_inventory.md` contains the broader Alvarez source inventory.
- `findings/unger_momentum_inventory.md` contains the broader Unger source inventory.
- `findings/robotwealth_momentum_inventory.md` contains the broader Robot Wealth source inventory.

The raw metrics, trade ledgers, warnings, frozen bars, and universe histories are under `reports/blog_momentum/`.
That directory is ignored by Git and is local to this machine.
The research scripts and notes are uncommitted worktree files.

## Verification

All 48 ledgers reconcile with final equity to within 0.00001 units of market currency.
Frozen-bar checks found no zero-volume entry or exit fills in the corrected results.
The corrected Chan cells repeat with exactly equal metrics.
All 28 additional-source cells also repeat with exactly equal metrics.
Nine research tests pass.
Ruff lint, Ruff format checks, and mypy pass for the research scripts.
The full offline suite passed with 2,874 tests passed and 17 skipped before the ninth research test was added.
The full run used `--no-cov` and does not certify the coverage threshold.

## Important limits

The India snapshot universe has long membership-observation gaps and 46 requested symbols lack price history.
US revision history also has missing snapshots.
Results affected by early symbol-history termination are provisional, not executable-performance evidence.
The ETF baskets are static surviving funds with different country exposures.
India ETF costs use the stock-delivery schedule as an approximation.
The US discretionary exits use the next close, whereas India exits use the next open.
These timing differences and basket differences prevent a clean country comparison.
The 50/200 US ETF crossover has only three trades.
A high Sharpe from so few trades is not reliable evidence.

## Work not completed

This batch does not test every strategy in the four inventories.
The inventory documents crawl coverage rather than claiming exhaustive discovery.
Monthly rotation and defensive allocation systems need synchronized portfolio rebalance, weight, and reserve-asset rules.
The GLD 170-day Donchian system needs intrabar stop entries and breakeven handling.
The Alvarez 260-day breakout needs close-based percentage exits and an explicit capital-constrained portfolio adaptation.
Earnings systems need historical event times and dates known before entry.
Intraday, futures, options, private, and incomplete rules remain outside the approved scope or blocked by missing data.
These must not be presented as backtested systems.

No strategy is ready for live deployment from this batch alone.

## HTML and monthly extension

The later extension adds 32 monthly-rule tests for a total of 80 cells.
Open `reports/blog_momentum/index.html` for charts, tables, filters, and CSV export.
Read `findings/blog_monthly_extension.md` for source rules and the explicit one-session-early rotation liquidation adaptation.
The final 80-cell ledgers pass reconciliation and zero-volume fill checks.
The full offline suite now passes with 2,878 tests passed and 17 skipped.
