# Blog momentum backtest results

Saved cells: **420**; provisional: **0**.

Each matched comparison uses the same frozen bars, universe, costs, dates, and portfolio sizing.
The 1/2/3/5-year windows all end on 2025-12-31.
The six-year window was also rerun and remains available in the HTML.

## Rerun after main PIT hardening

Rebased onto `main` at `b9c011f`, which includes the PIT hardening fix from PR #180.
All 420 comparisons were rerun with that engine.
The data and detailed artifacts remain in `/tmp/blog-momentum-pit-20261021/` and `/tmp/blog-momentum-fresh-20261021/`, outside Git.

### Membership basis

- **105 US stock comparisons:** all 25 requested quarterly/end-date S&P 500 snapshots were available for 2020-01-01 through 2025-12-31.
  The strict loader accepted the complete requested snapshot sequence.
  This is quarterly snapshot PIT, not exact daily index-change membership.
  The universe contains 624 selected symbols; 75 lack price history.
- **105 India stock comparisons:** dated Nifty 500 snapshots, with four observation gaps of at least 180 days and a largest gap of 648 days.
  Their dates are not verified official index effective dates.
  These runs were explicitly approved with warnings and are **not complete PIT**.
  Of 797 selected symbols, 104 lack price history.
- **210 ETF comparisons:** fixed surviving ETF baskets, used as controls.
  These are not PIT fund-universe reconstructions.
- `metrics.csv` records `membership_basis` for every row.
  Completed status means a simulation finished without an early history-end exit, not that membership or prices are complete.

### Verification

- All 420 cells and 52,240 trades passed ledger reconciliation and zero-volume fill checks.
- Maximum ledger reconciliation error is 0.000000005355 currency units.
- The verifier found no same-day monthly rotation expression exits.
- Full offline suite after rebase: 3,623 passed, 17 skipped; coverage 92.27%, above the 90% requirement.
- Ruff lint and formatting pass for 435 tracked Python files; core mypy passes for 245 source files.
- All seven research scripts pass mypy; the 12 focused research tests pass.
- Browser checks pass at 1440px desktop and 390px mobile, with working filters, no page-width overflow and no browser errors.

## Results

| Window | Market | Assets | Rule | CAGR | Max drawdown | Δ momentum12_1 | Δ Minervini | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| 6y | india | etfs | alvarez_rotation_roc_hv63 | 44.73% | -17.39% | 32.67 pp | 35.45 pp | completed |
| 6y | india | etfs | momentum_12_1 | 12.06% | -29.29% | 0.00 pp | 2.78 pp | completed |
| 6y | india | etfs | mark_minervini | 9.28% | -51.65% | -2.78 pp | 0.00 pp | completed |
| 6y | india | stocks | momentum_12_1 | 38.88% | -37.56% | 0.00 pp | 27.58 pp | completed |
| 6y | india | stocks | mark_minervini | 11.30% | -33.22% | -27.58 pp | 0.00 pp | completed |
| 6y | us | etfs | chan_return_12m | 22.58% | -35.77% | 0.24 pp | 10.43 pp | completed |
| 6y | us | etfs | momentum_12_1 | 22.34% | -35.24% | 0.00 pp | 10.19 pp | completed |
| 6y | us | etfs | mark_minervini | 12.15% | -29.25% | -10.19 pp | 0.00 pp | completed |
| 6y | us | stocks | alvarez_monthly_sma10 | 25.33% | -30.42% | 5.27 pp | 23.35 pp | completed |
| 6y | us | stocks | momentum_12_1 | 20.06% | -35.11% | 0.00 pp | 18.08 pp | completed |
| 6y | us | stocks | mark_minervini | 1.98% | -37.73% | -18.08 pp | 0.00 pp | completed |
| 5y | india | etfs | alvarez_rotation_roc_hv63 | 34.99% | -17.39% | 29.60 pp | 24.48 pp | completed |
| 5y | india | etfs | momentum_12_1 | 5.39% | -28.68% | 0.00 pp | -5.13 pp | completed |
| 5y | india | etfs | mark_minervini | 10.51% | -38.20% | 5.13 pp | 0.00 pp | completed |
| 5y | india | stocks | momentum_12_1 | 30.20% | -37.21% | 0.00 pp | 17.62 pp | completed |
| 5y | india | stocks | mark_minervini | 12.59% | -33.00% | -17.62 pp | 0.00 pp | completed |
| 5y | us | etfs | momentum_12_1 | 24.92% | -39.56% | 0.00 pp | 10.32 pp | completed |
| 5y | us | etfs | mark_minervini | 14.61% | -29.25% | -10.32 pp | 0.00 pp | completed |
| 5y | us | stocks | alvarez_ma200_confirm3 | 23.24% | -30.42% | 1.42 pp | 19.46 pp | completed |
| 5y | us | stocks | momentum_12_1 | 21.82% | -31.62% | 0.00 pp | 18.04 pp | completed |
| 5y | us | stocks | mark_minervini | 3.78% | -28.89% | -18.04 pp | 0.00 pp | completed |
| 3y | india | etfs | alvarez_rotation_roc_hv63 | 28.37% | -16.67% | 14.18 pp | 0.01 pp | completed |
| 3y | india | etfs | momentum_12_1 | 14.19% | -30.58% | 0.00 pp | -14.17 pp | completed |
| 3y | india | etfs | mark_minervini | 28.36% | -13.53% | 14.17 pp | 0.00 pp | completed |
| 3y | india | stocks | chan_log_ma_50_200 | 34.07% | -29.48% | 5.96 pp | 20.26 pp | completed |
| 3y | india | stocks | momentum_12_1 | 28.11% | -40.73% | 0.00 pp | 14.29 pp | completed |
| 3y | india | stocks | mark_minervini | 13.82% | -32.57% | -14.29 pp | 0.00 pp | completed |
| 3y | us | etfs | momentum_12_1 | 32.60% | -18.02% | 0.00 pp | 21.04 pp | completed |
| 3y | us | etfs | mark_minervini | 11.55% | -21.55% | -21.04 pp | 0.00 pp | completed |
| 3y | us | stocks | alvarez_ma200_confirm3 | 40.75% | -25.11% | 12.34 pp | 25.93 pp | completed |
| 3y | us | stocks | momentum_12_1 | 28.41% | -31.47% | 0.00 pp | 13.59 pp | completed |
| 3y | us | stocks | mark_minervini | 14.82% | -22.08% | -13.59 pp | 0.00 pp | completed |
| 2y | india | etfs | mark_minervini | 40.45% | -13.53% | 23.29 pp | 0.00 pp | completed |
| 2y | india | etfs | momentum_12_1 | 17.16% | -28.83% | 0.00 pp | -23.29 pp | completed |
| 2y | india | stocks | alvarez_ma200_confirm3 | 13.85% | -28.44% | 7.14 pp | 8.77 pp | completed |
| 2y | india | stocks | momentum_12_1 | 6.71% | -40.67% | 0.00 pp | 1.63 pp | completed |
| 2y | india | stocks | mark_minervini | 5.08% | -31.96% | -1.63 pp | 0.00 pp | completed |
| 2y | us | etfs | chan_return_12m | 37.66% | -10.89% | 17.13 pp | 17.12 pp | completed |
| 2y | us | etfs | momentum_12_1 | 20.53% | -19.50% | 0.00 pp | -0.02 pp | completed |
| 2y | us | etfs | mark_minervini | 20.54% | -10.78% | 0.02 pp | 0.00 pp | completed |
| 2y | us | stocks | alvarez_rotation_roc_hv126 | 35.16% | -32.97% | 2.06 pp | 8.90 pp | completed |
| 2y | us | stocks | momentum_12_1 | 33.10% | -31.68% | 0.00 pp | 6.84 pp | completed |
| 2y | us | stocks | mark_minervini | 26.25% | -22.08% | -6.84 pp | 0.00 pp | completed |
| 1y | india | etfs | alvarez_ma200_confirm3 | 65.83% | -10.42% | 34.26 pp | 0.90 pp | completed |
| 1y | india | etfs | momentum_12_1 | 31.57% | -17.26% | 0.00 pp | -33.36 pp | completed |
| 1y | india | etfs | mark_minervini | 64.93% | -10.42% | 33.36 pp | 0.00 pp | completed |
| 1y | india | stocks | alvarez_monthly_either10 | 15.12% | -9.26% | 18.62 pp | 28.89 pp | completed |
| 1y | india | stocks | momentum_12_1 | -3.50% | -24.99% | 0.00 pp | 10.27 pp | completed |
| 1y | india | stocks | mark_minervini | -13.77% | -26.05% | -10.27 pp | 0.00 pp | completed |
| 1y | us | etfs | chan_return_12m | 55.76% | -10.31% | 34.09 pp | 23.89 pp | completed |
| 1y | us | etfs | momentum_12_1 | 21.67% | -18.94% | 0.00 pp | -10.20 pp | completed |
| 1y | us | etfs | mark_minervini | 31.87% | -7.11% | 10.20 pp | 0.00 pp | completed |
| 1y | us | stocks | alvarez_rotation_roc_hv126 | 46.64% | -32.97% | 22.56 pp | 29.97 pp | completed |
| 1y | us | stocks | momentum_12_1 | 24.08% | -31.68% | 0.00 pp | 7.42 pp | completed |
| 1y | us | stocks | mark_minervini | 16.67% | -21.57% | -7.42 pp | 0.00 pp | completed |

## Limits

Rules are disclosed research adaptations, not exact publisher replications.
Monthly rotation liquidates one session before new entries; stock ROC-HV omits the source market MA200 gate.
Provisional means an early history-end sale at the last available quote.
Membership gaps, missing price histories, surviving ETF baskets, and approximate India ETF fees remain limitations.
Reference deltas do not remove provisional reference risk; reference statuses are included in the full CSV.
The strongest in-sample result is not a recommendation or an out-of-sample validation.

## Files

- `findings/blog_momentum_results/index.html`: standalone HTML, light/dark themes, window filters, reference comparisons.
- `findings/blog_momentum_results/metrics.csv`: all saved cells and matched-reference differences.
- `findings/blog_monthly_extension.md`: monthly rule and execution assumptions.
- Source inventories: Alvarez, Unger, Robot Wealth, and `findings/blog_momentum.md` for Chan.

Price data and invalid first-run rotation results are not committed.
