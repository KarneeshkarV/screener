# Blog momentum backtest results

Saved cells: **420**; provisional: **98**.

Each matched comparison uses the same frozen bars, universe, costs, dates, and portfolio sizing.
The 1/2/3/5-year windows all end on 2025-12-31.
The original six-year window remains available in the HTML.

## Results

| Window | Market | Assets | Rule | CAGR | Max drawdown | Δ momentum12_1 | Δ Minervini | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| 6y | india | etfs | alvarez_rotation_roc_hv63 | 44.73% | -17.39% | 32.67 pp | 35.45 pp | completed |
| 6y | india | etfs | momentum_12_1 | 12.06% | -29.29% | 0.00 pp | 2.78 pp | completed |
| 6y | india | etfs | mark_minervini | 9.28% | -51.65% | -2.78 pp | 0.00 pp | completed |
| 6y | india | stocks | unger_sma200_slope50 | 18.91% | -35.68% | -24.97 pp | 7.30 pp | completed |
| 6y | india | stocks | momentum_12_1 | 43.88% | -38.19% | 0.00 pp | 32.26 pp | provisional |
| 6y | india | stocks | mark_minervini | 11.62% | -36.15% | -32.26 pp | 0.00 pp | completed |
| 6y | us | etfs | chan_return_12m | 22.58% | -35.77% | 0.24 pp | 10.43 pp | completed |
| 6y | us | etfs | momentum_12_1 | 22.34% | -35.24% | 0.00 pp | 10.19 pp | completed |
| 6y | us | etfs | mark_minervini | 12.15% | -29.25% | -10.19 pp | 0.00 pp | completed |
| 6y | us | stocks | alvarez_monthly_momentum10 | 24.69% | -39.11% | 2.15 pp | 23.16 pp | completed |
| 6y | us | stocks | momentum_12_1 | 22.53% | -33.87% | 0.00 pp | 21.01 pp | completed |
| 6y | us | stocks | mark_minervini | 1.53% | -39.26% | -21.01 pp | 0.00 pp | provisional |
| 5y | india | etfs | alvarez_rotation_roc_hv63 | 34.99% | -17.39% | 29.60 pp | 24.48 pp | completed |
| 5y | india | etfs | momentum_12_1 | 5.39% | -28.68% | 0.00 pp | -5.13 pp | completed |
| 5y | india | etfs | mark_minervini | 10.51% | -38.20% | 5.13 pp | 0.00 pp | completed |
| 5y | india | stocks | unger_sma200_slope50 | 24.21% | -22.44% | -7.46 pp | 11.61 pp | completed |
| 5y | india | stocks | momentum_12_1 | 31.67% | -37.50% | 0.00 pp | 19.08 pp | provisional |
| 5y | india | stocks | mark_minervini | 12.60% | -36.18% | -19.08 pp | 0.00 pp | completed |
| 5y | us | etfs | momentum_12_1 | 24.92% | -39.56% | 0.00 pp | 10.32 pp | completed |
| 5y | us | etfs | mark_minervini | 14.61% | -29.25% | -10.32 pp | 0.00 pp | completed |
| 5y | us | stocks | momentum_12_1 | 22.50% | -31.62% | 0.00 pp | 19.25 pp | completed |
| 5y | us | stocks | mark_minervini | 3.25% | -30.38% | -19.25 pp | 0.00 pp | provisional |
| 3y | india | etfs | alvarez_rotation_roc_hv63 | 28.37% | -16.67% | 14.18 pp | 0.01 pp | completed |
| 3y | india | etfs | momentum_12_1 | 14.19% | -30.58% | 0.00 pp | -14.17 pp | completed |
| 3y | india | etfs | mark_minervini | 28.36% | -13.53% | 14.17 pp | 0.00 pp | completed |
| 3y | india | stocks | chan_log_ma_50_200 | 33.86% | -29.48% | 7.77 pp | 20.89 pp | completed |
| 3y | india | stocks | momentum_12_1 | 26.08% | -42.02% | 0.00 pp | 13.12 pp | provisional |
| 3y | india | stocks | mark_minervini | 12.96% | -36.19% | -13.12 pp | 0.00 pp | completed |
| 3y | us | etfs | momentum_12_1 | 32.60% | -18.02% | 0.00 pp | 21.04 pp | completed |
| 3y | us | etfs | mark_minervini | 11.55% | -21.55% | -21.04 pp | 0.00 pp | completed |
| 3y | us | stocks | alvarez_ma200_confirm3 | 40.75% | -25.11% | 12.29 pp | 25.58 pp | completed |
| 3y | us | stocks | momentum_12_1 | 28.46% | -31.47% | 0.00 pp | 13.29 pp | completed |
| 3y | us | stocks | mark_minervini | 15.17% | -22.43% | -13.29 pp | 0.00 pp | provisional |
| 2y | india | etfs | mark_minervini | 40.45% | -13.53% | 23.29 pp | 0.00 pp | completed |
| 2y | india | etfs | momentum_12_1 | 17.16% | -28.83% | 0.00 pp | -23.29 pp | completed |
| 2y | india | stocks | alvarez_ma200_confirm3 | 11.39% | -23.83% | 2.07 pp | 7.60 pp | completed |
| 2y | india | stocks | momentum_12_1 | 9.32% | -40.67% | 0.00 pp | 5.53 pp | provisional |
| 2y | india | stocks | mark_minervini | 3.78% | -35.41% | -5.53 pp | 0.00 pp | completed |
| 2y | us | etfs | chan_return_12m | 37.66% | -10.89% | 17.13 pp | 17.12 pp | completed |
| 2y | us | etfs | momentum_12_1 | 20.53% | -19.50% | 0.00 pp | -0.02 pp | completed |
| 2y | us | etfs | mark_minervini | 20.54% | -10.78% | 0.02 pp | 0.00 pp | completed |
| 2y | us | stocks | alvarez_rotation_roc_hv252 | 37.27% | -34.60% | -0.32 pp | 10.69 pp | completed |
| 2y | us | stocks | momentum_12_1 | 37.59% | -31.62% | 0.00 pp | 11.01 pp | provisional |
| 2y | us | stocks | mark_minervini | 26.58% | -22.43% | -11.01 pp | 0.00 pp | completed |
| 1y | india | etfs | alvarez_ma200_confirm3 | 65.83% | -10.42% | 34.26 pp | 0.90 pp | completed |
| 1y | india | etfs | momentum_12_1 | 31.57% | -17.26% | 0.00 pp | -33.36 pp | completed |
| 1y | india | etfs | mark_minervini | 64.93% | -10.42% | 33.36 pp | 0.00 pp | completed |
| 1y | india | stocks | alvarez_monthly_either10 | 15.12% | -9.26% | 23.42 pp | 30.71 pp | completed |
| 1y | india | stocks | momentum_12_1 | -8.30% | -24.99% | 0.00 pp | 7.29 pp | provisional |
| 1y | india | stocks | mark_minervini | -15.59% | -26.74% | -7.29 pp | 0.00 pp | completed |
| 1y | us | etfs | chan_return_12m | 55.76% | -10.31% | 34.09 pp | 23.89 pp | completed |
| 1y | us | etfs | momentum_12_1 | 21.67% | -18.94% | 0.00 pp | -10.20 pp | completed |
| 1y | us | etfs | mark_minervini | 31.87% | -7.11% | 10.20 pp | 0.00 pp | completed |
| 1y | us | stocks | alvarez_rotation_roc_hv126 | 40.82% | -35.45% | 16.74 pp | 24.96 pp | completed |
| 1y | us | stocks | momentum_12_1 | 24.08% | -31.68% | 0.00 pp | 8.22 pp | completed |
| 1y | us | stocks | mark_minervini | 15.86% | -22.16% | -8.22 pp | 0.00 pp | completed |

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

## Validation

- 420 cells: 84 per window; 98 provisional results.
- All ledgers reconcile; no zero-volume fills; no same-day monthly rotation expression exits.
- Full offline tests: 2,952 passed, 17 skipped (`--no-cov`, coverage not checked locally).
- Ruff lint/format and core/research mypy passed.
- PR CI passed after correcting an export-script formatter difference.
- Stacked on PR #171 to keep the existing performance changes separate.
- Hosted report: https://omen.tail263417.ts.net/momentum (tailnet only).
