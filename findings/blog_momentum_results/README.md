# Blog momentum backtest results

Saved cells: **293**; provisional: **61**.

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
| 5y | india | stocks | chan_log_ma_50_200 | 17.27% | -28.00% | n/a | n/a | completed |
| 5y | us | etfs | momentum_12_1 | 24.92% | -39.56% | 0.00 pp | 10.32 pp | completed |
| 5y | us | etfs | mark_minervini | 14.61% | -29.25% | -10.32 pp | 0.00 pp | completed |
| 5y | us | stocks | momentum_12_1 | 22.50% | -31.62% | 0.00 pp | 19.25 pp | completed |
| 5y | us | stocks | mark_minervini | 3.25% | -30.38% | -19.25 pp | 0.00 pp | provisional |
| 3y | us | etfs | momentum_12_1 | 32.60% | -18.02% | 0.00 pp | 21.04 pp | completed |
| 3y | us | etfs | mark_minervini | 11.55% | -21.55% | -21.04 pp | 0.00 pp | completed |
| 3y | us | stocks | alvarez_ma200_confirm3 | 40.75% | -25.11% | n/a | n/a | completed |
| 2y | india | etfs | mark_minervini | 40.45% | -13.53% | 23.29 pp | 0.00 pp | completed |
| 2y | india | etfs | momentum_12_1 | 17.16% | -28.83% | 0.00 pp | -23.29 pp | completed |
| 2y | india | stocks | alvarez_ma200_confirm3 | 11.39% | -23.83% | n/a | n/a | completed |
| 2y | us | etfs | chan_return_12m | 37.66% | -10.89% | 17.13 pp | 17.12 pp | completed |
| 2y | us | etfs | momentum_12_1 | 20.53% | -19.50% | 0.00 pp | -0.02 pp | completed |
| 2y | us | etfs | mark_minervini | 20.54% | -10.78% | 0.02 pp | 0.00 pp | completed |
| 2y | us | stocks | alvarez_rotation_roc_hv252 | 37.27% | -34.60% | -0.32 pp | 10.69 pp | completed |
| 2y | us | stocks | momentum_12_1 | 37.59% | -31.62% | 0.00 pp | 11.01 pp | provisional |
| 2y | us | stocks | mark_minervini | 26.58% | -22.43% | -11.01 pp | 0.00 pp | completed |
| 1y | us | etfs | chan_return_12m | 55.76% | -10.31% | 34.09 pp | 23.89 pp | completed |
| 1y | us | etfs | momentum_12_1 | 21.67% | -18.94% | 0.00 pp | -10.20 pp | completed |
| 1y | us | etfs | mark_minervini | 31.87% | -7.11% | 10.20 pp | 0.00 pp | completed |
| 1y | us | stocks | chan_weighted_13612 | 37.03% | -28.72% | n/a | n/a | completed |

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
