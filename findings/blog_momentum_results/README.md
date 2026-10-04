# Blog momentum backtest results

Saved cells: **146**; provisional: **39**.

Each matched comparison uses the same frozen bars, universe, costs, dates, and portfolio sizing.
The 1/2/3/5-year windows all end on 2025-12-31.
The original six-year window remains available in the HTML.

## Results

| Window | Market | Assets | Rule | CAGR | Max drawdown | Δ momentum12_1 | Δ Minervini | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| 6y | india | etfs | alvarez_rotation_roc_hv63 | 44.73% | -17.39% | 32.67 pp | n/a | completed |
| 6y | india | etfs | momentum_12_1 | 12.06% | -29.29% | 0.00 pp | n/a | completed |
| 6y | india | stocks | unger_sma200_slope50 | 18.91% | -35.68% | -24.97 pp | n/a | completed |
| 6y | india | stocks | momentum_12_1 | 43.88% | -38.19% | 0.00 pp | n/a | provisional |
| 6y | us | etfs | chan_return_12m | 22.58% | -35.77% | 0.24 pp | n/a | completed |
| 6y | us | etfs | momentum_12_1 | 22.34% | -35.24% | 0.00 pp | n/a | completed |
| 6y | us | stocks | alvarez_monthly_momentum10 | 24.69% | -39.11% | 2.15 pp | n/a | completed |
| 6y | us | stocks | momentum_12_1 | 22.53% | -33.87% | 0.00 pp | n/a | completed |
| 5y | us | etfs | momentum_12_1 | 24.92% | -39.56% | 0.00 pp | 10.32 pp | completed |
| 5y | us | etfs | mark_minervini | 14.61% | -29.25% | -10.32 pp | 0.00 pp | completed |
| 5y | us | stocks | alvarez_monthly_momentum10 | 19.60% | -38.23% | n/a | n/a | completed |
| 2y | us | etfs | chan_return_12m | 37.66% | -10.89% | 17.13 pp | 17.12 pp | completed |
| 2y | us | etfs | momentum_12_1 | 20.53% | -19.50% | 0.00 pp | -0.02 pp | completed |
| 2y | us | etfs | mark_minervini | 20.54% | -10.78% | 0.02 pp | 0.00 pp | completed |
| 2y | us | stocks | alvarez_ma200_confirm3 | 33.01% | -25.15% | n/a | n/a | completed |

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
