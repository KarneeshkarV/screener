# Monthly blog extension

## New rules

The batch adds eight fixed rules, with four market and asset cells per rule.
The original four websites remain the source scope; this extension does not add an unrelated publisher.
This adds 32 backtests to the earlier 48.

| Rule | Source | Implementation |
|---|---|---|
| Positive 10-month return | [Alvarez](https://alvarezquanttrading.com/blog/trend-following-vs-momentum-in-etfs/) | Month-end signal, long/cash |
| Close above 10-month SMA | Same source | Monthly closes, not daily SMA200 |
| Both trend and momentum positive | [Alvarez combined](https://alvarezquanttrading.com/blog/trend-following-plus-momentum-in-etfs/) | Fully invested or cash |
| Either trend or momentum positive | Same source | Source alternate 100% rule |
| ROC times historical volatility, 63 days | [Alvarez rotation](https://alvarezquanttrading.com/blog/different-ranking-methods-for-a-monthly-sp500-stock-rotation-strategy/) | Monthly turnover with source score |
| ROC times historical volatility, 126 days | Same source | Fixed source lookback |
| ROC times historical volatility, 252 days | Same source | Fixed source lookback |
| Three-factor rank | [Alvarez three-factor](https://alvarezquanttrading.com/blog/three-factor-etf-rotation-strategy/) | 40% ROC63 rank, 40% ROC20 rank, 20% HV20 rank; top one before MA200 filter |

The public Alvarez monthly momentum, combined, and rotation posts were retrieved again for this extension.
The Robot Wealth rotation article was also retrieved, but its public code and narrative select different assets.
That system is not silently replaced by an Alvarez rule.

## Adaptations

These are portfolio adaptations through the existing rolling engine.
They are not exact source replications.
The source ETF tests are separate instruments; this batch uses the earlier one-slot ETF baskets.
Stock applications of ETF-only rules are explicitly research adaptations.
The stock ROC times HV tests retain the own-price MA200 gate but omit the source's market MA200 gate.
This omission is material and is not described as an original source replication.
The three-factor score substitutes 63 trading sessions for three calendar months.
It uses the existing ETF baskets or historical stock universe, not the source's sector or five-asset baskets.
Cross-sectional ranks exclude names outside observed membership windows.
Ties use average component ranks and symbol-column order for the final top-one tie.

Month ends are the last observed union-calendar session in each completed month.
The source orders use a synchronized next-open portfolio rotation.
The current engine does not reproduce that order sequence: exits and entries follow its existing country-specific execution and event order.
Rotation exit signals occur on the session before month end.
India liquidation fills at the month-end open and US liquidation fills at the month-end close.
New positions enter at the following session's open.
This creates a one-session gap compared with the original next-open synchronized rotation.
It prevents a stale rotation exit signal from closing the new position on its entry day.
The initial rotation run had that error and is excluded under `reports/blog_momentum/monthly_invalid/`.
Its summary is named `excluded_summary.json` so the HTML builder cannot load it.
The four monthly trend and momentum rule variants are unaffected and use their normal next-session exit conventions.
No claim is made that these returns equal an exact monthly rebalance simulator.

The batch keeps the earlier capital, slippage, fee model, frozen bars, frozen membership history, and 2020-2025 window.
Zero-volume instrument bars are removed.
Early history-end sales mark a cell provisional.
The earlier membership gaps and India ETF cost limits still apply.

## HTML report

Build the offline report with `uv run python -m scripts.build_blog_dashboard`.
Open `reports/blog_momentum/index.html` in a browser.
The page embeds the data and has no CDN, API, package, or network dependency.
It contains risk/return scatter, CAGR comparison bars, a sortable ledger, selected equity and month-end drawdown charts, and monthly-return heatmaps.
Filters cover market, asset class, source, name, and provisional status.
Provisional results are hidden by default.
The table's `No early EOD` label is narrow and does not imply complete historical data.
The report was tested at 1440-pixel desktop width and 390-pixel mobile width.
Filter, search, empty state, sorting, row selection, chart selection, and CSV download checks passed with no browser script errors.
The browser's normal sandbox was unavailable on this machine.
UI verification used a temporary headless browser with the sandbox disabled, restricted to loopback HTTP; the browser was then closed.
The monthly drawdown chart is sampled monthly; daily maximum drawdown remains in the table.

## Verification

The full offline suite passed with 2,878 tests passed and 17 skipped.
The run used `--no-cov`, so it does not verify the coverage threshold.
Twelve research/report tests and the import-boundary checks passed, with 70 tests in that focused run.
Ruff lint, Ruff format checks, and mypy passed for all five research scripts.
All eight corrected ETF rotation cells repeat with exactly equal metrics on frozen bars.
The rotation ledgers have approximately 29 calendar days median holding time, rather than the erroneous entry-day exits.

The final report contains 80 cells across 20 rule variants.
All 80 ledgers reconcile with final equity, and no entry or exit uses a zero-volume bar.
There are 30 provisional cells with early history-end sales.
Invalid first-run rotation results are excluded from the HTML.
