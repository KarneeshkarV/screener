# Additional blog daily backtests

## Sources and limits

The source inventories are `findings/alvarez_momentum_inventory.md`, `findings/unger_momentum_inventory.md`, and `findings/robotwealth_momentum_inventory.md`.
Those files record crawl counts, original rules, citations, and blocked strategies.
The first executable batch tests simple daily price rules from each source on the same stock and ETF baskets as the Chan study.
These are long-only portfolio adaptations, not replicas of the publishers' charts.
The source inventories are broader than the implemented batch.
Monthly asset allocation, short books, intraday stop orders, breakeven stops, and futures systems are not reproduced by these tests.

## Tested rules

[Alvarez MA200 confirmation](https://alvarezquanttrading.com/blog/reducing-whipsaws-when-using-200-day-moving-average-for-market-timing/) buys after three or five consecutive daily closes above SMA200 and exits after the same number below it.
The two confirmation lengths are fixed source variants, not an optimized selection.
The original rule is tested on individual index ETFs; the multi-ETF basket and ten-slot stock portfolios are adaptations.

[Unger moving-average regimes](https://ungeracademy.com/blog/moving-averages-trading-market-regimes) provides close versus SMA200, SMA50 versus SMA200, and SMA200 now versus its value 50 observations ago.
The tests hold eligible assets while these measures are positive and exit when negative.
Turning a regime filter into a standalone long/cash portfolio is an adaptation.
Daily observations are explicit research conventions.
The SMA50/200 test uses the positive state rather than requiring a new crossing event.

[Robot Wealth's educational SMA strategy](https://robotwealth.com/parameter-optimisation-for-systematic-trading/) enters on a close/SMA100 positive crossover and exits on the negative crossover.
The 100-bar default comes from the public code.
Daily equity bars, the selected universes, sizing, costs, and order timing are research assumptions because the source snippet leaves them unspecified.

The Unger five-day Donchian negative control is also adapted to completed-close confirmation followed by next-open entry.
It exits when close falls below the prior five-session low.
The source uses intrabar channel breaks, so a close-confirmed test is not an exact replica.
The original Nasdaq 100 universe is replaced by the available S&P 500 and NIFTY 500 membership histories for the stock tests.

## Shared execution conventions

The main window is January 1, 2020 through December 31, 2025.
The runner uses ten stock slots or one ETF slot.
The capital, adjusted-price convention, fees, slippage, and universe limits are recorded in `findings/blog_momentum.md`.
Signals use completed bars.
India expression exits fill at the next open.
US exit signals are delayed by one bar and fill at the next close to avoid same-close signal lookahead.
This US exit timing differs from the publishers' next-open instructions and is another adaptation.
There is no profit target or separate protective stop in this batch.
A 10,000-session time exit does not bind the main window.
The cash balance earns no interest.
ETF basket members compete for one slot, so these are neither individual-ETF tests nor equal-weight allocations across the ETF list.
Stock and ETF candidate conflicts use the existing engine's turnover ranking.

Zero-volume instrument bars and invalid OHLC bars are removed before the engine runs.
Index benchmark bars can have zero volume.
A result with an early history-end exit is marked provisional because its forced last-quote sale is not a verified fill.
The India ETF fee model is an equity-delivery approximation, not an exact ETF tax schedule.

## Artifacts

Results are saved locally under `reports/blog_momentum/daily_filters/`.
Each cell contains configuration, universe, warnings, trade ledger, equity curve, and metrics.
The raw acquired bars and membership windows are frozen for a same-window repeat check.
The report directory is ignored by Git.
The source research notes and scripts remain available in the worktree.

## Results

The first batch completed 28 cells, including four close-confirmed Donchian controls.
All 28 trade ledgers reconcile with final equity to within 0.00001 units of market currency.
All 28 cells repeated with exactly equal metrics on frozen-bar and frozen-universe replay.
No entry or exit fill uses a zero-volume instrument bar.
Provisional cells contain early history-end last-quote sales and are not verified executable returns.

| Market | Assets | Rule | Status | Calendar CAGR | Max drawdown | Sharpe | Trades |
|---|---|---|---|---:|---:|---:|---:|
| us | etfs | alvarez_ma200_confirm3 | completed | 16.32% | -29.22% | 0.77 | 11 |
| us | etfs | alvarez_ma200_confirm5 | completed | 7.99% | -32.33% | 0.48 | 9 |
| us | etfs | unger_close_sma200 | completed | 6.82% | -37.26% | 0.44 | 24 |
| us | etfs | unger_sma50_200 | completed | 5.49% | -33.72% | 0.36 | 9 |
| us | etfs | unger_sma200_slope50 | completed | 11.39% | -33.72% | 0.58 | 6 |
| us | etfs | robotwealth_close_sma100 | completed | 10.41% | -27.26% | 0.71 | 44 |
| us | stocks | alvarez_ma200_confirm3 | provisional | 23.36% | -33.93% | 0.92 | 120 |
| us | stocks | alvarez_ma200_confirm5 | provisional | 16.70% | -32.31% | 0.72 | 110 |
| us | stocks | unger_close_sma200 | provisional | 18.77% | -35.73% | 0.78 | 197 |
| us | stocks | unger_sma50_200 | provisional | 23.60% | -32.56% | 0.90 | 80 |
| us | stocks | unger_sma200_slope50 | provisional | 22.92% | -34.22% | 0.87 | 57 |
| us | stocks | robotwealth_close_sma100 | completed | 9.99% | -31.15% | 0.58 | 633 |
| india | etfs | alvarez_ma200_confirm3 | completed | 12.19% | -23.88% | 0.83 | 9 |
| india | etfs | alvarez_ma200_confirm5 | completed | 11.50% | -26.53% | 0.79 | 8 |
| india | etfs | unger_close_sma200 | completed | 16.09% | -21.95% | 0.98 | 15 |
| india | etfs | unger_sma50_200 | completed | 8.70% | -36.83% | 0.56 | 7 |
| india | etfs | unger_sma200_slope50 | completed | 11.19% | -36.83% | 0.74 | 4 |
| india | etfs | robotwealth_close_sma100 | completed | 30.91% | -13.89% | 1.77 | 27 |
| india | stocks | alvarez_ma200_confirm3 | provisional | 17.10% | -36.13% | 0.86 | 112 |
| india | stocks | alvarez_ma200_confirm5 | provisional | 10.80% | -38.55% | 0.58 | 100 |
| india | stocks | unger_close_sma200 | provisional | 21.11% | -36.17% | 1.01 | 152 |
| india | stocks | unger_sma50_200 | provisional | 10.68% | -43.72% | 0.56 | 93 |
| india | stocks | unger_sma200_slope50 | completed | 18.91% | -35.68% | 0.89 | 73 |
| india | stocks | robotwealth_close_sma100 | provisional | 9.26% | -37.79% | 0.58 | 614 |
| us | etfs | unger_donchian5 | completed | 8.15% | -51.21% | 0.48 | 92 |
| us | stocks | unger_donchian5 | provisional | 9.65% | -27.71% | 0.54 | 962 |
| india | etfs | unger_donchian5 | completed | 14.37% | -15.91% | 0.92 | 45 |
| india | stocks | unger_donchian5 | provisional | 8.98% | -40.86% | 0.53 | 922 |

These tests do not exhaust the source inventories.
The monthly rotational strategies require a synchronized portfolio rebalance that this batch does not implement.
The GLD 170-day system needs intrabar stop-entry and breakeven handling to preserve its rules.
Do not substitute the close-confirmed Donchian control for that system.
