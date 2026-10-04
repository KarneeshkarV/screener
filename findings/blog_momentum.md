# Blog momentum research

## Scope and status

The user requested momentum strategies from Ernie Chan, Alvarez Quant Trading, Unger Academy, and Robot Wealth.
The approved scope is daily equities and ETFs in India and the US.
This is a source-led research comparison, not a trading recommendation.
A mention of momentum is not a complete strategy specification.
Incomplete, private, intraday, futures, and alternative-data strategies are recorded as blocked rather than replaced with invented rules.

## Ernie Chan archive coverage

The Blogger public feed returned 238 posts in two requests, with 150 and 88 entries.
The feed reported 238 total posts.
A case-insensitive search for `momentum`, `trend.follow`, or `breakout` identified 57 candidate posts.
The candidate index is in `reports/blog_momentum/chan_keyword_posts.json`.
This coverage concerns the public post bodies in the feed, not all comments, books, workshop material, or linked academic papers.
A keyword search can miss strategies described without these words.

Sources:

- <https://epchan.blogspot.com/feeds/posts/default?alt=json&max-results=150>
- <https://epchan.blogspot.com/feeds/posts/default?alt=json&max-results=150&start-index=151>

## Tested Chan price-based rules

### Full 12-month momentum

[Looking for momentum? Check outside the US](https://epchan.blogspot.com/2008/02/looking-for-momentum-check-outside-us.html) describes buying the highest 12-month return stocks, shorting the lowest, and holding for about one month.
The research adapter ranks `close[t] / close[t-252] - 1` without skipping the latest month.
It buys only positive-return candidates because the rolling equity portfolio does not model the original short book.
It uses ten stock slots and an entry-open to 21st-session-close holding period, configured as `hold=20` under the engine's counting convention.
The ETF adaptation uses one slot in a fixed ETF basket.
Slots refill when available, so this is a staggered holding-period implementation, not a synchronized calendar-month rebalance.
An exit also occurs if the signal becomes nonpositive.
These are explicit departures from the original long/short description.

### 12-1 momentum

[Update on the fundamentals factors](https://epchan.blogspot.com/2014/03/update-on-fundamentals-factors-their.html) cites price momentum over 11 months with the most recent month skipped, and prediction of quarterly returns.
The test uses the existing `momentum_12_1` strategy and an entry-open to 63rd-session-close holding period, configured as `hold=62` under the engine's counting convention.
It ranks the existing causal 12-1 factor, buys positive candidates, and omits the short book.
The same rule is also applied to ETFs as an adaptation.
The cited paragraph is a factor discussion, not a fully specified Chan portfolio.

### Log-price moving-average crossover

[Moving Average Crossover = Triangle Filter on 1-Period Returns](https://epchan.blogspot.com/2014/09/moving-average-crossover-triangle.html) specifies moving averages of log prices and direction changes when the short-minus-long average crosses zero.
The post's illustrated windows are 7 and 10 observations.
The first test uses those daily windows.
The second uses 50 and 200 sessions as a predeclared slower sensitivity check, not as parameters prescribed by Chan.
The long-only adapter enters on a positive crossover and exits on a negative crossover.
It uses next-session open entries, no profit target, no separate price stop, and a 10,000-session time limit that does not bind this study window.
The source also describes short positions after negative crossovers, which these tests omit.
Where many stock signals compete for ten slots, the existing engine's turnover ranking selects candidates.
Where ETF signals compete for one slot, the same engine selection rule applies.
These crossover basket tests are not rotational momentum portfolios.

### 13612W signal

[Welcome to Our Feature Zoo](https://epchan.blogspot.com/2021/09/welcome-to-our-feature-zoo-with-600.html) describes a 1-, 3-, 6-, and 12-month weighted momentum signal for canary assets in defensive and vigilant allocation.
The adapter uses `12*r21 + 4*r63 + 2*r126 + r252`, divided by 19.
The post does not explicitly print these weights, so they are a reciprocal-horizon interpretation of the named 13612W method.
The adapter ranks positive signals with the same slots and 21-session holding rule as the full-year momentum test.
It does not reproduce either full allocation model, its canary basket, cash fractions, or calendar-month rebalance.
The blog spells one canary symbol `VMO`; this study does not silently assume that spelling is a valid ETF or correct it within a claimed replica.

## Chan rules not tested

| Source | Published rule or idea | Reason |
|---|---|---|
| [Earnings momentum and reversal](https://epchan.blogspot.com/2008/05/combination-momentum-and-mean-reversal.html) | Top-percentile 12-month return names bought five days before earnings, sold before announcement; a second long leg starts after earnings and holds five days | Requires historical announcement times and the date known to traders before each entry; ambiguity in the post-event entry phrasing must be resolved |
| [Trading with earnings estimates](https://epchan.blogspot.com/2015/01/trading-with-estimize-and-ibes-earnings.html) | Short-term average estimate above long-term average predicts upward prices | Requires historical I/B/E/S estimate revisions and dated analyst updates, not current fundamentals |
| [A leveraged ETFs strategy](https://epchan.blogspot.com/2012/10/a-leveraged-etfs-strategy.html) | Index move of at least 2% from prior close to 14:15 ET; trade in that direction until close | Intraday and long/short; daily close cannot stand in for the 14:15 observation |
| [Beware of Low Frequency Data](https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html) | Futures long above prior-session 95th-percentile trade or quote price, exit below 60th percentile | Requires trade and quote data; OHLCV cannot recover the price distribution |
| [COT soybean strategy](https://epchan.blogspot.com/2015/02/commitments-of-traders-cot-strategy-on.html) | Follow speculator positioning using a long/short ratio and threshold exits | Futures and publication-lagged COT data are outside the approved scope |
| [Futures and forex momentum](https://epchan.blogspot.com/2011/03/momentum-strategies.html) | Overnight gaps and London breakout | No full entry, exit, range, or session rule; original markets are outside scope |
| [Time-of-day effects](https://epchan.blogspot.com/2011/05/time-of-day-effects-in-fx-trading.html) | Fixed-time FX momentum | Intraday FX, not daily equities |
| [Tail Reaper](https://epchan.blogspot.com/2020/03/why-does-our-tail-reaper-program-work.html) | Intraday directional crisis strategy; a 1% move is an explanatory example | Private strategy and intraday; the example is not a complete specification |
| [Momentum Crash and Recovery](https://epchan.blogspot.com/2013/07/momentum-crash-and-recovery.html) | DTI futures and proprietary soybean momentum | Complete soybean rules are explicitly withheld |
| [Momentum strategies: a book review](https://epchan.blogspot.com/2012/06/momentum-strategies-book-review.html) | Diversified futures trend indicator and orange juice marginal production cost example | Book discussion, futures, and no complete repeatable rule |
| [Order flow](https://epchan.blogspot.com/2012/10/order-flow-as-predictor-of-return.html) and [VPIN](https://epchan.blogspot.com/2013/10/how-useful-is-order-flow-and-vpin.html) | Signed trade volume predicts direction | Requires tick data, aggressor classification, and defined trading thresholds |
| [StockTwits sentiment](https://epchan.blogspot.com/2017/09/stocktwits-sentiment-analysis_7.html) | Weekly sentiment momentum | Historical messages and exact portfolio rules are unavailable |
| [Linear regression](https://epchan.blogspot.com/2011/04/many-facets-of-linear-regression.html) | Buy price far above estimated equilibrium | No threshold, window, or exit specification |
| [Nonlinear strategies](https://epchan.blogspot.com/2013/05/nonlinear-trading-strategies.html) | Options and tail-risk ideas | Options are outside scope and ideas are not daily equity momentum rules |

The other keyword matches primarily discuss momentum risk, workshops, generic features, or unrelated mean-reversion strategies.
They are not additional executable momentum strategies.

## Backtest design

The fixed main window is January 1, 2020 through December 31, 2025.
This includes the COVID decline and rebound, the 2022 US decline, and later bull markets.
Signals use trailing adjusted prices.
The existing rolling backtester controls fills, portfolio accounting, costs, and equity-curve metrics.
Entry decisions use completed bars before next-session execution.
India daily expression exits use the previous completed signal and fill at the next open.
The existing US engine otherwise evaluates expression exits at the current close.
The research adapter shifts US exit signals by one session to avoid using a not-yet-known closing signal for that same closing fill.
US discretionary exits therefore fill at the next close, whereas India exits fill at the next open.
This market-specific execution difference limits direct country comparison.
The study does not modify either backtest engine.
Initial capital is USD 100,000 or INR 1,000,000.
Sizing uses marked-equity equal slots.
Idle cash earns no interest, and the risk-free Sharpe hurdle is zero.
Slippage is five basis points per fill.
US commission is one basis point per fill as an assumed generic fee, not a broker-specific schedule.
India stock tests use the existing NSE equity-delivery statutory fee model.
India ETF tests also use that equity-delivery model as a conservative approximation, not an instrument-correct ETF tax schedule.
In particular, gold ETF and equity ETF transaction taxes differ from delivery stocks.
Do not treat the ETF net returns as exact after-tax returns.
Slippage is not a liquidity-capacity model, and personal income taxes are excluded.

The stock universes are the available S&P 500 revision history and the repository's NIFTY 500 snapshot history.
Historical membership is incomplete in both sources.
Two S&P 500 revision requests were rate-limited during initial acquisition.
The India snapshots include four gaps of at least 180 calendar days, with the largest 648 days.
Snapshot dates are declared but unverified effective dates.
These are membership-history-aware tests, not proof of a survivorship-free investable universe.
Prices for delisted or renamed securities can also be incomplete.
The frozen fetch adapter removes non-finite or nonpositive OHLC bars and zero-volume instrument bars before simulation.
Removed placeholder sessions cannot generate signals or fills, and holding periods count the remaining observed sessions.
The engine can still force an exit at a symbol's last quote if its history ends before the study window ends.
Every cell with such an early `eod` exit is marked `provisional`, and the affected trades are saved separately.
Those exits are valuation assumptions, not verified historical sales.
Missing-price warnings must be read next to returns.

The fixed US ETF basket is SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, and USO.
The India basket is NIFTYBEES, JUNIORBEES, BANKBEES, and GOLDBEES.
The baskets are surviving instruments selected for research, not point-in-time complete ETF universes.
They differ in asset mix, so their results are not a controlled comparison of country effects.

## Artifacts and reproduction

Research rules are in `scripts/blog_momentum_rules.py`.
The runner is `scripts/run_blog_momentum.py`.
Run `uv run python -m scripts.run_blog_momentum` from the repository root.
Each cell writes its full config, universe, metrics, warnings, trade ledger, and daily equity curve under `reports/blog_momentum/chan/`.
The runner also saves fetched bars for same-window offline replay.
Changing the date window requires a new output directory because the frozen bar files do not acquire a wider interval.
Reports and bar snapshots are local artifacts under the repository's ignored `reports/` directory.
The committed research note must not depend on those files being available to another clone.
Formula tests are in `tests/test_blog_momentum_rules.py`.
Nine tests passed for formulas, invalid prices, future-data independence, exit-signal lag, frozen-bar replay, zero-volume placeholder removal, and the additional-source trend formulas.
The existing momentum, registry, rolling characterization, and expression tests also passed, with 59 tests in that regression run.
The import-boundary checks passed.
Ruff lint, Ruff format checks, and mypy passed for the new research scripts.
The initial 20 cells repeated with exactly equal metrics on a same-window frozen replay.
The corrected 20-cell run also repeated with exactly equal metrics.
All 48 Chan and additional-source cells were checked against frozen bars and had no zero-volume entry or exit fills.
All 48 trade ledgers reconcile with final equity.
After the research-script corrections, the full offline suite passed with 2,874 tests passed and 17 skipped.
One further formula test was then added, and all nine research tests passed.
The earlier new-formula and import-boundary run passed with 66 tests.
The full suite used `--no-cov`, so it does not verify the coverage threshold.

## Results

The corrected run completed 20 cells.
Cells marked provisional contain early history-end sales at last available quotes.
They are not verified executable returns.

| Market | Assets | Rule | Status | Calendar CAGR | Max drawdown | Sharpe | Trades |
|---|---|---|---|---:|---:|---:|---:|
| us | etfs | chan_return_12m | completed | 22.58% | -35.77% | 0.89 | 74 |
| us | etfs | chan_weighted_13612 | completed | 0.75% | -52.03% | 0.16 | 83 |
| us | etfs | chan_log_ma_7_10 | completed | -6.98% | -56.98% | -0.27 | 113 |
| us | etfs | chan_log_ma_50_200 | completed | 21.77% | -18.36% | 1.28 | 3 |
| us | etfs | momentum_12_1 | completed | 22.34% | -35.24% | 0.85 | 24 |
| us | stocks | chan_return_12m | provisional | 14.79% | -35.92% | 0.58 | 721 |
| us | stocks | chan_weighted_13612 | provisional | 8.16% | -32.88% | 0.42 | 774 |
| us | stocks | chan_log_ma_7_10 | provisional | 4.29% | -38.70% | 0.31 | 1492 |
| us | stocks | chan_log_ma_50_200 | completed | 0.76% | -30.48% | 0.14 | 101 |
| us | stocks | momentum_12_1 | completed | 22.53% | -33.87% | 0.77 | 240 |
| india | etfs | chan_return_12m | completed | 6.97% | -52.94% | 0.43 | 71 |
| india | etfs | chan_weighted_13612 | completed | 26.13% | -15.68% | 1.30 | 83 |
| india | etfs | chan_log_ma_7_10 | completed | 16.86% | -19.34% | 1.01 | 90 |
| india | etfs | chan_log_ma_50_200 | completed | 9.62% | -30.63% | 0.72 | 4 |
| india | etfs | momentum_12_1 | completed | 12.06% | -29.29% | 0.70 | 24 |
| india | stocks | chan_return_12m | provisional | 31.61% | -43.33% | 1.13 | 712 |
| india | stocks | chan_weighted_13612 | provisional | 13.35% | -43.36% | 0.57 | 735 |
| india | stocks | chan_log_ma_7_10 | provisional | -3.66% | -41.14% | -0.10 | 1452 |
| india | stocks | chan_log_ma_50_200 | completed | 16.59% | -44.66% | 0.83 | 83 |
| india | stocks | momentum_12_1 | provisional | 43.88% | -38.19% | 1.48 | 241 |

The provisional cells must not be ranked against verified fills as if they had equal data quality.
The India 12-1 stock result remains provisional, even though its headline return is high.
The US 50/200 ETF result has only three trades, which is too few for a reliable strategy conclusion.
The India ETF numbers use an approximate fee model and a four-fund surviving basket.
No strategy is ready for deployment from this comparison alone.
