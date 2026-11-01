# Robot Wealth daily momentum and trend inventory

## Decision for the AFK backtest

Start with the public 252-session SPY/EFA/SHY rotation code in [S1](https://robotwealth.com/etf-rotation-strategies-in-zorro/).
It has the clearest universe, lookback, schedule, and allocation intent.
Run the code rule and the article's prose rule as separate variants because they select different assets.
The next simplest daily test is the long-only price/SMA crossover in [S4](https://robotwealth.com/parameter-optimisation-for-systematic-trading/), with its stated 100-bar default.
Its conversion to daily equity bars is an adaptation because the example does not set a bar interval or an asset.
Monthly top-four six-month momentum with inverse three-month volatility is the next useful test, but [S3](https://robotwealth.com/momentum-dead-alive/) does not disclose enough inputs for an exact replica.
Do not start with the continuous BTC signal, the leveraged crypto simulation, or the private SPY webinar.

This note records source rules, not new backtest results.
No published return or Sharpe value here is evidence that a screener implementation matches the source.
The companion machine-readable inventory is `reports/blog_momentum/robotwealth_inventory.json`.

## Scope and coverage

Research date is 2026-10-02 UTC.
The public [blog archive](https://robotwealth.com/blog/) advertised 18 pages.
All 18 pages were fetched with curl and yielded 207 post links, all unique.
A separate Python requests attempt returned HTTP 403 on all 18 pages and yielded no links.
The 207 linked post bodies were fetched and keyword-screened for momentum, trend following, moving average, crossover, breakout, and SPY.
This was a text screen, not a manual strategy review of every post.
Four responses had fewer than 2,000 text characters, so a fetch is not proof that the full article was available.
Those four were `researching-trading-ideas-in-excel`, `debunking-market-myths-in-excel`, `basics-of-edge-extraction`, and `machine-learning-in-algorithmic-trading-systems-opportunities-and-pitfalls`.
Seventeen focal posts were inspected for rules, limitations, or access status, as listed below.
The post sitemap contained 483 top-level URL records, including private material and the blog page.
Its 1,755 nested `loc` values include image URLs and must not be called a post count.
The sitemap was screened by URL name, not fully crawled.
Two relevant private webinar pages were inspected and explicitly refused access.
This is a bounded inventory of public candidates, not an exhaustive inventory of Robot Wealth strategies.
Image-only details, member notebooks, deleted pages, and strategies with unrelated titles can remain missing.

The `agent-browser` skill and its `core` guide were read before research.
Text pages were read with `agent-browser read` and source links were extracted from the same first-party HTML.
Launching the rendered browser failed because Chromium had no usable sandbox.
The browser doctor reported no local installation failure, and no desktop or sandbox configuration was changed.
Linked original notebooks and C++ source were read directly from Robot Wealth's public GitHub repositories.
No package install, code execution from the website, or backtest was needed.

## Definitions and common test assumptions

`C_i,t` is the observed close of asset `i` at completed session `t`.
`R_i,t,L = C_i,t / C_i,t-L - 1` is a simple trailing return without a skipped recent month.
A calendar-month lookback is not automatically the same as 21, 63, 126, or 252 sessions.
A replica preserves the source universe, signal calculation, schedule, allocation, data convention, and execution semantics.
An adaptation changes any of those, including replacing bonds with idle cash or replacing ETFs with individual stocks.
Where execution timing is absent, the proposed daily adaptation calculates signals after close `t` and trades at open `t+1`.
That proposal is not a claim about the source fill price.
A third-session rebalance at its open must use the second session's completed data, whereas a signal calculated after the third session closes trades on the fourth session.
Record the chosen interpretation before testing.
Do not calculate a signal from a closing price and assume an order filled at that same known close without a separate execution justification.
Use a fixed ETF universe for the first tests and start only after every required ETF has sufficient observed history.
For an individual-stock adaptation, use point-in-time membership, delisted securities, and corporate-action history.
Reject missing or non-finite inputs rather than allowing an absent asset to win a rank.
Use adjusted prices for a total-return adaptation and separately model raw execution prices and distributions where required.
Do not invent pre-inception ETF prices or silently mix synthetic mutual-fund history with tradeable ETF bars.

## 1. Public Zorro ETF rotation

Source is [S1](https://robotwealth.com/etf-rotation-strategies-in-zorro/), published 2018-06-04.
The complete original Lite-C listing and asset-list CSV are embedded in the post.
There is no need to use the member-only improved sizing script to inspect the basic strategy.

The universe is SPY, EFA, and SHY.
EFA is developed-market equities outside the US and Canada, not an all-world or emerging-market fund.
The code sets `DAYS=252`, `NUM_ASSETS=3`, `BarPeriod=1440`, `LookBack=252`, and `MaxLong=1`.
It sets `StartDate=20040101`, `EndDate=20170630`, and `Capital=10000`.
It reads Alpha Vantage end-of-day history with `assetHistory(Name, FROM_AV)`.
The asset list sets leverage to 1, spread to 0.1, and commission to 0.02 in Zorro's asset-list fields.
Those are source configuration values, not a modern broker cost estimate.

At `tdm()==3`, the third trading day of each month, compute `R_i,t,252` for all three ETFs.
The actual code ranks all three ETFs, including SHY, and chooses the largest return.
If SHY ranks first, buy SHY without a positive-return test.
If the highest-ranked ETF is SPY or EFA and its return is positive, buy it and exit SHY.
Otherwise, enter SHY.
The code exits lower-ranked equity positions on the rebalance date.
It does not trade between monthly rebalance dates and declares no stop, profit target, or time exit.
Allocation intent is one unlevered holding using starting capital plus realized profits, `Margin = Capital + WinTotal - LossTotal`.
The code does not explicitly resize an unchanged holding to 100% of current marked equity each month.

The prose instead compares SPY and EFA first, then holds their positive-return winner or SHY.
That prose can hold equities when SHY has the largest return, whereas the code holds SHY.
The code also does not explicitly exit an already-held highest-ranked equity in the nonpositive-winner fallback branch.
If all three returns are negative and an equity ranks first, that branch can retain the equity while entering SHY.
This is a source-code edge case, not a demonstrated Zorro run in this research.
The code does not define tie handling, dividend adjustment, bar timestamp alignment, or a separate order-fill timestamp.
A source-engine reproduction must resolve those details before calling its result an exact replica.

This is naturally long-only and uses no short, futures, or intraday signal data.
Screener can represent the return calculation and long entries, but a faithful implementation must coordinate the three-asset comparison, SHY fallback, and third-session portfolio transition.
Replacing fallback SHY with cash, skipping the newest month, or using daily rank exits is an adaptation.
A cleaned single-holding version that exits every nonselected asset is also an adaptation to the listing.

## 2. Modular dual momentum

Source is [S2](https://robotwealth.com/dual-momentum-review/), published 2017-04-28.
Every month, compare the prior twelve-month returns of two related assets.
For module assets `a` and `b`, select `j = argmax(R_a,t,12m, R_b,t,12m)`.
Hold `j` if its return is positive, otherwise hold treasury bonds or investment-grade bonds.
Exit the previous module holding when the selected asset changes.
The public post identifies equity pair SPY/CWI and bond pair CSJ/HYG.
It also shows a combined portfolio with 60% in the equity module and 40% in the bond module.
The public source does not identify the exact defensive ETF for each module, its monthly session, fill price, intra-module sizing implementation, or tie and zero-return rules.
The original research code is available to members, not as a public linked listing.
The post explicitly excludes transaction costs and ETF distributions from its reported simulations.
Do not claim a dividend-adjusted, net-of-cost test replicates those published charts.

The source is long-only and monthly, and daily ETF data can support a specified adaptation.
A 252-session lookback is a proposed daily approximation to twelve calendar months, not a disclosed parameter here.
One slot per module does not reproduce the 60/40 portfolio unless the allocation mechanism enforces those weights.
Current CSJ history also needs a verified successor-ticker and fund-change policy rather than silently substituting a present-day fund.

The post distinguishes this rule from Antonacci's GEM.
As reported there, GEM uses the US equity return versus short-term treasury returns as the equity risk gate, irrespective of foreign equities' performance.
These Robot Wealth variants use the selected asset's positive absolute return instead.
Do not label them exact GEM replicas.
The linked [author FAQ](https://www.optimalmomentum.com/faq/) and [original paper](https://ssrn.com/abstract=2042750) identify the original research, but this inventory does not claim to reproduce their complete rules.

## 3. Global macro dual-momentum rotation

Source is also [S2](https://robotwealth.com/dual-momentum-review/).
Rank a cross-asset ETF universe by formation-period returns and buy up to the best three whose returns are positive.
The displayed example uses a six-month formation period.
Selection is monthly in the article's dual-momentum context, but an exact session is not given.
A natural mathematical transcription is `S_t = {i in top_3(R_i,t,6m): R_i,t,6m > 0}`.
The post does not specify whether unfilled allocations become cash, how winners are weighted, or whether existing winners are resized.
Selling assets that leave `S_t` is necessary for a rotation adaptation, but the public post gives no executable exit schedule.

The stated universe covers Russell 1000 stocks, S&P 500 stocks, FTSE-listed stocks, European stocks, Japanese stocks, Asia-Pacific ex-Japan stocks, 7-10 year Treasuries, 1-3 year Treasuries, and gold.
These are ETF exposures, not a request to rank every underlying stock.
The exact nine ticker symbols are not listed in the article text.
Code remains member-only.
The author explicitly says the displayed parameter set was among the best for that universe.
Use fixed parameters and unseen periods rather than treating the chart as an unbiased estimate.

This can become a long-only daily-data strategy in screener after the universe and allocation gaps are specified.
Mapping six months to 126 sessions, equal-weighting three slots, leaving unused slots in cash, and using next-open fills are all adaptation choices.
It is not equivalent to screener's `momentum_12_1` rule.

## 4. Monthly top-four momentum with inverse volatility

Source is [S3](https://robotwealth.com/momentum-dead-alive/), published 2019-04-29.
Each month, rank assets by trailing six-month returns and select the top four.
Weight those four inversely to their volatility over the preceding three months.
The allocation formula implied by that instruction is `w_i,t = (1 / sigma_i,t,3m) / sum_j_in_S(1 / sigma_j,t,3m)` for selected assets, and zero otherwise.
Unlike the dual-momentum variants, this rule does not state a positive-return gate.
Therefore, it can hold the least-negative assets when the whole universe falls.
The rebalance changes membership and weights monthly, not after a fixed six-month holding period.
There is no stated stop or take-profit rule.

The analysis shows eight ranks, but the article text does not disclose all eight tradeable tickers or their synthetic-history mappings.
It does not define volatility as simple-return or log-return standard deviation, its sampling frequency, degrees of freedom, annualization, missing-data treatment, zero-volatility handling, or the precise monthly session.
The earlier factor study uses formation and forward holding periods in `{1,3,6,9,12}` months.
Those 25 analysis combinations are not 25 fully specified tradeable strategies.
The source says its long history uses synthetic index, mutual-fund, and related data and reports before-cost results.
The public article provides no original strategy-code link.

The strategy is long-only and can use daily ETF bars.
An adaptation could use 126-session returns and sample standard deviation of 63 daily simple returns, with positive finite volatility required.
Those are proposed parameters, not recovered defaults.
Screener's candidate ranking and equal-capital slots do not by themselves deliver monthly inverse-volatility target weights.
Changing inverse-volatility allocation to equal slots must appear in the result label.

## 5. Small momentum tilts on baseline risk-premia weights

Source is [S3](https://robotwealth.com/momentum-dead-alive/).
The source keeps some exposure to every asset and slightly overweights high relative momentum while underweighting low relative momentum.
The baseline is long-only, always invested, and reweighted monthly.
The post does not disclose the exact baseline weight formula, momentum-to-weight function, tilt strength, normalization, or bound constraints.
There is no exact implementable formula to transcribe from the public evidence.
Inventing `w_i = normalize(baseline_i * (1 + k * momentum_i))` would be a new strategy, not a replica.
Do not prioritize this for an unattended backtest without the original member code.

## 6. Long-only price/SMA crossover

Source is [S4](https://robotwealth.com/parameter-optimisation-for-systematic-trading/), published 2020-04-19.
The post includes original Lite-C code for a simple long-only system used to demonstrate optimization.
`MA_t,N = sum_k=0..N-1(C_t-k) / N`.
Enter long on `crossOver(closes, ma)` and exit long on `crossUnder(closes, ma)`.
The declared default is `N=100`, and the example optimization grid is 20 through 200 in steps of 20.
`LookBack=200` covers the largest tested window.
There is no short order, stop, target, fixed holding period, or portfolio rebalance in this example.
The snippet does not declare a ticker, `BarPeriod`, date range, sizing, transaction costs, or execution price.
It is an educational rule, not a published validated daily equity portfolio.

For a daily screener adaptation, compute SMA100 on completed daily bars and use next-session open orders.
Use the explicit crossing convention `C_t-1 <= MA_t-1 and C_t > MA_t` for entry and `C_t-1 >= MA_t-1 and C_t < MA_t` for exit.
These equality conventions are a proposed implementation contract because the excerpt calls Zorro helpers without defining them.
Do not replace a crossing entry with the level condition `C_t > MA_t` without recording the change.
The difference matters when a run starts in an existing uptrend or when a portfolio slot becomes free midtrend.
Screener's existing `ma_cross` is EMA10/EMA20, not price/SMA100, so its name does not establish a match.
This is the easiest independent long-only signal to implement after the execution and sizing assumptions are fixed.

## 7. BTC price/SMA20 signal and continuous ratio

Source is [S5](https://robotwealth.com/trading-signals-in-high-definition/), published 2025-08-15.
Its linked [original notebook](https://github.com/Robot-Wealth/trader-tales/blob/2accaec819c526607ab0026aaa2812e5d554e8c3/quant-techniques/Convert_Binary_Signals_to_Continuous.ipynb) was read at repository commit `2accaec819c526607ab0026aaa2812e5d554e8c3`.
The notebook uses the packaged BTC data from `RWLab/tlaqData`, not equity or ETF data.
It computes `MA_t = SMA20(Close)_t`.
The binary signal is `s_t = +1` when `Close_t >= MA_t`, otherwise `-1`.
The continuous feature is `q_t = Close_t / MA_t`.
It sets `position_t = q_t / max_all_sample(abs(q))` and multiplies this position by the provided `fwd_simple_return`.
It plots the cumulative sum of these returns, not compounded NAV.
A comparison also scales continuous positions by full-sample `sd(binary_returns) / sd(continuous_returns)`.

For positive prices, `q_t` is always positive, so the continuous example is not a signed long/short ratio-minus-one strategy.
The full-sample maximum and volatility scaling use future observations and cannot be carried into a causal backtest unchanged.
The public notebook does not calculate the forward-return horizon; it loads precomputed fields from the packaged dataset.
It defines no executable costs, fills, cash constraints, rebalance buffer, entry, exit, or maximum exposure contract.
The article expressly says its cumulative signal returns are not a real-world backtest.

The binary source is long/short BTC and requires a short-side implementation for a replica.
A daily equity long/cash adaptation can hold while `C_t >= SMA20_t` and go flat while `C_t < SMA20_t`, using next-open fills.
Unlike section 6, this is a level rule rather than crossing-only entry.
A causal long-only continuous exposure rule needs a separately fixed scaler and cap and is a new adaptation.
Do not advertise the article's attribution charts as a daily equity backtest.

## 8. Recent-high crypto trend and leveraged simulation

Source is [S6](https://robotwealth.com/how-much-damage-can-i-do-turbo-punting-shitcoins/), published 2023-12-13.
The opening describes getting long assets within five days of their 20-day high.
It does not provide complete live-universe, weighting, exit, or execution rules for that statement.
A daily-equity adaptation could define `age20_t = 19 - argmax_first(C_t-19,...,C_t)` and hold when `age20_t < 5`.
The strict boundary, close-based high, and next-open fill in that formula are explicit adaptation choices.
Do not call it the original strategy.

The actual code in the post instead uses synthetic hourly prices and all-time highs.
It sets `hold_period=5*24` hours and takes position 1 when time since the prior bar's all-time-high index is between 1 and 120 inclusive and that high index exceeds 1.
Otherwise the position is zero.
It attributes returns from the high's hourly price, so a daily next-open implementation changes its execution.
Its position-index calculation uses `match(all_time_high, unique(price))`, which needs scrutiny if repeated prices exist.
The leverage rule is `L(D) = max(0, min(5 - (5/0.9)*D, 5))`, where `D` is drawdown from the highest portfolio equity.
The illustrated leverage refresh is every 24 hours or when positions change.
The simulation uses 100 paths, 100 days of hourly observations, seed 503, annual drift -1, annual volatility 1.5, AR coefficient 0.1, jump probability 0.2 per step, jump mean 0, and jump standard deviation 1.
These are simulation assumptions, not daily stock parameters.

This example is long-only but uses crypto, hourly synthetic data, up to 5x leverage, and a 90% drawdown endpoint.
A cash-only daily equity recent-high rule cannot reproduce that experiment.
Use it only as a clearly named adaptation after specifying the daily age boundary and whether repeated highs reset the clock.

## Related material and exclusions

[S7](https://robotwealth.com/using-digital-signal-processing-in-quantitative-trading-strategies/) is a cycle-filter experiment on EUR/USD, USD/JPY, and SPX500 with H1, H4, and D1 components.
It reverses long and short on crossings of an Ehlers trigger and band-pass output and later suppresses reversals when `DominantPeriod >= 45`.
The executable listing uses `crossOver(Trigger, BP)` for long and `crossUnder` for short, whereas surrounding prose describes the reverse direction.
It relies on Zorro filters, hourly base bars, and zeroed costs.
It is not a clean daily ETF momentum strategy, and dropping its short leg or changing SPX500 to SPY is an adaptation.

[S8](https://robotwealth.com/using-exponentially-weighted-moving-averages-to-navigate-trade-offs-in-systematic-trading/) gives `EWMA_t = (1-lambda)*Y_t + lambda*EWMA_t-1` and an example `lambda=0.9`, warmup 20.
It compares SPY/TLT prices and correlations but does not specify an entry or exit rule.
An EWMA calculation tutorial is not evidence for an EWMA crossover strategy.
[S9](https://robotwealth.com/quant-signal-trade-offs-in-the-real-world/) discusses forecast persistence and costs, not a fully specified daily trade.
[S10](https://robotwealth.com/to-trend-or-not-to-trend-wrong-question/) discusses the rationale for trend and warns against unexplained pattern searches, but gives no daily momentum system.
[S11](https://robotwealth.com/three-types-of-systematic-strategy-that-work/) and [S12](https://robotwealth.com/harvesting-risk-premia/) supply strategy and allocation context rather than additional reproducible momentum rules.

[S13](https://robotwealth.com/exploring-the-rsims-package-for-fast-backtesting-in-r/) demonstrates a long/short cryptocurrency portfolio and a no-trade buffer.
Its exact momentum formation rule is not disclosed by the loaded weight dataset.
Its example buffer 0.06 is fitted to that crypto example, not a daily equity default.
[S14](https://robotwealth.com/a-simple-effective-way-to-manage-turnover-and-not-get-killed-by-costs/) combines Binance perpetual funding, reversal, and recent-high features on the top 30 assets by trailing 30-day dollar volume.
Its code uses `momo = lag(close - lag(close,10)/close)`, not a conventional ten-day return.
Operator precedence makes that expression approximately a price-level feature, and its decile weight is negated, so it must not be silently corrected and called a momentum replica.
Its combined score is `0.5*carry_weight + 0.2*momo_weight + 0.3*breakout_weight`, normalized by the sum of absolute scores.
It requires short positions, funding cash flows, perpetual-futures margin, and a crypto calendar.
Removing funding and shorts produces a materially different strategy.

[S15](https://robotwealth.com/reduce-trading-costs-and-boost-profits-with-the-no-trade-region-strategy/) is a seven-ETF equal-dollar allocation example, not a momentum signal.
The [linked original notebook](https://github.com/Robot-Wealth/rsims/blob/116f47e9692acc242fd7af17553514c50a21b7f0/examples/A_simple_trick_to_minimise_trading_costs.ipynb) uses $50,000, no profit reinvestment, and illustrative commission parameters of $0.0035/share, $0.35 minimum/order, and 1% maximum/order.
It tests buffers from 0 to 0.8 by 0.025 and illustrates 0.5.
The article describes a relative buffer around target weight and trading back to target under minimum commissions.
However, the inspected [current minimum-commission C++ function](https://github.com/Robot-Wealth/rsims/blob/116f47e9692acc242fd7af17553514c50a21b7f0/src/positionsFromNoTradeBufferMinComm.cpp) computes absolute bounds `w - trade_buffer/2` and `w + trade_buffer/2`, clipped at zero to preserve sign.
Its header comment still describes relative bounds and therefore disagrees with its executable body.
The [fixed-percentage function](https://github.com/Robot-Wealth/rsims/blob/116f47e9692acc242fd7af17553514c50a21b7f0/src/positionsFromNoTradeBuffer.cpp) uses the same absolute half-width but trades to the nearest boundary.
Both functions force a target of zero or NA flat.
Current source is pinned to repository commit `116f47e9692acc242fd7af17553514c50a21b7f0`, not the historical version used for the post.
Do not import any of these buffer numbers into an equity test without declaring units and verifying the source version.

[S16](https://robotwealth.com/rw-pro-webinar-28-mar-2024-trade-sizing-and-rebalancing-trend-following-on-spy/) advertises SPY trend following but shows an access restriction instead of rules.
[S17](https://robotwealth.com/rw-pro-webinar-27-july-2023-momentum-of-trend-factor-in-stocks-munging-options-data/) advertises a stock momentum-of-trend factor but also restricts access.
Neither supports a public formula or parameter claim.
Do not guess a SMA200 SPY strategy from the webinar title.

## Screener fit and implementation order

The checked repository supports expression entry/exit rules and prepared-bar `rank_score` values.
Evidence is `screener/strategies/plugins/ma_cross.py`, `screener/strategies/plugins/momentum_12_1.py`, and `screener/backtester/rolling_simulation.py`.
The existing momentum score is `C_t-21 / C_t-252 - 1`, which skips the newest month and differs from every unskipped return rule above.
The existing `ma_cross` uses EMA10/EMA20, which differs from the SMA100 and SMA20 price rules.
Rank exits can close positions that leave a ranked set, but they are not proof of source-compatible third-session rotation or full target-weight rebalancing.
Equal-capital slots also do not establish inverse-volatility weights or 60/40 module allocations.
These observations are a source inspection, not an end-to-end test of any proposed new rule.

1. Implement the long-only SMA100 crossing adaptation first if the parent needs the simplest independent daily equity rule.
2. Implement the 252-session three-ETF code rule next for the closest public daily ETF specification, with a separate prose variant and explicit fallback edge-case policy.
3. Implement the six-month top-four ETF adaptation only after fixing its universe, volatility convention, monthly date, and target-weight allocation.
4. Consider the positive-return top-three global macro adaptation after recovering or explicitly choosing the nine ETF mappings and unused-allocation rule.
5. Defer the baseline momentum tilt, private stock factor, and private SPY strategy until original rules are available.

For all runs, retain the source ID, replica/adaptation label, universe, lookback convention, signal timestamp, fill timestamp, sizing method, cash return assumption, and costs in the result metadata.
Screener's risk-free hurdle is not interest credited to idle cash, as documented in `CONTEXT.md`.
Adding profit targets, automatic stop losses, maximum holding periods, regime gates, or earnings exclusions changes these source rules.
No files outside the two requested inventory outputs were changed for this research.
