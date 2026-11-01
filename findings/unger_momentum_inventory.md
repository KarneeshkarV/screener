# Unger Academy daily momentum and trend-following inventory

## Result for the parent task

Research date: 2026-10-02 UTC.
The scope is public Unger Academy material linked from the English blog and its navigation.
I treated ETFs as in scope even when the ETF holds gold rather than equities.
No strategy code was implemented, no backtest was run, and no paid access was used.

The useful daily candidates are:

| ID | Rule | What can be implemented | Decision |
| --- | --- | --- | --- |
| U01 | GLD 170-day Donchian | Daily channel orders with 7% stop, 10% target, 5% breakeven activation, and 35-bar time exit | Best parameterized ETF candidate, but source execution details still need conventions |
| U02 | Nasdaq stocks 5-day Donchian | Long at the 5-day high, flat at the 5-day low | Complete basic signal, but the source says it is not tradable after costs |
| U03 | Close versus SMA200 | Bullish above, bearish below | Ready as a daily filter after the daily-bar convention is made explicit |
| U04 | SMA50 versus SMA200 | Golden Cross and Dead Cross | Ready as a daily filter, not a fully specified portfolio strategy |
| U05 | SMA200 slope over 50 bars | Compare SMA200 now with SMA200 50 bars ago | Ready as a daily filter, not a fully specified portfolio strategy |
| U06 | Positive first half, buy second half | Buy the Dow after a positive January-to-June move | Testable on a Dow ETF only as an explicit adaptation, with timing correction |
| U07 | Best-stock monthly rotation | Top 10 by composite momentum | Blocked by missing momentum lookbacks and weights |
| U08 | Best-sector ETF rotation | Top ETF by composite momentum over 1, 3, 6, and 12 periods | Blocked by missing weights, complete universe, and period units |
| U09 | NVIDIA multiday Donchian | Long-only breakout with volatility filter and timed exit | Blocked by missing numeric inputs and bar interval |
| U10 | Nasdaq stocks 24-week Donchian | Weekly breakout and opposite-channel exit | Useful longer-horizon lead, but not a native daily rule |

U03 through U05 are the safest rules to pass to the parent as directly usable filters.
U01 has the fullest ETF trade specification, but it is a gold ETF and uses intrabar stop orders.
U02 is a useful negative control, not a recommended strategy.
Do not describe U06 through U10 as exact daily reproductions.
All reported source returns are publisher claims, not independent validation.

## Evidence and crawl coverage

The first-party robots file links the English sitemap index. [S24]
The sitemap index links `post-sitemap.xml`, `page-sitemap.xml`, and `category-sitemap.xml`. [S25]
The English post sitemap contained 992 URLs at retrieval, including 913 `/blog/` URLs, 78 `/posts/` URLs, and one other URL. [S26]

I fetched the blog index and pages 2 through 31.
Pages 1 through 30 each contained 30 article cards, and page 31 contained 13.
The cards yielded 913 unique article URLs, matching the number of `/blog/` URLs in the post sitemap.
This establishes coverage of the visible English blog article titles at retrieval, not full-text coverage of every article.

I searched the collected titles and sitemap URL names for `momentum`, `trend`, `stock`, `ETF`, `equity`, `moving average`, `Donchian`, `breakout`, `rotation`, `allocation`, `relative strength`, `ROC`, `RSI`, `MACD`, `ADX`, `Bollinger`, and `Keltner`.
I checked additional stock and ETF leads from the title list, including `bias-on-spy`, the Tesla 2025 article, and the Dow half-year article.
The JSON file contains the exact index-page URLs, article counts, search terms, and 46 selected article URLs fetched successfully.
The selected set includes background indicator articles and exclusions, not 46 daily strategies.

I read public article text and public video transcripts embedded in the articles.
Some pages have no useful transcript, notably the top-10 best-stock article.
I did not watch the embedded videos or extract code visible only inside video frames.
I did not inspect all 78 `/posts/` pages, the Italian site, all categories, or every article body.
I did not use a site search endpoint because the robots file disallows `/?s=`.
There can be relevant rules inside articles whose titles do not match the search terms.
This inventory does not establish that every Unger strategy is covered.

The `agent-browser` skill and its installed core guide were read before retrieval.
Browser launch failed because Chromium had no usable sandbox.
`agent-browser read` worked without browser launch for initial index pages and articles.
Some later reads timed out, so I used ordinary public HTTP HTML retrieval for the remaining selected pages and index pages 30 and 31.
No login or access restriction was bypassed.

## Native daily or daily-compatible rules

### U01: GLD 170-day Donchian

The source explicitly uses GLD daily bars and a 170-period Donchian channel. [S01, section "A Strategy on GLD"]
The upper channel is the highest high over 170 bars, and the lower channel is the lowest low over 170 bars.
The source buys with stop orders at the upper channel and sells with stop orders at the lower channel.
It also mentions time exits for both long and short positions, so this must not be silently labeled long-only.
The public narration does not fully establish whether every lower-channel order exits, reverses, or opens a short position.

The source sets fixed position value to $10,000 per trade, without profit reinvestment.
The stop loss is $700 per position, equivalent to 7% of the allocated position value.
The profit target is 10%, equivalent to $1,000 on that allocation.
The breakeven stop activates after at least $500 open profit, equivalent to 5%.
The time exit is after 35 daily bars.
These are position-level monetary rules, not ATR rules or percentages of total account equity. [S01]

Use the stated 170 bars, not the source's loose description of "5-6 months".
170 trading sessions are not five or six calendar months.
The source reports about three trades per year and warns that the small sample needs more validation. [S01]

Missing details are channel inclusion, exact order issue timing, reversal behavior, integer share rounding, entry-bar counting, exact time-exit inequality, and the order type for the time exit.
The public text also omits how the platform handles a bar that reaches entry, stop, target, and breakeven levels in the same session.
A daily OHLC backtest cannot determine that path without a conservative convention or lower-frequency data.

A proposed research convention is to calculate channels from completed bars at close `t` and issue orders for session `t+1`.
A long-only version can buy at the upper channel and close at the lower channel, but removing shorts is an adaptation.
Use the original position-level monetary thresholds after share rounding instead of assuming exact entry-price percentages.
If a gap crosses a stop, fill at the available opening price rather than the stale stop price.
Record any time-exit counting and ambiguous-bar policy before testing.
These conventions are proposed, not quoted source rules.

### U02: Nasdaq stocks 5-day Donchian

The source tests the historical Nasdaq 100 stock universe, including delisted stocks and date-specific index membership. [S02, section "Testing on a Shorter Time Frame"]
It uses daily bars and $10,000 per trade.
The trend-following rule buys when price breaks above the highest high of the last 5 bars and closes the long position when price breaks below the lowest low of the last 5 bars.
No extra stop, profit target, or time exit is specified for this basic version. [S02]

The source reports average trade profit of only $2.88 on $10,000 exposure and explicitly calls this strategy non-tradable after commissions and slippage. [S02]
Keep it as a baseline or negative control.
The source mentions 6, 7, and 20 periods as educational variations, not validated replacements for 5.

Missing details are precise order issue timing, channel inclusion, share rounding, available portfolio capital, and simultaneous-signal allocation.
A proposed daily convention is to use five completed sessions and next-session stop orders at the channel levels.
Do not use the current session's high to set an order that supposedly filled earlier that session.

### U03: Close versus SMA200

The source uses a simple moving average, not an EMA. [S03, sections "Approach 1" and "The Trading System Used for the Test"]
Bullish regime means closing price above SMA200, and bearish regime means closing price below SMA200.
The examples include SPY and a basket of equity, bond, gold, and silver ETFs.
The narration describes daily regime changes, but does not explicitly state every test chart's bar interval.
Daily close inputs are therefore an explicit application convention.

The source prefers this method as a regime filter because it identifies a regime change sooner than the crossover and slope alternatives. [S03, section "What's the Best Way to Use Moving Averages?"]
The rule is implementable as a filter without inventing an exit strategy.
A standalone long/cash system that holds the ETF while close exceeds SMA200 and sells otherwise is a proposed adaptation.
For that adaptation, calculate after the daily close and trade at the next session's open.
Equality, cash return, dividends, costs, and initial state must be specified separately.

### U04: SMA50 versus SMA200

The Golden Cross occurs when SMA50 crosses above SMA200.
The Dead Cross occurs when SMA50 crosses below SMA200. [S03, section "Approach 2"]
The source describes these as bullish and bearish regime signals.
A daily filter can use `SMA50 > SMA200` as the bullish state after a confirmed cross.
Daily close inputs, equality handling, and the initial state before the first observed cross remain application conventions.

There is no complete portfolio sizing or execution specification in the public text.
A standalone long/cash adaptation would enter after an upward cross and exit after a downward cross, at the next daily open.
Do not report that adaptation as the exact publisher backtest.

### U05: SMA200 slope over 50 bars

The source connects the current SMA200 value to its value 50 periods earlier. [S03, section "Approach 3"]
Positive difference means bullish regime, and negative difference means bearish regime.
This is `SMA200(t) - SMA200(t-50)`, not SMA50 slope, a one-day change, or linear regression.
A daily application needs 250 closes before both SMA values are available.

The filter is implementable with daily close inputs after making that bar convention explicit.
Zero slope and first-state handling are not specified.
A next-open long/cash strategy based on slope sign is a proposed adaptation, not a fully stated source strategy.

### U06: Dow positive-first-half rule

The source tests the Dow Jones Industrial Average index from 1920 onward, not DIA or an executable stock basket. [S04]
It records the opening value on the first trading day of January and the closing value on the last trading day of June.
If June's closing value exceeds January's opening value, it buys for the second half of the year.
The illustrative allocation is $1,000,000 per trade because the tested index has a large numeric level.
The source adds no stop loss or other protective exit. [S04, sections "Analyzing the code" and "Examining the equity line"]

There is a timing conflict in the narration.
It says the entry occurs at the first available July bar and the position closes at year-end.
Its code explanation detects July on the current bar and says to sell next bar at market after the year changes.
Without the actual code, a daily next-bar implementation could enter on July's second session and exit on January's second session.
Do not silently call that July-open through December-close.

A proposed tradable adaptation uses DIA, evaluates the first-half test after the final June close, buys at the first July open, and exits at the first January open.
This changes the instrument and resolves timing prospectively, but does not reproduce the historical index test.
A December market-on-close exit would be another adaptation and needs a separate execution assumption.
Use only DIA's available history and actual ETF returns, not synthetic century-long ETF fills.

## Incomplete daily-compatible leads

### U07: Best-stock monthly rotation

The public page says to hold the 10 best stocks, selected monthly using composite momentum, for S&P 500, Nasdaq 100, and Russell 2000 tests. [S05]
The page does not give the momentum horizons, component weights, full ranking formula, tie handling, weights per holding, rebalance day, or execution time.
Its transcript heading contains no substantive transcript.
This is not enough for an exact implementation.
Do not borrow the sector ETF horizons from U08 and attribute them to U07.

### U08: Best-sector ETF rotation

The source chooses the single best sector ETF from a list of XL-series ETFs using composite momentum over 1, 3, 6, and 12 periods with different weights. [S06]
It explicitly names XLE, XLF, XLI, and XLK as examples, not as the complete tested universe.
It tests weekly, monthly, two-month, and quarterly rotation.
The narration does not define whether each momentum period is a week, a month, or the selected rotation interval.
The numeric component weights are not given.

The source rejects or questions the weekly, monthly, and two-month versions and doubts the quarterly result because of its sharp difference from the two-month result. [S06]
It is not an endorsement of a robust ETF momentum strategy.
The formula, complete ETF list, ETF inception handling, negative-momentum cash rule, ranking ties, rebalance date, and order timing are missing.
Equal weights, calendar-month horizons, or a current 11-sector universe would all be proposed adaptations.
None is justified as an exact source reconstruction.

### U09: NVIDIA multiday Donchian

The August 2026 winner is a long-only multiday NVIDIA Donchian strategy with a volatility filter. [S07]
The exits are a fixed number of bars, stop loss, or profit target.
The illustrative size is $100,000 per trade.
The article gives no channel length, volatility formula or threshold, bar interval, holding limit, stop amount, target amount, or order timing.
The source directs readers to its member strategy service for code.
I did not access that service.
The strategy is blocked, and "multiday" must not be converted to "daily bars" without evidence.

## Longer-horizon source and proposed daily adaptation

### U10: Nasdaq stocks 24-week Donchian

The same comparison used for U02 first tests a 24-period channel on weekly stock bars. [S02, section "Strategy Adjustment"]
It enters with stop orders at the highest high of the last 24 weeks and changes the earlier mean-reversion logic to trend following.
The paired lower-channel exit follows the stated inversion of the original high/low logic, but the weekly trend paragraph does not repeat the full exit instruction.
The capital allocation is $10,000 per trade, and index membership is point in time.
The source reports a better weekly trend result than the weekly mean-reversion result. [S02]

For a daily-data test, preserve 24 completed exchange-calendar weeks and update their channel after the final session of each week.
Use the resulting fixed levels for the next week.
This is a proposed daily-data execution convention for a weekly signal.
Replacing 24 weeks with 120 daily bars is a different adaptation and is not calendar-equivalent.
Weekly opposite-channel exit wording and precise order timing still need confirmation before an exact-reproduction claim.

## Out-of-scope rules and near matches

### Intraday stock rules

- Eni uses hourly bars, an 18-bar channel, and entries from 11:00 to 16:00 exchange time.
  Its final stop is 5%, target is 7%, time exit is about 90 hourly bars, and prior-day absolute body must be more than 25% and less than 75% of prior-day range. [S08]
  The source corrects a displayed 17:00 end time to 16:00.
  A two-day channel with a 10-day exit would be an adaptation, not the source strategy.
- The Tesla open-code article uses 15-minute bars and current-session high/low breakout levels.
  Its initial 10:00 to 15:30 window becomes 11:30 to 15:15, with 2.5% stop and 7.5% target on $10,000 exposure. [S09]
  Daily OHLC cannot reproduce the current-session levels inside that window.
- The Amazon open-code article uses intraday current-session breakout levels, a final 10:45 to 14:30 window, and prior-day body/range below 0.5. [S10]
  It states a 2% stop, but its early code explanation gives 7% target while its starting setup gives 4% target.
  That conflict and the intraday window block a daily reproduction.
- Google uses 15-minute bars, open above prior close, current price above open, and a prior-day-high stop entry from 10:00 to 16:00.
  Its allocation is $10,000, stop 1%, target 3%, and dynamic exit is the lowest low of two preceding sessions. [S11]
- The paired Tesla contraction system uses 60-minute bars, 8-bar versus 10-bar ranges, a 9-bar setup life, price below MA100, and latest daily True Range below its 7-period average.
  It sets a 4.5% stop, 11% target, and exit after more than 5 days, but omits the contraction ratio threshold. [S11]
- Netflix uses 15-minute data1 and daily data2, buys a current-week-high breakout from Tuesday, and applies an undisclosed volatility filter and timed exit. [S12]
  Microsoft in that article uses daily Williams %R reversal logic with undisclosed inputs.
- The later Tesla article uses a 15-minute momentum system based on previous close plus a standard-deviation amount over 15 hours for longs.
  It omits the multiplier, short lookback, and target amount, although it states an exit within five days. [S13]
- The Amazon and Starbucks article includes a 60-minute Amazon rebound rule below SMA100 with a 9-bar breakout.
  Starbucks buys below a four-session low, which is reversal rather than momentum, and its exit rules are missing. [S14]

### Futures and crypto examples

- The general Donchian study uses 60-minute bars, skips the first session bar, starts at 20 bars, and tests 10 through 200 in steps of 10 on a futures-oriented basket. [S15]
  It is not evidence for a daily equity 20-day rule.
- The momentum oscillator article uses `close(t) - close(t-N)` on daily crypto, starting at 30 bars and later considering 50 after a parameter study.
  Its futures comparison also uses 50, but it recommends momentum mainly as a filter rather than as a trigger. [S16]
  No equity or ETF validation is supplied.
- The moving-average backtest article uses 240-minute futures bars with 200 and 250 single-average tests and 50/200 and 50/300 crossover tests. [S17]
  Its narration also mentions 50/250 in one comparison.
  Those futures results are separate from U03 through U05.
- The MACD strategy uses 12/26/9 daily inputs on a futures basket, adds a MACD sign filter, and exits after two bars of convergence.
  Its later target multiplier is 2. [S18]
  This does not establish an equity or ETF MACD strategy.
- The ADX example uses 14 daily periods, directional-indicator crossings, and next-day stop entries at the signal bar high or low on futures.
  Its filter accepts ADX above either directional indicator, and its trend version performs poorly before a reversal adaptation. [S19]
- The RSI example uses 14 daily periods with 30/70 thresholds on futures, then changes to an intraday Mini S&P system with daily RSI2, SMA5, SMA200, and a chosen RSI threshold of 20. [S20]
  These are futures and reversal examples, not daily equity momentum rules.
- The simple-strategy comparison uses a four-day stop-and-reverse channel with a $2,000 stop on commodity futures.
  The other two comparison strategies use 60-minute bars and end-of-session exits. [S21]
- The newer Nasdaq comparison is explicitly 15-minute futures trading, with 6 ATR long and 8.5 ATR short thresholds in one system and 65%/120% of prior-session range in the other. [S22]
  ATR length and several filters remain undisclosed.
- The breakout-channel guide uses daily 10-high/2-low channels on cryptocurrencies, not stocks or ETFs. [S23]
- The Supertrend article tests hourly gold with default ATR10 and multiplier 3, then chooses ATR18 and multiplier 5. [S27]
- The Bollinger Band Width article uses prior-session breakouts and end-of-day exits on futures, with a final 0.5% bandwidth threshold. [S28]

### Daily rules that are not momentum

The daily Nasdaq Bollinger rule has explicit next-bar market timing, close-based bands with length 5 and 1.5 standard deviations, and a close-above-SMA200 entry filter. [S29]
It buys after a downward lower-band cross and exits after an upward upper-band cross, using $10,000 per trade and point-in-time membership.
It is mean reversion, despite the trend filter, so do not add it to the momentum set.
The same article proposes a band-filter plus low-limit-entry version, but does not fully define the low reference in its text.

The SPY bias article uses 15-minute bars, Monday 15:45 signal submission, a Thursday exit about one hour before close, a 3% stop, a 0.5% preceding-week decline condition, and no May entries. [S30]
Its narration about Friday close and Monday open does not fully resolve the decline calculation.
It is a calendar/reversal bias strategy, not daily trend following.

The SMA article also presents RSI2 mean reversion on SPY and TLT with 5/95 crossings and a five-period moving-average exit. [S03]
Do not relabel it as momentum because it has an SMA200 filter.

The sideways-stock series chooses middle-ranked stocks rather than winners. [S31]
It uses monthly one- or six-month rankings, then weekly five-day rankings with weekend rotation.
Examples include 50 S&P 500 stocks, 100 Russell 1000 stocks, 10 Nasdaq stocks, and 5 Dow stocks.
The Russell phrase "450th to 550th" conflicts with a count of 100 if both endpoints are inclusive.
The source warns about the cost of weekly rotation and does not recommend the weekly Nasdaq version.
This is a useful comparison, not a top-momentum selection rule.

## Proposed adaptation policy for screener research

Use completed daily bars for signals and next-session orders unless a source explicitly gives another executable timing rule.
Do not substitute same-close fills for next-open fills.
Keep daily regime filters separate from trade entry and exit logic.
Keep source thresholds, proposed thresholds, and missing values in separate fields.

Use historical index membership, delisted stocks, and sufficient warmup for universe tests.
The stock comparison explicitly explains why testing today's Nasdaq members over their full past is biased. [S02]
Declare split adjustment, dividend treatment, benchmark return convention, and ETF inception handling before testing.
Use conservative stop fills and explicit treatment of same-bar ambiguity for U01 and U02.
Include commissions and slippage rather than relying on the publisher's gross average-trade claims.
Do not infer parameter quality from a single optimized article or silently replace a poor source rule with a better-looking variation.

The JSON report records missing details and adaptations per rule.
It is a research inventory, not a deployment configuration.

## Primary sources

- [S01] [GLD daily ETF strategy](https://ungeracademy.com/blog/trading-on-etfs-great-strategy-on-gold-open-source-code-easy-explanation-backtest-results).
- [S02] [Stocks: trend following versus mean reverting](https://ungeracademy.com/blog/stocks-trading-trend-following-or-mean-reverting).
- [S03] [Moving averages and market regimes](https://ungeracademy.com/blog/moving-averages-trading-market-regimes).
- [S04] [Dow positive-first-half test](https://ungeracademy.com/blog/the-dow-jones-strategy-tested-on-100-years-of-data-you-need-to-know).
- [S05] [Best-stock composite momentum, part 2](https://ungeracademy.com/blog/equity-indices-and-trading-systems-should-i-buy-the-best-stocks-part-2).
- [S06] [Best-sector ETF allocation](https://ungeracademy.com/blog/asset-allocation-with-the-best-sector-etfs-does-it-work).
- [S07] [NVIDIA Strategy of the Month, August 2026](https://ungeracademy.com/blog/trading-on-nvidia-stock).
- [S08] [Eni Donchian strategy](https://ungeracademy.com/blog/how-to-build-a-trading-system-on-energy-stocks-with-the-donchian-channel-open-source-code).
- [S09] [Tesla open-code strategy](https://ungeracademy.com/blog/trading-system-on-tesla-stocks-script-backtest-and-optimization).
- [S10] [Amazon open-code strategy](https://ungeracademy.com/blog/trading-system-on-amazon-stocks-creating-a-strategy-open-code).
- [S11] [Google and Tesla systems](https://ungeracademy.com/blog/google-tesla-trading-strategies).
- [S12] [Netflix and Microsoft systems](https://ungeracademy.com/blog/trading-systems-us-stocks-netflix-microsoft).
- [S13] [Tesla systems, 2025](https://ungeracademy.com/blog/algorithmic-trading-strategies-tesla-2025).
- [S14] [Amazon and Starbucks systems](https://ungeracademy.com/blog/trading-systems-usa-stocks-amazon-starbucks).
- [S15] [Donchian periods and hourly study](https://ungeracademy.com/blog/donchian-channel-strategy-does-it-work-how-to-choose-the-number-of-periods).
- [S16] [Momentum oscillator study](https://ungeracademy.com/blog/momentum-indicator-is-it-worth-using-it-in-our-strategies).
- [S17] [Moving-average futures backtests](https://ungeracademy.com/blog/moving-averages-types-and-settings-for-trading-systems-backtest-explanation).
- [S18] [MACD strategy, part 2](https://ungeracademy.com/blog/macd-and-trading-systems-or-how-to-build-the-best-strategy-on-macd-or-part-2).
- [S19] [ADX strategy, part 2](https://ungeracademy.com/blog/adx-and-trading-systems-or-how-to-use-the-adx-indicator-or-part-2-2).
- [S20] [RSI strategy, part 2](https://ungeracademy.com/blog/rsi-and-trading-systems-or-best-strategy-with-the-rsi-indicator-or-part-2).
- [S21] [Simple trend, reversal, and breakout comparison](https://ungeracademy.com/blog/the-simplest-trading-strategy-trend-following-mean-reverting-or-breakout).
- [S22] [Nasdaq futures win-rate comparison](https://ungeracademy.com/blog/nasdaq-trading-strategies-win-rate).
- [S23] [Daily breakout channel on crypto](https://ungeracademy.com/posts/the-breakout-channel-indicator-how-to-use-it-in-systematic-trading).
- [S24] [Robots file](https://ungeracademy.com/robots.txt).
- [S25] [English sitemap index](https://ungeracademy.com/sitemap_index.xml).
- [S26] [English post sitemap](https://ungeracademy.com/post-sitemap.xml).
- [S27] [Hourly gold Supertrend study](https://ungeracademy.com/blog/supertrend-indicator-easy-explanation-how-to-use-it-in-a-trading-strategy-open-source-code).
- [S28] [Futures Bollinger Band Width study](https://ungeracademy.com/blog/bollinger-bands-and-trading-system-or-how-to-exploit-the-bandwidth-or-with-code).
- [S29] [Daily stock Bollinger reversal](https://ungeracademy.com/blog/automated-stock-trading-bollinger-bands).
- [S30] [SPY weekly bias](https://ungeracademy.com/blog/bias-on-spy).
- [S31] Sideways-stock series: [part 1](https://ungeracademy.com/blog/trading-systems-tips-or-should-i-buy-sideways-stocks-or-part-1), [part 2](https://ungeracademy.com/blog/trading-systems-tips-or-should-i-buy-sideways-stocks-or-part-2), [part 3](https://ungeracademy.com/blog/trading-systems-tips-or-should-i-buy-sideways-stocks-or-part-3), and [part 4](https://ungeracademy.com/blog/trading-systems-tips-or-should-i-buy-sideways-stocks-or-part-4).
