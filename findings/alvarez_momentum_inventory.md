# Alvarez daily-bar momentum and trend inventory

Research date: 2026-10-03.

## Scope and coverage

This note covers public research on [Alvarez Quant Trading](https://alvarezquanttrading.com/) for equities and ETFs, with proposed US and India research use.
Daily-bar input does not mean daily signals.
The inventory labels daily, weekly, monthly, quarterly, and semiannual schedules separately.
No strategy code was implemented.
The machine-readable inventory is `reports/blog_momentum/alvarez_inventory.json`.

I fetched 20 public blog archive pages, starting at [page 1](https://alvarezquanttrading.com/blog/) and ending at [page 20](https://alvarezquanttrading.com/blog/page/20/).
These pages exposed 193 unique post URLs.
I selected 46 post URLs by momentum, trend, rotation, breakout, volatility, and related titles.
I read 38 unique full public post bodies with `agent-browser read`, including the VPN post before a later repeat request failed.
Eight selected posts remained readable only as archive excerpts.
I also read seven successful category pages across [Trend Following](https://alvarezquanttrading.com/blog/category/trend-following/), [Rotation](https://alvarezquanttrading.com/blog/category/rotation/), and [Breakouts](https://alvarezquanttrading.com/blog/category/breakouts/).
This is a bounded inventory, not exhaustive coverage of the website, comments, videos, external books, or paid services.

Full-post requests later returned HTTP 403 or 406, after one earlier timeout.
A rendered-browser fallback also failed because Chromium had no usable sandbox.
I did not disable the sandbox or try to bypass access controls.
I did not use paid access, submit email forms, register, or download gated spreadsheets.
Image-only result tables and universe tables were not transcribed.
An exact rule available only in those tables remains a gap, not an inferred parameter.

## Classification

- **Fully specified** means that the public article provides signal rules, exits, schedule, and execution timing sufficient for a declared research implementation.
- **Adaptation** means an author-tested modification or a proposed market or execution change, not an unchanged reproduction of the original strategy.
- **Partially specified** means that a material rule, execution detail, universe, or contradictory formula prevents exact reproduction.
- **Mention only** means that only a title, excerpt, or passing reference was verified.

A fully specified signal system is not necessarily a fully specified portfolio backtest.
Costs, cash yield, tie handling, price adjustment, initial holdings, and position sizing remain explicit gaps where the source does not state them.
A parameter list such as 63, 126, 189, and 252 describes separate tests, not one combined strategy.
A reported improvement is a result from the author's sample, not evidence that the same improvement holds in India or after costs.

## Strongest candidates for the main agent

| Priority | Candidate | Schedule | Why test it first | Main limit |
| --- | --- | --- | --- | --- |
| 1 | MA200 with 3 to 5 consecutive closes | Daily | Exact next-open timing, few rules, ETF implementation | Cash yield and cost conventions need declaration |
| 2 | Sector MA200 with 10-day confirmation and IEF | Daily | Explicit allocation and next-open timing in follow-up | India sector basket and reserve are different exposures |
| 3 | Stock rotation ranked by ROC times HV | Monthly from daily bars | Explicit return and volatility formulas, next-open rotation | Point-in-time members and portfolio sizing |
| 4 | 260-day closing-high breakout baseline | Daily | Exact entry and next-open exits, ordinary OHLCV inputs | Published study is not a constrained portfolio |
| 5 | Original VPN filters on that breakout | Daily | Public indicator code and stated cutoffs | Simplified VPN's selected cutoff not recovered |
| 6 | Ten-month absolute momentum | Monthly | Exact, simple ETF rule with next-open execution | Separate ETF tests, not a multi-asset allocation |
| 7 | Three-factor ETF rotation | Monthly from daily bars | Explicit ranks, weights, universe, and timing | Sizing and tie handling require declared assumptions |

For a strictly daily-signal task, start with MA200 confirmation and the 260-day breakout.
For daily-data research with monthly execution, start with ROC times HV and three-factor rotation.
DTAYS is promising weekly breakout research, but missing fill conventions make it a lower-priority exact-replication target.
These priorities rank rule clarity and test feasibility, not expected future return.

## Fully specified daily systems

### MA200 with consecutive-close confirmation

Source: [Reducing Whipsaws When Using 200-day Moving Average for Market Timing](https://alvarezquanttrading.com/blog/reducing-whipsaws-when-using-200-day-moving-average-for-market-timing/), September 27, 2023.
The test range is January 1, 2000 through June 30, 2023.
Buy after the close has been above its own 200-day moving average for N consecutive sessions.
Sell after the close has been below the same moving average for the same N consecutive sessions.
Both orders execute at the next session open.
The explicit rule grid is N equal to 1, 2, 3, 4, or 5.
The author says he normally uses 3 to 5 days.
The results discussion also mentions SPY values through 6 and QQQ values through 10, which are broader than the explicit rule grid.
Do not merge those two lists into an undocumented optimization grid.
The strategy is long-only, with no short entry on a negative trend.
The author also uses this rule as a regime filter.
Cash returns, allocation size, moving-average convention, and equality behavior are not stated.

US research can use SPY or QQQ daily history with enough warm-up bars.
An India test on a liquid Nifty 50 or Nifty Bank ETF is a proposed adaptation.
Using the underlying index for signals and an ETF for fills is another adaptation, not the same rule on the ETF.

### Sector MA200 confirmation with reserve allocation

Sources: [Original sector MA200 post](https://alvarezquanttrading.com/blog/sector-trading-using-the-200-day-moving-average/), [Part 2](https://alvarezquanttrading.com/blog/sector-trading-using-the-200-day-moving-average-part-2/), and [Daily, weekly and monthly timeframes](https://alvarezquanttrading.com/blog/etf-sector-trading-the-effect-of-daily-weekly-and-monthly-timeframes/).
The universe is XLY, XLP, XLF, XLE, XLV, XLI, XLB, XLK, and XLU.
The first post states 5 consecutive daily closes above MA200 for entry and 5 below for exit.
The timeframe follow-up explicitly defines the 10-day version and next-open fills.
For that version, buy a 10% allocation after 10 consecutive closes above its own MA200.
Sell that allocation after 10 consecutive closes below MA200 and move it to IEF.
IEF has an initial 10% allocation, plus 10% for each inactive sector.
Thus, two inactive sectors imply 30% in IEF.
This is a long-only sector and bond portfolio.
It is not a rule that shorts weak sectors.

The daily-bar schedule variants permit signals every session, at week end, or at month end.
The weekly-bar variants use MA40 and two confirming weekly bars, evaluated every week or at the first week end of the month.
The monthly-bar version uses MA10 and one confirming monthly bar.
These are separate strategies, not equivalent conversions of MA200.
Part 2 tests entry and exit confirmation counts from 1, 5, 10, 15, and 20 independently.
It also tests 11% sector allocations, but the statement that all active sectors leave 0% in IEF conflicts with nine times 11% being 99%.
Do not silently correct that rounding detail when claiming exact replication.
The author finds only small gains from the extra allocation and asymmetric rules.

A separate Part 2 rule switches the whole portfolio between SPY and IEF after 10 confirming daily closes above or below SPY MA200.
This is an author-tested adaptation of the single-ETF confirmation rule.
The posts do not fully state how weights drift or rebalance between signal events.
An India version needs an actual liquid sector-ETF basket and a separately specified cash or bond reserve.
A local bond reserve is not assumed to have IEF's duration, currency, or crisis behavior.

### Daily 260-day closing-high breakout

Source: [Volume Positive Negative Indicator for Breakouts](https://alvarezquanttrading.com/blog/volume-positive-negative-indicator-for-breakouts/), July 14, 2021.
The baseline test runs from January 1, 2007 through December 31, 2020.
The stock must be a Russell 3000 member, with price greater than USD 5.
The 20-day moving average of close times volume must exceed USD 1 million.
The article says "column" in this liquidity bullet, which I interpret as volume because the surrounding indicator uses volume.
This correction must remain explicit.
SPX must be above its MA200.
The stock close must be at a 260-day high of closes.
More than 15 trading days must have passed since the previous 260-day closing high.
Buy at the next open.
Exit after a closing loss of at least 15%, a closing gain of at least 25%, or a holding period of 60 trading days.
Execute the exit at the next open, not at the percentage threshold.
A gap can therefore produce a larger loss than 15%.
This is long-only.

The author studies equal-dollar individual trades and takes all possible signals.
He does not provide a maximum-position portfolio, a candidate ranking, or capital constraints in this post.
A portfolio implementation must be labelled an adaptation.
An India test needs local price and INR liquidity thresholds, point-in-time members, and treatment of suspensions and price limits.
The US dollar thresholds must not be copied numerically into rupees.

### VPN breakout filters

The same [VPN article](https://alvarezquanttrading.com/blog/volume-positive-negative-indicator-for-breakouts/) provides executable AmiBroker indicator formulas.
For period P, typical price is `(H + L + C) / 3` and its daily change is compared with `0.1 * ATR(P)`.
Volume is positive only when that change exceeds the positive threshold, and negative only when it is below the negative threshold.
The raw indicator is `100 * (sum_positive_volume - sum_negative_volume) / (MA(volume, P) * P)`.
Apply EMA3 to the raw indicator.
The article discusses VPN30 greater than 25 and VPN10 greater than 40 as filters on the baseline trades.
The original tested lengths are 5, 10, 20, 30, 40, and 60.

The simplified version replaces typical price with close and the ATR threshold with zero, retaining the normalization and EMA3.
Its tested lengths are 5, 10, 20, and 30.
The article says VPN_S10 improves the trade study, but its selected cutoff is not stated in readable text.
Do not invent a VPN_S10 cutoff or call it a complete portfolio strategy.
The original VPN filters are implementable daily adaptations, while the selected simplified-filter strategy remains incomplete.

## Fully specified monthly systems from daily data

### Stock rotation by ROC times historical volatility

Source: [Different ranking methods for a monthly S&P500 Stock Rotation Strategy](https://alvarezquanttrading.com/blog/different-ranking-methods-for-a-monthly-sp500-stock-rotation-strategy/), May 12, 2021.
The sample is January 1, 2007 through December 31, 2020.
On the last trading day of the month, select S&P500 members above their own MA200 while SPX is above its MA200.
Rank descending by `ROC(C, L) * HV(L)` and select the highest-ranked 10 eligible stocks.
Test L equal to 63, 126, 189, and 252 daily bars separately.
The formula is `HV(L) = StDev(log(C_t / C_(t-1)), L) * sqrt(252) * 100`.
Sell all holdings at the first trading session open of each month and enter the new selection at that open.
The source explicitly uses next-open entry, which avoids the older posts' same-close signal issue.
The strategy is long-only.

Comparisons use ROC of close, `ROC(EMA(C, L), L)`, ROC divided by HV, and HV alone.
The same L is used for ROC and EMA or HV.
The author reports better sample results for ROC times HV, not for the smoother ROC divided by HV variant.
The statement is not a claim that high volatility always improves trend strategies.
Equal weighting is a sensible test assumption, but it is not explicitly stated here.
Tie handling and the standard-deviation divisor also need declaration.
Point-in-time membership and delisted stock bars are necessary for an unbiased source-style test.
A Nifty 100 or Nifty 500 universe with a local MA200 benchmark is a proposed India adaptation.
Keep the four published L values as fixed research variants rather than claiming an Indian optimum.

### Ten-month ETF momentum and trend

Source: [Trend-following vs. Momentum in ETFs](https://alvarezquanttrading.com/blog/trend-following-vs-momentum-in-etfs/), November 13, 2019.
The sample is January 1, 2006 through September 30, 2019.
The tested ETF list is AGG, DBC, EEM, EFA, GLD, HYG, IWM, LQD, QQQ, SPY, TLT, and VNQ.
These are individual ETF tests, not a published allocation across all 12 assets.

The momentum rule buys when the 10-month return is greater than zero at month end and sells when it is less than zero.
Both fills occur at the next session open.
The trend rule buys when month-end close is above its 10-month moving average and sells when below it.
Both fills also occur at the next session open.
The article separately reports 12-month tests and lists spreadsheet lookbacks from 3 through 15 months.
Neither rule goes short when its condition fails.
The exact-zero and exact-MA cases are not stated.
Do not substitute daily MA200 for monthly MA10 without calling it an adaptation.
The author finds no universal winner between momentum and trend across the ETF list.

### Combined trend and momentum allocation

Source: [Trend-Following Plus Momentum in ETFs](https://alvarezquanttrading.com/blog/trend-following-plus-momentum-in-etfs/), December 11, 2019.
Use the same 12-ETF list as separate tests, with sample end October 31, 2019.
At month end, invest 100% if close exceeds MA10-month and 10-month return is positive.
Invest 50% if exactly one of those conditions is true, and exit if both are false.
Scale any changed allocation at the next session open.
The source also tests 100% allocation when either condition is true.
The spreadsheet ranges are 2 through 12 months and 50%, 75%, or 100% when one condition is true.
The strategy remains long-only and does not use margin.
Allocating 100% to every ETF at once would be a different, leveraged portfolio, not this study.
The author reports that the combined rule did not generally improve on pure momentum under his comparison method.

### Three-factor ETF rotation

Source: [Three Factor ETF Rotation Strategy](https://alvarezquanttrading.com/blog/three-factor-etf-rotation-strategy/), September 14, 2022, corrected January 18, 2023.
Use the corrected post, not pre-correction cached results.
The sample is January 1, 2007 through June 30, 2022.
The sector universe is XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, and XLY.
The asset-class universe is SPY, EFA, IEF, GLD, and ICF.

At month end, rank 3-month return descending, 20-day return descending, and HV20 ascending.
The best value gets rank 1 in each ranking.
Compute `FRV = 0.4 * rank_3m + 0.4 * rank_20d + 0.2 * rank_HV20`.
Select the top 1, 2, or 3 by lowest FRV, then enter only selected ETFs above their own MA200.
Do not filter the whole universe first and refill failed selections from lower ranks.
Sell every position at the monthly rotation, including positions selected again, to rebalance.
All fills use the next open.
The strategy is long-only.

The optimization varies monthly return over 1, 2, and 3 months, daily return and HV over 10, 20, 30, and 40 days, and each return weight over 20%, 30%, 40%, and 50%.
The HV weight is the remainder.
The exact best rows were not recovered from image tables or the gated spreadsheet.
Position weighting, ties, and unfilled-slot cash treatment need declaration.
An India asset-class or sector basket is a proposed adaptation with different inception dates and liquidity.

### Five-ETF three-month rotation

Source: [Five ETF Monthly Rotation Strategy](https://alvarezquanttrading.com/blog/five-etf-monthly-rotation-strategy/), November 23, 2015, with a November 24 clarification.
The universe is SPY, EEM, TLT, GLD, and DBC.
Rank 3-month ROC and hold the top two long positions with 50% allocations and no rebalancing of retained positions.
Alvarez explicitly selects the top two even when their ROC is negative.
Positive-ROC-only selection was the other researcher's interpretation, not Alvarez's base rule.
Both same-close and next-open variants were tested.
Use the next-open variant for a simple executable research baseline.
The test uses USD 100,000 starting capital, zero commission, and 10-share lots.
The sample is January 2007 through October 31, 2015.
The article shows that execution and positive-return assumptions materially change results.
An Indian five-asset substitution would not reproduce these US exposures.

### Middle-ranked sector rotation

Source: [Sector Rotation Strategy: Should Trading Rules Make Sense?](https://alvarezquanttrading.com/blog/sector-rotation-strategy-should-trading-rules-make-sense/), November 15, 2023.
At month end, rank the same nine sector ETFs by 1-month return with the best at rank 1.
Buy ranks 4, 5, and 6 equally at the next open and sell the previous month's holdings at the rotation.
This is a complete long-only middle-ranked relative-strength strategy, not selection of the strongest three.
The sample is January 1, 2010 through September 30, 2023.
The author also compares ranks 1 to 3 and 2 to 4, plus return lengths of 1, 2, 3, 4, and 6 months and 2, 3, or 4 positions.
I rank it below simple trend and momentum rules because the author questions the rank-window rationale and sensitivity.

## Author adaptations and incomplete strategy candidates

### Technical Minervini adaptation

Source: [External Strategy Rule Evaluation. Too many rules?](https://alvarezquanttrading.com/blog/external-strategy-rule-evaluation-too-many-rules/), December 6, 2017.
The nine technical conditions are close above MA150, close above MA200, MA150 above MA200, MA200 above its value one month ago, MA50 above MA150, MA50 above MA200, close above MA50, price at least 30% above the 52-week low, and price within 25% of the 52-week high.
Alvarez adds Russell 3000 membership and SPX above MA200.
Buy at the next open, with a maximum of 20 positions and 5% per position.
When candidates exceed free slots, rank by 1-year return descending.
Exit next open after close below MA200 or a closing loss of 10% or more.
This is a daily, long-only, mechanically testable adaptation.
It is not the full Minervini strategy, because fundamentals, chart discretion, and original position sizing are absent.
The author tests 256 technical-rule combinations while retaining stock-above-MA200 as mandatory.
He does not conclude that the full nine-rule version is a good finished trend strategy.
Exact month, year, and high/low conventions need declaration.
An India implementation must remain labelled as a local-universe adaptation.

### DTAYS weekly breakout

Sources: [DTAYS Weekly Breakout Strategy](https://alvarezquanttrading.com/blog/dtays-weekly-breakout-strategy/) and [Time stops follow-up](https://alvarezquanttrading.com/blog/dtays-weekly-breakout-strategy-with-time-stops/).
I inspected the original HTML emphasis to recover the marked DTAYS values rather than guessing from the parameter lists.
The marked original settings are 20 positions, a 20-week closing-high window, return greater than 30% over that window, an 8% maximum loss, and ATR100 times 3.
The tested grids are 10, 20, or 30 weeks; returns greater than 20%, 30%, or 40%; 10 or 20 positions; 8% or 12% loss; and ATR100 times 3 or 5.

At week end, require a new NWEEK high of weekly closes and the specified NWEEK return.
Require 21-day average close times volume above USD 15 million and SPY above daily MA100.
Rank excess candidates by NWEEK return descending.
Evaluate the maximum-loss stop and the ATR100 trailing stop from the highest close since entry at daily closes.
The original post adds an optional HV100 below 40 filter and describes better sample results with it.
Entry and exit fill timing are not stated in the article body.
Next-open fills are therefore a proposed execution adaptation, not an exact recovered rule.
ATR fixing versus updating and position sizing also need specification.

Alvarez could not obtain historical IBD50 data and used a different liquid-stock universe.
The time-stop follow-up explicitly changes the universe to Russell 3000.
It adds a week-end losing-trade check after 2, 4, 6, or 8 weeks, with no time stop as a comparison.
The source does not make clear whether that check repeats after the selected week or occurs only once.
The follow-up finds no clear CAR or MDD improvement from the added stop.
This is weekly entry research with daily stop checks, not a daily breakout entry system.

### MA100 slope and volatility filter

Source: [Avoiding Volatile Trades](https://alvarezquanttrading.com/blog/avoiding-volatile-trades/), August 3, 2022.
For Nasdaq100 stocks, require MA100 to rise versus the prior day's MA100 on each of the last five sessions.
Enter at next open and select at most 10 stocks ranked by HV20 descending.
The baseline sell condition is five consecutive daily declines in MA100.
The filtered variant also requires HV20 below its two-year median plus one standard deviation for entry.
It lists HV20 above that median plus two standard deviations as a sell condition.
The text does not explicitly state the exit Boolean operator or sell fill timing.
Those gaps prevent exact replication without assumptions.
The author changed the external article's MA70 and one-day confirmation to MA100 and five days.
He reports lower drawdown but a substantial return reduction for the initial volatility filter.
The best optimized entries tended to favor HV below the median, but the exact best pair is not recovered.
An India version needs prior local history for the HV distribution, not a threshold inferred from a full future sample.

### Low-volatility uptrend portfolio

Sources: [Initial volatility comparison](https://alvarezquanttrading.com/blog/should-one-trade-high-or-low-volatility-stocks/), [Stops](https://alvarezquanttrading.com/blog/stops-and-trading-high-vs-low-volatility-stocks/), [Profit targets](https://alvarezquanttrading.com/blog/low-volatility-stocks-and-profit-targets/), and [Portfolio](https://alvarezquanttrading.com/blog/low-volatility-stocks-20-cagr-portfolio/).
The portfolio post tests Russell 3000 membership as it existed from January 1, 2004 through December 31, 2013.
Require close above USD 5, six-month return above 10%, stock and SPX above MA200, and HV100 between 20 and 35 for five sessions.
The liquidity rule says the 21-day average dollar volume exceeds USD 10 million "over the last 3 months".
Its exact rolling interpretation is not explicit.
Hold at most 10 stocks.

The four portfolio exit pairs are a six-month time stop with 20% target, a six-month time stop with ATR10 times 4 target, a 20% trailing stop with 20% target, and a six-month time stop without loss stop or target.
The ranking study includes 18 rankings and their inverses.
One promising ranking is `100 - abs(MA50 / MA200 - 1)`, favoring stocks with those moving averages close together.
The no-stop six-month version produces the strongest discussion in the article, but one trade loses about 70%.
Do not call that version safe because the entry volatility is low.
Entry fills, portfolio sizing, ATR fixing, trailing reference, and calendar-month counting are not fully specified.
This is a useful daily-input research family, not an exact ready-to-run portfolio.

### Sector and country relative momentum families

Sources: [ETF Sector Rotation](https://alvarezquanttrading.com/blog/etf-sector-rotation/), [Sector reader ideas](https://alvarezquanttrading.com/blog/etf-sector-rotation-ideas-from-readers/), [Country ETF Rotation](https://alvarezquanttrading.com/blog/country-etf-rotation/), and [Country reader suggestions](https://alvarezquanttrading.com/blog/country-etf-rotation-readers-suggestions/).
The basic sector rule ranks 6- or 12-month return at month end and buys the top 1 or 2 at next open.
The trend variant replaces a selected ETF below its 6- or 12-month MA with SHY or TLT.
The dual-rank sector variant adds two return ranks selected from 3, 6, 9, and 12 months, uses Rank1 to break ties, selects two, and substitutes TLT for a failed trend gate.
The exact successful pair and full portfolio weight and exit conventions are not recovered from text.

The sector reader variant averages 3-, 6-, 9-, and 12-month returns and selects the top three only if their average exceeds SHY, IEF, or TLT's corresponding average.
It trades at the beginning-of-month open.
A skip-recent-month variant is discussed, but its formula conflicts with its date example.
Do not silently turn that text into standard 12-1 momentum.

The country universe contains EIS, EPI, EWA, EWC, EWD, EWG, EWH, EWI, EWK, EWL, EWM, EWN, EWO, EWP, EWQ, EWT, EWU, EWY, EWZ, FXI, RSX, THD, TUR, EZA, EWW, EWS, EWJ, and ECH.
EPI is US-listed Indian exposure, not a strategy executed in the Indian market.
The initial country liquidity gate is 21-day average close times volume above USD 5 million.
The country tests use analogous single and dual ranks, SHY or IEF reserve, and 6- or 12-month trend gates.
A 21-day correlation cap against SPX of 0.75, and later 0.50, does not improve the author's sample.
RSX and other discontinued or unavailable instruments require proper historical and terminal-value treatment.

The country reader tests include the sum of 3-, 6-, 9-, and 12-month ranks, reverse momentum, and hybrid reversal plus momentum.
The reversal inputs include percent off a 5-, 10-, 15-, or 20-day high, RSI over those lengths, or ascending 1-, 2-, or 3-month return.
The long-momentum input uses 6-, 9-, or 12-month return in the recent-return hybrid.
Hold 2, 5, or 8 ETFs, with a 6- or 12-month trend gate and IEF reserve.
These are long-only hybrid selection rules, not short strategies.
The source warns that reserve profits can dominate the reverse-momentum results.
Exact best rows, some rank directions, and full portfolio conventions remain incomplete.

### Older stock rotation studies

[Monthly S&P500 Stock Rotation Strategy](https://alvarezquanttrading.com/blog/monthly-sp500-stock-rotation-strategy/) buys the top 20 by 6- or 12-month return at the first trading day close, with optional SPX above MA200, and holds until next month.
Its sample is 2001 through October 2013, commission is USD 0.01 per share, and idle cash uses the three-month T-bill rate.
The lowest-return version buys weak stocks long and is not a short momentum strategy.
Same-day closing signals and closing fills need an explicit operational convention.
A next-open conversion changes the source strategy.

[S&P500 Monthly Rotation-Readers' Ideas](https://alvarezquanttrading.com/blog/sp500-monthly-rotation-readers-ideas/) uses current 9-month return, the top 10, and first-session close entry.
It tests a three-month high in the prior 10 sessions and SPX MA200 as helpful filters.
Its intraday percentage stops have no numerical cutoff recovered from readable text.
Do not add an invented stop value.

[Intermediate Term Stock Rotation Strategy](https://alvarezquanttrading.com/blog/intermediate-term-stock-rotation-strategy-using-sp500-stocks/) ranks six-month return measured six months earlier, then buys the top 10 at the first trading day close of each quarter or half-year.
Its January 2, 2014 example uses January 1, 2013 to July 1, 2013 for the return window.
It optionally gates with SPX MA200.
Do not relabel this example as exact academic 12-to-7-month momentum.
Rotation exits and sizing need declaration.

[StockCharts Technical Rank Rotation](https://alvarezquanttrading.com/blog/stockcharts-technical-rank-sctr-rotation-strategy/) sells all at quarter-end close and buys 20 equal-weight stocks at that close.
It compares high and low SCTR with high 3-, 6-, 9-, and 12-month return, plus high and low HV100.
Optional gates are stock or SPX above MA200.
The source finds simpler 6- and 9-month return rankings competitive with SCTR.
The return-based variants avoid historical SCTR data, but same-close timing remains a replication issue.
SCTR formulas and historical cross-sectional values were not independently gathered.

### Price and high-distance rankings

[The Simplest Momentum Indicator](https://alvarezquanttrading.com/blog/the-simplest-momentum-indicator/) selects the 10 highest-priced Nasdaq100 stocks at month end, requiring stock and SPX above MA200, then buys and sells at next open.
The ranking must use historical as-traded price before split and dividend adjustment.
Retrospectively adjusted close is not a valid substitute for that ranking.
The author would not trade the strategy as presented because results were insufficient and splits can force an exit.
Sizing and the adjustment basis for the other indicators remain gaps.

[Using 52-week highs](https://alvarezquanttrading.com/blog/using-52-week-highs-in-a-sp500-monthly-rotation-strategy/) buys the 25 S&P500 stocks nearest to their 52-week high of closes at monthly next-open rotation.
It compares the furthest-from-high selection and buckets of 25.
The text does not explicitly specify percentage versus absolute distance, sizing, or the exact daily conversion of 52 weeks.
The author does not find a compelling ranking edge under this monthly context.
Do not present `C / rolling_max(C, 252)` as a quoted source formula.

### Other related families

[SPY, SSO and TLT Strategy](https://alvarezquanttrading.com/blog/spy-sso-and-tlt-strategy/) makes decisions at the last monthly close.
If SPY is not above MA200, the initial rule buys TLT.
Otherwise it buys SSO when VIX is below 25 and SPY when VIX is not below 25.
An added bond filter buys TLT only when TLT is above its own MA200, else holds cash.
A no-SSO variant holds SPY above its MA200, else TLT above its MA200, else cash.
The exact fill timing is not separately stated, so this remains partially specified.
The source shows why bonds cannot be assumed to protect every equity bear market.
India VIX is not numerically interchangeable with VIX, and futures are not equivalent to SSO.

[Using Historical Volatility for Parameter Adjustment](https://alvarezquanttrading.com/blog/using-historical-volatility-for-parameter-adjustment/) uses HV21 below 17% to select 12-month momentum, otherwise 1-month momentum, at monthly close.
Positive selected return buys on close and negative selected return exits on close.
The author notes that the 80th-percentile threshold changes to 21.2% for the post-1999 sample and tests expanding and rolling thresholds using available historical data.
The dynamic variants lose much of the initial drawdown benefit.
He declines to continue using the idea in his dual-momentum work.
A tradable ETF, next-open execution, or an Indian threshold is a proposed adaptation.

[Heikin-Ashi Charts](https://alvarezquanttrading.com/blog/heikin-ashi-charts/) buys after a green monthly synthetic candle and exits after a red one at the next monthly bar open.
The article points elsewhere for the candle formula and does not fully state its recursive seed or execution-price interpretation.
A valid adaptation must use real prices for fills, not synthetic Heikin-Ashi opens.
Daily Heikin-Ashi paired with MACD or stochastic is a failed or incomplete mention in this post, not a recoverable successful daily strategy.

[Rotation-day sensitivity](https://alvarezquanttrading.com/blog/day-of-month-pattern-or-luck-for-a-monthly-etf-rotation-strategy/) combines 3- and 6-month ranks, holds two, breaks ties with low HV21, and trades at next open around month boundaries.
The written formula defines RankC using RankA plus RankC, which is self-referential.
RankA plus RankB is a plausible correction, not an exact quotation of the source rule.
The exact tested offset grid and sizing were not recovered.

[Multiple Time Frames for Scoring ETF Rotational Strategies](https://alvarezquanttrading.com/blog/multiple-time-frames-for-scoring-etf-rotational-strategies/) holds three 33.3% allocations using weighted ranks of 1-, 3-, and 6-month return at monthly next-open rotation.
Negative weights favor high returns under the article's scoring convention.
The examples `[-20, -40, 40]` and `[0, -100, 0]` illustrate scoring, not a selected final strategy.
The exact universe is in an unread image, and the optional weekly-close percentile filter needs precise interpretation.

[Mutual Fund Sector Rotation](https://alvarezquanttrading.com/blog/mutual-fund-sector-rotation-ideas-from-readers/) applies average 3/6/9/12-month momentum versus a reserve to Fidelity sector mutual funds, with beginning-of-month close entry.
Mutual-fund NAV timing is outside the direct equity and ETF scope.
An ETF or Indian mutual-fund version would need separate execution rules.

[Backtesting a Dividend Strategy](https://alvarezquanttrading.com/blog/backtesting-a-dividend-strategy/) and [S&P500 Dividend Aristocrats](https://alvarezquanttrading.com/blog/sp-500-dividend-aristocrats/) add momentum and MA200 to dividend-stock selection.
The earlier proxy requires at least 16 years of rising dividends and ranks by `1000 * rising_dividend_years + six_month_return` for 10 holdings.
The later post uses historical Aristocrat membership, monthly entries, optional stock or SPX MA200 gates, removal exits, and annual resets.
These are dividend-universe adaptations, not pure price momentum.
An India dividend-history analogue and point-in-time announcement data were not verified.

## Mention-only and inaccessible sources

The following are primary-source titles and public excerpts, not complete strategies recovered in this review.
Their JSON entries have empty parameter objects and no inferred trading rules.

| Source | Verified information | Missing information |
| --- | --- | --- |
| [The 50/50 SPY Strategy](https://alvarezquanttrading.com/blog/the-50-50-spy-strategy/) | Balances trend protection against missed rallies | Actual allocation, signals, and timing |
| [SPY/TLT rotation](https://alvarezquanttrading.com/blog/spy-tlt-rotation/) | Mentions N-month selection and allocation variants | N, full selection rules, exits, and timing |
| [ETF Bond Rotation](https://alvarezquanttrading.com/blog/etf-bond-rotation/) | Applies preceding rotation ideas to bond ETFs | Basket and complete rules |
| [Stiffness Indicator Analysis](https://alvarezquanttrading.com/blog/stiffness-indicator-analysis/) | References a trend indicator and November 2018 magazine article | Indicator formula and trading system |
| [UPRO/TQQQ Leveraged ETF Strategy](https://alvarezquanttrading.com/blog/upro-tqqq-leveraged-etf-strategy/) | Moves among leveraged ETFs, unleveraged ETFs, and TLT | Signal values, allocations, exits, and timing |
| [Mean Reversion vs Trend Following Through the Years](https://alvarezquanttrading.com/blog/mean-reversion-vs-trend-following-through-the-years/) | Discusses persistence of market behavior and edges | Any exact trend strategy |
| [SPX and Gold Momentum Portfolio](https://alvarezquanttrading.com/blog/spx-and-gold-momentum-portfolio/) | Discusses stocks and gold in an inflation context | Weights, momentum period, and timing |
| [More ideas for stock rotation rankings](https://alvarezquanttrading.com/blog/more-ideas-for-ranking-methods-on-a-monthly-sp500-stock-rotation-strategy/) | Reader follow-up to the fully read ROC/HV article | Additional ranking formulas and settings |

Do not infer a 50% permanent SPY allocation from the 50/50 title.
Do not copy generic dual-momentum rules into the inaccessible SPY/TLT article.
Do not call these posts paid-only, because the observed problem was HTTP access failure, not a verified paywall.

## US and India adaptation requirements

### Price and universe data

Use split and dividend treatment consistently for signals and returns.
Retain actual executable opens separately from adjusted signal prices.
[The author's dividend-adjustment study](https://alvarezquanttrading.com/blog/to-dividend-adjust-or-not-to-dividend-adjust-that-is-the-question/) shows that adjustment can change both trades and results, especially for S&P500 strategies.
As-traded-price ranking is the explicit exception and needs historical raw prices.
Do not apply adjusted price times raw volume without checking the vendor's conventions.

US equity tests need historical S&P500, Nasdaq100, or Russell 3000 membership, including removed and delisted stocks.
Current index lists introduce survivor and pre-inclusion bias.
India equity tests likewise need point-in-time local membership, corporate actions, symbol history, and delisted or suspended names.
No local data availability was verified in this website review.
The requirement is a blocked data need until the main agent confirms it.

ETF tests must respect inception, closures, reorganizations, and actual available sessions.
Do not backfill new Indian ETFs with an index and call it tradable ETF history.
If an index proxy is needed, label the pre-inception segment and its assumed tracking, fees, and fills.
Use the exchange calendar to form week-end and month-end bars.
US holidays must not define Indian signal dates.

### Execution and costs

Use information from a completed bar for next-open rules.
A closing stop is not an intraday stop and does not guarantee a fill at its threshold.
Same-close research rules need a stated pre-close decision method or a labelled next-open adaptation.
Declare costs, spread, slippage, turnover, round lots, and sizing before comparing variants.
An India study must include applicable local transaction costs and model suspensions or circuit-limit days that prevent a fill.
Cash interest is separate from the risk-free hurdle used to calculate risk-adjusted metrics.

### Exposure and portfolio semantics

IEF, TLT, SHY, cash, and an Indian bond or cash instrument are not interchangeable return series.
A reserve also needs inception and liquidity checks.
Daily-reset leveraged ETFs carry path dependence, so futures or margin borrowing are different adaptations.
None of the verified stock or ETF strategies here supplies a systematic short-selling leg.
Low-rank, weak-stock, inverse-ranking, and negative-trend comparisons generally remain long-only selections or cash exits.
For inaccessible posts, direction stays unverified.

Stock studies that take all equal-dollar signals cannot establish a capital-constrained portfolio result.
Declare slot count, position weight, ranking, cash allocation, and rebalancing before portfolio tests.
Record India substitutions separately from unchanged US reproductions in any future result file.
Do not treat the inventory as evidence that these strategies already work in the local screener engine.

## Handoff

The best first exact US daily test is MA200 confirmation with N equal to 3, 4, and 5 and next-open fills.
The best daily stock signal test is the published 260-day breakout baseline, followed by the stated original VPN cutoffs.
The best monthly stock-ranking test is ROC times HV with the four published daily lookbacks and MA200 stock and market gates.
The best monthly ETF portfolio test is the corrected three-factor rule on its two published universes, with sizing and tie assumptions recorded.
For India, test those same ideas only as declared local-universe adaptations after point-in-time data and executable ETF history are confirmed.
The 8 inaccessible sources and the unread image or spreadsheet fields remain open research gaps.
