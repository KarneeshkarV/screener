# One filter that improves the default screen, and 54 that do not

Research, not financial advice.

## What was tested

The default `screener screen` criterion is `ema`: EMA5 > EMA20 > EMA100 > EMA200.
Call that BASE.
The question was whether adding a second filter on top of BASE improves it, and 55 causal candidate features were tested one at a time to find out.

A filter was applied as a cross-sectional rank cut: on each date, rank the names passing BASE by one feature and keep the top `q`.
That puts every feature on one scale and holds the surviving count fixed, so a difference in outcome is about *which* names were kept rather than how many.
`q` was varied over 0.7 / 0.5 / 0.3 alongside each feature's own parameters, because a filter that only works at one setting is not a filter.

- Universe `nifty_midsmall400_pit`, point-in-time membership, 820 names ever a member.
- 2021-08-18 to 2026-08-17, 1240 trading days.
- 4 expanding walk-forward folds, first 250 days held back; every figure below is pooled out-of-sample.
- Equal weight, refreshed every 21 bars, 20 bps one-way cost on turnover.
- 471 filter arms plus base.
- Repeated on a 250-name subsample as a sample-stability check.

Every feature was verified causal by recomputation on truncated history: the value at bar `t` is unchanged when the series stops at `t`.

## BASE, out of sample

| metric | value |
|---|---|
| Sharpe | 1.500 |
| CAGR | 27.9% |
| max drawdown | -25.2% |
| Calmar | 1.105 |
| names per day | 226 |

Per fold, BASE Sharpe is 2.49, 2.46, **-0.22**, 1.37.
That spread is wider than any filter effect measured, and it is the main reason the conclusions below lean on parameter and sample breadth rather than on fold counts.

## Result

**Downside volatility was the only survivor.**
It is positive on both samples with majority parameter breadth on both: +0.103 median Sharpe on 820 names (6 of 9 settings beat BASE), +0.120 on 250 names (8 of 9).
Full realized volatility is the same finding slightly weaker, and correlates 0.89 with it.

At matched candidate count the filter moves Sharpe from 1.500 to 1.695.

**The gain is in Sharpe, not in Calmar.**
Over the same comparison Calmar goes from 1.105 to 1.088 and CAGR falls about three points.
The filter suppresses day-to-day volatility much more than it suppresses drawdown.
A book judged on drawdown gains nothing from it and should use the bare stack.

## What did not work, and why it is unsurprising

Trend-quality features added nothing as a class: efficiency ratio -0.008, variance ratio -0.086, Hurst -0.078, return autocorrelation -0.114, directional share -0.134, trend persistence -0.084.
Zero of seven cleared the bar.
All liquidity and all relative-strength features failed too.

BASE is already a trend filter, so further trend, momentum or trend-quality information is largely redundant with the screen itself.
Volatility is the one dimension BASE says nothing about, which is where the whole effect turned out to be.

Causal spectral and wavelet features were the worst of the entire set - low/high-frequency wavelet energy at -0.257, beating BASE at zero of nine settings - so they do not identify clean trending regimes better than efficiency ratio or R-squared.

## Two features that looked good and were rejected

Volatility percentile against a name's own history beat BASE at **9 of 9** parameter settings on the 820-name panel, at +0.103.
On the 250-name subsample it is **-0.057 at 3 of 8**.
The short-vol / long-vol ratio behaves the same way: 9 of 12 becomes 1 of 12.
Both flip sign.

Parameter stability did not catch either of them.
Both were stable across their whole grid and still failed a change of sample, so parameter breadth and sample breadth are separate tests and a candidate has to pass both.

## Filters do not stack

At matched candidate count, tightening the volatility filter beat every two-factor combination tried.

| combination | names | Sharpe | CAGR | maxDD | Calmar |
|---|---|---|---|---|---|
| downside volatility, tighter cut | 110 | **1.695** | 0.250 | -0.230 | 1.088 |
| + distance from high | 110 | 1.605 | 0.258 | -0.251 | 1.030 |
| + turnover consistency | 131 | 1.553 | 0.265 | -0.248 | 1.069 |
| + trend deterioration | 142 | 1.543 | 0.252 | -0.239 | 1.056 |
| downside volatility alone | 158 | 1.532 | 0.248 | -0.243 | 1.018 |
| BASE | 226 | 1.500 | 0.279 | -0.252 | 1.105 |
| + volatility-adjusted momentum | 110 | 1.475 | 0.245 | -0.265 | 0.925 |

This is why the shipped result is two terms - the trend screen and one volatility filter - rather than a stack of complementary factors.

## Limits

One universe, over a period that was mostly a bull market; BASE earning 27.9% CAGR says as much about the window as about the screen.
Forward-return statistics pooled overlapping observations, so their effective sample is far smaller than the raw counts and they were not treated as load-bearing.
Results have not been reproduced on US equities.
