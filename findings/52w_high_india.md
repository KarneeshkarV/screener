# 52-week high factor on Indian stocks

Source idea: [Cofacto, "We tested 'buy the dip' on Indian stocks since 2015"](https://cofacto.substack.com/p/we-tested-buy-the-dip-on-indian-stocks).
The claim there is that stocks nearest their 52-week high earn far more than stocks furthest below it, and that the "avoid the dip" half of the signal is the statistically strong half.

## What was added to the codebase

The factor now exists once and is reused by every path:

- Recipe `dist_from_52w_high` in `screener/factors/recipes.py`: percent below the trailing 252-session closing high, `<= 0`, gated by `tradeable_span` so a stub quote cannot set a fake high.
- Strategy `dist_from_52w_high` in `screener/strategies/plugins/dist_from_52w_high.py`: writes `rank_score`, entry is `dist_from_52w_high > -100`, which only excludes the NaN of an untradeable window.
- Screen ranking registration `dist_from_52w_high` via `register_bar_scorer`, so the screen's ranking layer scores the same number as the backtest.
  Like `ha_momentum`, it is not a `-c` criterion, because no TradingView filter set fronts this rule and the screen must not prefilter on a proxy that is not the rule.
- Tests in `tests/test_strategy_dist_from_52w_high.py` and the gate sweep in `tests/test_tradeable_gate_coverage.py`.

This is distinct from the pre-existing `near_52_high` criterion, which is a snapshot breakout composite over TradingView columns.

Reproduce the tearsheet:

```
screener factor-tearsheet -m india --strategy dist_from_52w_high \
  --universe-file data/universes/india_broad_2015_2026.txt \
  --start 2015-02-01 --end 2026-09-30 --quantiles 10 --horizons 1,21
```

## Data used

- 518 NSE names with a continuous 2015-01 to 2026-09 history in the local yfinance cache.
  The list is committed as `data/universes/india_broad_2015_2026.txt` (520 requested, 518 with enough history to score).
- Split- and dividend-adjusted daily closes, `auto_adjust=True`.
- Monthly rebalance on the last observed trading day of each month, equal weight inside each decile.
- 138 months from 2015-02 to 2026-09.

## Decile depth matches the source

Median distance from the 52-week high at rebalance:

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|
| -49.8% | -35.9% | -29.2% | -24.2% | -20.0% | -16.2% | -12.6% | -8.8% | -4.9% | -0.5% |

The source says its bottom tenth starts near 50% off and the median stock sits 25% under its high.
Both match, so the universe is behaving like the source's, not like a large-cap-only sample.

## Monthly rebalance results

Equal-weight decile portfolios, gross of costs:

| Port | Mean / month | SD / month | CAGR |
|---|---|---|---|
| D1 (furthest) | 1.77% | 11.46% | 14.7% |
| D2 | 1.55% | 9.03% | 14.6% |
| D3 | 1.43% | 7.95% | 14.2% |
| D4 | 1.65% | 7.27% | 17.9% |
| D5 | 1.40% | 6.82% | 14.9% |
| D6 | 1.64% | 6.43% | 18.5% |
| D7 | 1.74% | 6.10% | 20.2% |
| D8 | 1.92% | 5.68% | 23.3% |
| D9 | 2.01% | 5.47% | 24.6% |
| D10 (nearest) | 1.75% | 5.40% | 21.1% |
| Equal-weight universe | 1.68% | 6.79% | 18.9% |

Spread tests, monthly series, Newey-West lag 1:

| Spread | Mean / month | t |
|---|---|---|
| D10 - D1 | -0.01% | -0.02 |
| D10 - equal-weight universe | +0.07% | 0.27 |
| D1 - equal-weight universe | +0.08% | 0.17 |
| D10 - D1, least liquid tercile | +0.20% | 0.23 |
| D10 - D1, middle tercile | +0.66% | 1.06 |
| D10 - D1, most liquid tercile | +0.86% | 1.16 |

NIFTY 50 price index over the same span: 8.44%/yr.
The source reports 10.0%/yr on the total-return index, so the price data lines up.

## Sharpe and drawdown

Sharpe is annualized from the 138 monthly returns as `mean / sd * sqrt(12)`, with a zero risk-free hurdle, which is the repo default.
Net means the 20 basis point per side cost leg.
Drawdown is from the compounded monthly curve.

| Port | CAGR gross | CAGR net | Sharpe gross | Sharpe net | Max drawdown |
|---|---|---|---|---|---|
| D1 (furthest) | 14.7% | 13.5% | 0.53 | 0.51 | -77.5% |
| D2 | 14.6% | 11.8% | 0.59 | 0.51 | -69.5% |
| D3 | 14.2% | 10.6% | 0.62 | 0.51 | -58.3% |
| D4 | 17.9% | 13.9% | 0.78 | 0.65 | -53.1% |
| D5 | 14.9% | 10.9% | 0.71 | 0.56 | -51.7% |
| D6 | 18.5% | 14.3% | 0.88 | 0.72 | -46.6% |
| D7 | 20.2% | 15.8% | 0.99 | 0.81 | -46.3% |
| D8 | 23.3% | 18.9% | 1.17 | 0.98 | -32.3% |
| D9 | 24.6% | 20.5% | 1.27 | 1.09 | -30.7% |
| D10 (nearest) | 21.1% | 17.5% | 1.12 | 0.96 | -33.7% |
| Equal-weight universe | 18.9% | 18.9% | 0.86 | 0.86 | -51.1% |

This is where the source reproduces best.
Sharpe climbs almost step by step from D1 (0.53) to D9 (1.27), and the dip decile has the worst drawdown (-77.5%) against the near-high deciles (-30.7% to -33.7%).
The source reports a 36% worst fall for its top group against 51% for its fair benchmark; here D10 is -33.7% against -51.1% for the equal-weight universe.
The very top decile is slightly worse than D9 on both measures.

Net of costs, D10 earns less than the equal-weight universe (17.5% against 18.9%) but carries a better Sharpe (0.96 against 0.86).
So the near-high end wins on risk-adjusted terms and loses on absolute terms at this turnover.

## What this does and does not reproduce

The ordering of compounded returns broadly matches: the near-high deciles (D7-D10) have the best CAGR, and volatility falls monotonically from D1 to D10.
But the CAGR gap is volatility drag, not a return edge.
Mean monthly return is flat across deciles, while monthly standard deviation halves from the dip decile to the near-high decile.
The same lower volatility is what lifts the Sharpe ratio.
So on this data the factor selects calmer names, not higher-returning ones.

The source's central result is that the bottom decile keeps falling at -9.1%/yr.
That does not appear here, and the reason is the sample, not the factor.
The local yfinance and FMP caches contain almost no dead Indian listings: of 532 names that existed in mid-2015, only 2 stop trading before 2026, and FMP India data starts in 2016-08 with DHFL showing 7 bars.
The source attributes the bottom decile's loss to the 14% of that decile that no longer trades.
Without those names in the panel, the survivorship-biased bottom decile is an ordinary small-cap basket.

Turnover is 63% per month in D10, matching the source's 65%.
At 20 basis points per side that costs about 2.5%/yr, which takes D10 from 21.1% gross to 17.5% net, below the 18.9% equal-weight universe.

## Where the daily tearsheet disagrees

`factor-tearsheet` on the same panel reports a monotone 21-bar spread: Q1 1.386%, Q10 2.070%, Q10-Q1 +0.683%, IC 0.054 with HAC t = 4.36.

That is not a monthly-rebalance edge.
Recomputed on month-end snapshots only, the same pooled 21-bar returns give Q1 2.152%, Q10 1.738%, so Q10-Q1 is -0.414%.
The non-overlapping calendar-month portfolios above agree with the month-end version.
The daily figure pools 2,881 overlapping windows and is sensitive to which day the window starts on.
Treat it as an artifact of overlapping daily windows until a non-overlapping implementation confirms it.

## Next step to settle the source's claim

The missing ingredient is delisted-name price history.
A run over NSE bhavcopy archives from 2015 onward, including names that later stopped trading, is the only way to test whether the bottom decile really collapses.
Until then the honest reading is: the risk-adjusted staircase and the drawdown gap reproduce, the mean-return spread does not, and costs eat the near-high edge in absolute return at this turnover.

## Limitations

- Static 518-name list, not dated index membership. All names survived to 2026, which is the known bias.
- Close-to-close returns, no next-open fills or slippage model beyond the cost leg.
- Equal-weight deciles, so the result is the small/mid-cap factor, not the large-cap one.
- The 20 basis point per side cost is an assumption, not measured.
- Sharpe uses monthly returns and a zero risk-free hurdle; it is not comparable to a daily-equity-curve Sharpe.
- Factor-tearsheet's `Top Q turnover` is daily turnover and is not comparable to the monthly 63% figure.
