# Trend filters on top of the default screen

Research, not financial advice.
End date 2026-08-17.
Code: worktree `/home/karneeshkar/filters-wt`, branch `research/trend-filters`.
Artefacts: `/home/karneeshkar/grill-me-filters`.

## Question

The default `screener screen` criterion is `ema`: EMA5 > EMA20 > EMA100 > EMA200.
Does adding a second filter to that screen improve it?

## Method

BASE is the `ema` stack written as a backtestable expression and evaluated on bars.

A filter is a cross-sectional rank cut: on each date, rank the names that pass BASE by one feature and keep the top `q`.
Every filter is therefore on one scale, and the surviving count is controlled by construction, so a difference in outcome is about *which* names were picked rather than how many.
`q` is part of the stability grid at 0.7 / 0.5 / 0.3, not a tuned constant.

- Universe `nifty_midsmall400_pit`, point-in-time membership, 820 names ever a member over the window.
- 2021-08-18 to 2026-08-17, 1240 trading days.
- 4 expanding walk-forward folds, first 250 days held back as the training window; every headline number is pooled out-of-sample.
- Equal weight, refreshed every 21 bars, 20 bps one-way cost charged on turnover.
- 55 features x up to 4 parameter settings each x 3 keep-fractions = 471 filter arms plus base.

A second panel repeats the study on a 250-name subsample and adds the experimental categories.
That subsample is the sample-stability test, and it is what most of the conclusions below turn on.

### Causality

Every feature is verified truncation-invariant: recomputed on history truncated at bar `t`, the value at `t` is unchanged.
176 tests, all 55 features, all 197 parameter settings.
See `tests/test_feature_causality.py`.
The textbook form of every category 7 and 8 method is non-causal and was reformulated: Kalman is the forward pass only, Savitzky-Golay is fitted on a trailing window and read at its right edge, the wavelet cascade uses trailing means, the FFT is windowed and nothing is reconstructed, and the L1 trend filter is solved per trailing window.

## BASE, out of sample

| metric | value |
|---|---|
| Sharpe | 1.500 |
| CAGR | 27.9% |
| max drawdown | -25.2% |
| Calmar | 1.105 |
| names per day | 226 |
| turnover | 6.1x / yr |

Per fold, BASE Sharpe is 2.49, 2.46, **-0.22**, 1.37.
That spread is larger than any filter effect measured below, and it is why fold-count evidence is treated as weak throughout.

## Final ranking

### Strong - keep in production

**`downside_vol`** (annualized semi-deviation of negative daily returns, 60d).
The only feature positive in both samples with majority parameter breadth in both: +0.103 median Sharpe on 820 names (6/9 settings), +0.120 on 250 names (8/9 settings).
`realized_vol` is the same finding at slightly lower magnitude (+0.062 / +0.058, 11/12 and 9/12) and correlates 0.89 with it.
Keep one, not both.

One qualification that matters for production.
The gain is in Sharpe, not in Calmar: at matched survivor count the filter takes Sharpe from 1.500 to 1.695 but Calmar from 1.105 to 1.088.
It suppresses day-to-day volatility more than it suppresses drawdown, and it costs 3 points of CAGR.
If the book is judged on drawdown rather than volatility, this filter earns nothing.

### Useful only in certain regimes

`channel_position`, `ema_distance_vol`, `distance_from_high`, `trend_deterioration`, `momentum_consistency`.
All positive in both samples but small, and their sign is carried by one market state.
`channel_position` and `ema_distance_vol` are sideways-market effects (+0.31 and +0.45 Sharpe in sideways, ~0 elsewhere).
`distance_from_high` is a bear-market effect (+0.18).

### Redundant with another feature

Spearman >= 0.9 against a better-scoring feature, or near it:

| feature | duplicate of | rho |
|---|---|---|
| `realized_vol`, `atr_pct` | `downside_vol` | 0.89, 0.75 |
| `variance_ratio` | `hurst_vr` | - |
| `logprice_slope` | `ema_slope` | - |
| `trend_r2` | `logprice_slope_t` | - |
| `adv_value` | `amihud_illiquidity` | - |
| `kalman_slope_snr` | `kalman_slope` | - |
| `wavelet_lf_ratio` | `wavelet_hf_energy` | - |

### No measurable improvement

The entire trend-quality family: `efficiency_ratio` (-0.008), `variance_ratio` (-0.086), `hurst_vr` (-0.078), `return_autocorr` (-0.114), `directional_share` (-0.134), `trend_persistence` (-0.084), `ema_crossings` (+0.007).
All liquidity features. All relative-strength features. Most acceleration features.

By category, features earning better than `no_improvement` on the 820-name panel: volatility 5 of 8, trend 3 of 11, acceleration 2 of 7, liquidity 2 of 7, relative 2 of 3, quality **0 of 7**.

The reading is straightforward.
BASE is already a trend filter, so more trend, momentum or trend-quality information is redundant with the screen itself.
Volatility is the one dimension BASE says nothing about, and it is where the entire payoff sits.

### Likely overfit - reject

**`vol_percentile` and `vol_ratio`.**
These were the headline result on the 820-name panel: `vol_percentile` beat BASE at **9 of 9** parameter settings, +0.103.
On the 250-name subsample it is **-0.057 at 3 of 8**. `vol_ratio` goes +0.061 (9/12) to -0.090 (1/12).
Both flip sign.

This is a failure mode worth naming separately, because parameter stability did not catch it.
Both features were stable across their whole parameter grid and still did not survive a change of sample.
Parameter breadth and sample breadth are different tests and a filter has to pass both.

**The spectral and wavelet family.**
`wavelet_lf_ratio` and `wavelet_hf_energy` are the two worst features in the study at -0.257, beating BASE at **0 of 9** settings.
`spectral_entropy` -0.157 (0 of 4 folds), `spectral_lf_ratio` -0.090 (2 of 12), `dominant_freq_stability` -0.041.

This answers the brief's explicit question directly: no, causal spectral features do not identify clean trending regimes better than efficiency ratio or R-squared.
They are worse than both, and efficiency ratio and R-squared are themselves worth nothing here.

### Experimental, one positive

`kalman_slope` is the only category 7 or 8 feature that helps: +0.082, 8 of 9 settings beat BASE, and the tightest parameter spread of any trend feature at 0.318.
It has no cross-sample confirmation because the experimental panel is the subsample, so it is a lead rather than a result.
`savgol_slope` +0.047 and `l1_trend_slope` -0.006 do not clear the bar.

## Do filters stack?

No. At matched survivor count, one filter held tighter beats every two-filter stack tried.

| combination | names | Sharpe | CAGR | maxDD | Calmar |
|---|---|---|---|---|---|
| `downside_vol` @ q=0.49 | 110 | **1.695** | 0.250 | -0.230 | 1.088 |
| `downside_vol` + `distance_from_high` | 110 | 1.605 | 0.258 | -0.251 | 1.030 |
| `downside_vol` + `turnover_consistency` | 131 | 1.553 | 0.265 | -0.248 | 1.069 |
| `downside_vol` + `trend_deterioration` | 142 | 1.543 | 0.252 | -0.239 | 1.056 |
| `downside_vol` @ q=0.7 | 158 | 1.532 | 0.248 | -0.243 | 1.018 |
| `downside_vol` + `hurst_vr` | 110 | 1.525 | 0.259 | -0.250 | 1.037 |
| BASE | 226 | 1.500 | 0.279 | -0.252 | 1.105 |
| `downside_vol` + `vol_adjusted_momentum` | 110 | 1.475 | 0.245 | -0.265 | 0.925 |

The target shape of Trend x Trend Strength x Trend Quality x Relative Strength x Volatility x Liquidity x Risk is not supported by this evidence.
What the evidence supports is two terms: **the trend screen, and one volatility filter.**
Every additional term tested subtracts.

## Threats to these conclusions

**Fold variance dominates.** One of four OOS folds has BASE at negative Sharpe. A +0.10 Sharpe edge is not separable from that.

**Overlapping forward returns.** The forward-return statistics pool every (date, name) pair; observations overlap heavily and the effective sample is far smaller than the raw counts. They are reported but are not load-bearing.

**One universe, one regime.** Indian mid/small caps over a period that was mostly a bull market. BASE earning 27.9% CAGR says as much about the window as about the screen.

**A bug that produced a spectacular false positive.** The first run ranked `trend_deterioration` at Sharpe 6.5. It is a binary flag, and percentile-ranking a mostly-tied column pushed the 95% tied block below the 30% cut, leaving 3-to-7 name portfolios. Fixed by keeping tied blocks whole (`method="min"`) and excluding any arm averaging under 20 names. Recorded because that class of error yields exactly the number one would most want to believe.

## Reproduce

```bash
cd /home/karneeshkar/grill-me-filters
./launch.sh filter_panels.py --years 5 --tag core
./launch.sh filter_eval.py --tag core --folds 4 --min-train 250
./launch.sh filter_combos.py --tag core --anchor downside_vol --q 0.7
```
