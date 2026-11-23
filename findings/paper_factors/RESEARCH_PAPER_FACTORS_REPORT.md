# Paper-Factor Research Round 2: accounting anomalies, reversal, risk factors, earnings momentum, trend/TSMOM

**Date**: first run 2026-08-12/13, base matrix rerun 2026-11-06 on main at `2ff190e` plus this branch, India rerun on the point-in-time universe 2026-11-08.
**Method**: `screener backtest-rolling`, base config.
**Data**: yfinance prices, FMP fundamentals (new opt-in accounting fields).
**Universes**: India `nifty500_pit` (`--universe-config universes.yaml --point-in-time`, 840 candidate symbols, 6007 membership windows), US `sp500` (point-in-time from Wikipedia revisions, the default).
**Costs**: NSE statutory + 10 bps slippage for India; flat 1 bp commission + 5 bps slippage for US.
**Capital**: 100,000, top 10, equal slots.
**Windows**: trailing 1/2/3/5y, all ending 2026-08-12.
**Runs**: 160 base runs (20 strategies x 2 markets x 4 windows), 0 errors, 0 runs with zero trades.
Raw: `findings/paper_factors/results_paper_factors.csv` (the 160 rerun rows, India rows on `nifty500_pit`).
`findings/paper_factors/results_paper_factors_all.csv` keeps the 197 runs of the first version, including the lever variants, for history only.

## What changed since the first version

The numbers in this report replace the first version (2026-08-12/13).
Four things moved them.

1. **The 9 fundamental strategies made 0 trades after the rebase onto main.**
   FMP `ratios`, `key_metrics` and `enterprise_values` rows have no filing date, so main dropped them all, and `pe_ttm`, `pb_ttm`, `roe_ttm`, `market_cap` and `fcf_yield` were NaN.
   Every gate failed closed.
   #210 dates those rows from the later of the income and balance filing dates of the same period.
   This branch adds the same rule for the `enterprise_values` row that `fcf_yield` and `z_score` read (`a75311f`).
2. **The first India fundamental results had lookahead.**
   FMP India stamps `acceptedDate` at the period end itself, which is a placeholder and not a filing date.
   The first run used it as the filing date, so India fundamentals were visible about 45 to 60 days before Indian companies publish.
   Main now treats those rows as period end + 60 days (`INDIA_FUNDAMENTAL_FILING_LAG_DAYS`) and labels them estimated.
   On the same (non-PIT) universe, India `piotroski_value` fell from 1.35 to 0.79 mean Sharpe and `fcf_yield_value` from 1.09 to 0.53, which fits a lookahead that was removed.
3. **The non-fundamental strategies also moved.**
   They use no fundamentals, but main changed between the two runs, and the price data was fetched again.
   For example, India `ma_timing_200` (non-PIT) went from 0.91 to 1.73 and US `tsmom_12_1` from 1.72 to 1.32.
   Treat the size of these moves as a measure of how sensitive single-path results are, not as a finding about the strategies.
4. **India now runs on the point-in-time Nifty 500.**
   The first version and the 2026-11-06 rerun used `--universe nifty500`, which applies today's 500 members to every past date.
   Today's members are the stocks that survived and grew into the index, so that universe has survivorship bias.
   The India rows now use `nifty500_pit`, where a name is eligible only on days it was in the index.
   Same code, same windows: India `ma_timing_200` 2y is +0.97 on today's list and -0.66 on PIT, and `hurst_trend_quality` 5y is +1.83 and +1.04.
   The "non-PIT mean" column in the India table is the 2026-11-06 result, for comparison.
   US rows did not change: `sp500` was already point-in-time.

The lever experiments below were not rerun.
Their numbers come from the first version.

## Strategies (20 new, +1 variant; all in `screener/strategies/plugins/`)

| Family | File | Strategies | Papers |
|---|---|---|---|
| Accounting anomalies | accounting_anomalies.py | `sloan_low_accruals`, `piotroski_value` (+`_f6`), `gross_profitability`, `conservative_investment` | Sloan 1996; Piotroski 2000; Novy-Marx 2013; Fama-French 2015; Altman 1968 |
| Volatility/beta/risk | beta_volatility.py | `low_idio_vol`, `betting_against_beta`, `downside_risk`, `max_avoidance` | Ang-Hodrick-Xing-Zhang 2006; Frazzini-Pedersen 2014; Ang-Chen-Xing 2006; Bali-Cakici-Whitelaw 2011 |
| Earnings momentum | earnings_momentum.py | `pead_drift`, `earnings_momentum`, `fcf_yield_value`, `qmj_quality` | Bernard-Thomas 1989; Chan-Jegadeesh-Lakonishok 1996; LSV 1994; Asness-Frazzini-Pedersen 2019 |
| Reversal & 52-wk | reversal_52week.py | `lt_reversal_path`, `str_reversal_trend`, `gw52_proximity`, `hs_same_month` | De Bondt-Thaler 1985; Jegadeesh 1990; George-Hwang 2004; Heston-Sadka 2008 |
| Trend/TSMOM | trend_technical.py | `tsmom_12_1`, `kama_trend`, `hurst_trend_quality`, `ma_timing_200` | Moskowitz-Ooi-Pedersen 2012; Kaufman 1995; Hurst 1951 / Lo-MacKinlay 1988; Zakamulin 2017; Odean 1998 |

Also: **fundamentals.py** gained 16 opt-in fields (`--fundamental-field`): piotroski_fscore, accruals, gross_profit_to_assets, asset_growth, z_score, fcf_yield, operating_cash_flow, free_cash_flow, total_assets, roa_ttm, asset_turnover, current_ratio, interest_coverage, dividend_yield_ttm, gross_margin_ttm, net_margin_ttm.
Balance and cash-flow sections are fetched on demand, and 0-filled FMP cash-flow quarters are treated as missing.

## Headline: trend leads India, Piotroski value leads the US

Sharpe (rf hurdle) by window, base config, sorted by the mean of the four windows.
F marks a strategy that reads FMP fundamentals.
Hold is 126 days except `pead_drift` and `kama_trend` (63), `hs_same_month` and `str_reversal_trend` (21), and `ma_timing_200` (250).

### India (Nifty 500, point-in-time)

Benchmark `^NSEI`: -0.2% (1y), +0.4% (2y), +25.7% (3y), +47.8% (5y), total return.

| Strategy | 1y | 2y | 3y | 5y | mean | min | non-PIT mean | trades (1y/2y/3y/5y) |
|---|---|---|---|---|---|---|---|---|
| `hurst_trend_quality` | 1.97 | 0.57 | 1.43 | 1.04 | **1.25** | **0.57** | 1.85 | 41/79/104/163 |
| `tsmom_12_1` | 1.21 | 0.39 | 1.19 | 1.10 | 0.97 | 0.39 | 1.39 | 24/44/66/108 |
| `piotroski_value` F | 0.98 | 0.46 | 1.01 | 1.08 | 0.88 | 0.46 | 0.79 | 20/40/60/100 |
| `fcf_yield_value` F | 1.39 | 0.09 | 0.51 | 0.70 | 0.67 | 0.09 | 0.53 | 20/40/61/101 |
| `earnings_momentum` F | 1.39 | 0.20 | 0.48 | 0.59 | 0.67 | 0.20 | 0.91 | 20/41/61/101 |
| `gw52_proximity` | 0.12 | 0.20 | 1.23 | 0.89 | 0.61 | 0.12 | 0.71 | 20/40/60/100 |
| `conservative_investment` F | 0.93 | -0.32 | 0.80 | 0.87 | 0.57 | -0.32 | 0.99 | 18/35/57/95 |
| `qmj_quality` F | -0.25 | 0.04 | 1.57 | 0.82 | 0.54 | -0.25 | 0.88 | 11/21/37/64 |
| `downside_risk` | 0.12 | 0.42 | 1.10 | 0.42 | 0.52 | 0.12 | 1.01 | 20/40/60/100 |
| `max_avoidance` F | 0.73 | -0.04 | 0.65 | 0.52 | 0.47 | -0.04 | 0.91 | 20/40/60/99 |
| `betting_against_beta` | 0.31 | 0.39 | 0.81 | 0.29 | 0.45 | 0.29 | 1.17 | 20/40/60/100 |
| `sloan_low_accruals` F | 0.71 | -0.10 | 0.70 | 0.32 | 0.41 | -0.10 | 0.27 | 20/41/61/100 |
| `str_reversal_trend` | 0.54 | -0.01 | 0.70 | 0.37 | 0.40 | -0.01 | 0.69 | 120/230/340/562 |
| `low_idio_vol` | 0.10 | 0.09 | 0.73 | 0.63 | 0.39 | 0.09 | 0.50 | 20/40/60/100 |
| `pead_drift` F | 0.27 | -0.44 | 0.66 | 0.55 | 0.26 | -0.44 | 1.47 | 40/81/120/200 |
| `lt_reversal_path` | 0.15 | -0.18 | 0.51 | 0.45 | 0.23 | -0.18 | 0.57 | 20/40/60/100 |
| `hs_same_month` | 0.36 | -0.33 | 0.64 | -0.06 | 0.15 | -0.33 | 0.57 | 120/230/340/570 |
| `ma_timing_200` | 0.37 | -0.66 | 0.37 | 0.44 | 0.13 | -0.66 | 1.73 | 29/74/85/123 |
| `gross_profitability` F | -0.76 | -0.61 | 0.47 | 0.54 | -0.09 | -0.76 | 0.30 | 7/15/24/61 |
| `kama_trend` | -0.50 | -1.04 | -0.71 | -1.16 | -0.85 | -1.16 | -0.12 | 273/572/847/1466 |

**India verdict**: `hurst_trend_quality` is still the best India strategy, but much weaker than on the survivorship-biased list.
Its mean Sharpe is 1.25 (non-PIT 1.85), its worst window is 0.57 (2y), and its 5-year max drawdown is -35%.
`tsmom_12_1` is second (0.97), so price trend persistence is the only family that holds up.
`piotroski_value` is third (0.88, min 0.46) and is now the best India fundamental strategy.
It is one of only three strategies that did not fall on PIT, together with `fcf_yield_value` and `sloan_low_accruals`.
`ma_timing_200` (1.73 to 0.13) and `pead_drift` (1.47 to 0.26) lose most of their result, so their non-PIT numbers came mostly from survivorship.
No India strategy has every window above 0.6.
`kama_trend` is negative in every window and trades too much (1466 trades in 5 years).

### US (S&P 500)

Benchmark `SPY`: +21.5% (1y), +48.3% (2y), +78.9% (3y), +85.4% (5y), total return.

| Strategy | 1y | 2y | 3y | 5y | mean | min | trades (1y/2y/3y/5y) |
|---|---|---|---|---|---|---|---|
| `piotroski_value` F | 2.99 | 1.69 | 1.49 | 1.25 | **1.85** | **1.25** | 20/41/61/100 |
| `fcf_yield_value` F | 1.55 | 1.46 | 1.54 | 1.16 | 1.43 | 1.16 | 20/40/60/100 |
| `ma_timing_200` | 1.81 | 1.29 | 1.37 | 1.00 | 1.37 | 1.00 | 20/48/71/124 |
| `tsmom_12_1` | 1.36 | 1.48 | 1.42 | 1.01 | 1.32 | 1.01 | 25/46/66/109 |
| `max_avoidance` F | 1.76 | 1.29 | 1.09 | 0.56 | 1.18 | 0.56 | 20/40/60/100 |
| `lt_reversal_path` | 2.51 | 0.91 | 0.65 | 0.35 | 1.10 | 0.35 | 21/41/61/100 |
| `gw52_proximity` | 0.76 | 1.57 | 1.18 | 0.82 | 1.08 | 0.76 | 20/40/60/102 |
| `sloan_low_accruals` F | 0.94 | 1.17 | 1.18 | 0.94 | 1.06 | 0.94 | 20/40/60/100 |
| `hurst_trend_quality` | 1.20 | 1.30 | 0.87 | 0.34 | 0.93 | 0.34 | 45/79/119/196 |
| `pead_drift` F | 0.80 | 0.94 | 1.09 | 0.63 | 0.87 | 0.63 | 40/80/120/200 |
| `earnings_momentum` F | 1.07 | 0.81 | 0.97 | 0.28 | 0.78 | 0.28 | 20/40/61/100 |
| `kama_trend` | 1.40 | 0.41 | 0.37 | 0.42 | 0.65 | 0.37 | 292/582/872/1433 |
| `betting_against_beta` | 1.28 | 0.72 | 0.11 | 0.15 | 0.57 | 0.11 | 21/41/61/101 |
| `conservative_investment` F | 0.93 | 0.53 | 0.45 | 0.34 | 0.56 | 0.34 | 20/41/61/102 |
| `str_reversal_trend` | 0.56 | 0.52 | 0.27 | 0.44 | 0.45 | 0.27 | 120/230/350/573 |
| `qmj_quality` F | 0.52 | 0.48 | 0.59 | -0.03 | 0.39 | -0.03 | 10/20/28/49 |
| `downside_risk` | 0.77 | 0.39 | -0.08 | 0.12 | 0.30 | -0.08 | 21/41/62/105 |
| `hs_same_month` | -0.08 | 0.43 | 0.40 | 0.27 | 0.25 | -0.08 | 120/230/350/571 |
| `low_idio_vol` | -0.29 | 0.28 | 0.29 | 0.57 | 0.21 | -0.29 | 21/42/62/102 |
| `gross_profitability` F | -0.21 | -0.20 | 0.08 | 0.07 | -0.07 | -0.21 | 15/23/31/39 |

**US verdict**: `piotroski_value` (F-Score >= 7 inside the cheap universe) is still the best US strategy.
Its mean Sharpe is 1.85 (first version 1.89), no window is below 1.25, the hit rate is 74% to 85%, and the 5-year max drawdown is -24%.
US filing dates are real FMP timestamps, so this result did not depend on the India lookahead.
`fcf_yield_value` is second (mean 1.43, min 1.16, 5-year max drawdown -20%).
`ma_timing_200` and `tsmom_12_1` follow, with every window at 1.00 or above.
`sloan_low_accruals` is weak on the mean (1.06) but is the most even, with all four windows between 0.94 and 1.18.

### Across markets

`tsmom_12_1` and `piotroski_value` are the only strategies in the top four of both tables.
`piotroski_value` is first in the US (1.85) and third in India (0.88).
`ma_timing_200` is third in the US (1.37) but near last in India on PIT (0.13).
`pead_drift` is mid-table in the US (0.87) and weak in India on PIT (0.26).
`gross_profitability` is last or near last in both markets, and it trades thinly.

## Lever experiments (32 runs, first version, not rerun)

These ran on the first version and the old data path.
The India levers used today's Nifty 500 list (survivorship bias), and the India fundamental lever (relaxed F-Score) also used the placeholder filing dates with lookahead.
Read them as directions to test again, not as results.

- **Moreira-Muir vol targeting** (`--sizing inverse_vol --sizing-risk-pct 0.002`): the default 0.01 risk-pct never binds (equal-slot cap).
  At 0.002 it binds and helped India flat windows: tsmom 2y 0.04 to 0.13, max_avoidance 1y -0.10 to 0.57, hurst 1y/2y slightly up.
  US was roughly neutral.
- **Trailing stop 15% on ma_timing_200**: US improved (mean 1.60 to 1.75, MDD -42% to -22%), India worsened (-0.15 in 1y).
- **Sector-neutral**: gw52_proximity US 5y 0.83 to 1.29; qmj_quality US went to 0.00 Sharpe, so it was dropped.
- **Relaxed F-Score (>= 6) India**: 0.45/0.38/1.57/1.12, worse than >= 7 on the same data.

## Data-infrastructure changes

- `screener/backtester/fundamentals.py`: 16 opt-in fields.
  Balance and cash-flow FMP sections are fetched on demand, mapped by field.
  `_clean_cashflow` treats FMP's 0-filled CFO and free cash flow for profitable Indian names as missing, otherwise F-scores silently punish RELIANCE and HDFCBANK.
  Accruals, asset growth, z-score and fcf_yield are computed from dated statements.
  NaN fails gates closed (missing data excludes the name).
- Enterprise-value rows have no filing date, so `fcf_yield` and `z_score` now use the statement date of the same period (`a75311f`, test `test_fmp_enterprise_value_source_uses_statement_date`).

## Caveats

- Each cell is one path on one dataset.
  The non-fundamental strategies moved by up to 0.8 Sharpe between two runs with the same config, so differences of a few tenths in the tables are not reliable.
- India fundamental dates are estimated (period end + 60 days), not real publication dates, and original restatement vintages are not available.
- FMP India cash-flow coverage is gappy (0-filled quarters, for example RELIANCE).
  Those names go NaN and fail closed, so India F-Score, accruals and fcf_yield screens see a thinner universe.
- Both universes are point-in-time.
  `nifty500_pit` has 4 observation gaps of 180 days or more (largest 648 days), and its dates are snapshot dates, not guaranteed index effective dates.
  97 historical members have no yfinance price history (delisted or renamed), so they cannot be selected; this leaves some survivorship bias.
- `piotroski_value` US 1y Sharpe 2.99 is one 20-trade window with an 85% hit rate; the 2/3/5y figures (1.69/1.49/1.25) are the reliable ones.
- India 1y and 2y are flat regimes (benchmark -0.2% and +0.4%), so long-only results there depend on stock selection more than on market exposure.
- Research, not financial advice.

## Files

- `screener/strategies/plugins/`: accounting_anomalies.py, beta_volatility.py, earnings_momentum.py, reversal_52week.py, trend_technical.py (+ spec.py registration).
- `screener/backtester/fundamentals.py`: opt-in accounting and quality fields.
- `findings/paper_factors/results_paper_factors.csv`: the 160 rerun base runs, full metric set.
- `findings/paper_factors/results_paper_factors_all.csv`: the 197 runs of the first version, kept for history.
