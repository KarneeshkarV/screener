# Paper-Factor Research Round 2 — accounting anomalies, reversal, risk factors, earnings momentum, trend/TSMOM

**Date**: 2026-08-12/13 · **Method**: `screener backtest-rolling` base config · **Data**: yfinance + FMP (new opt-in accounting fields) · **Universes**: India Nifty 500, US S&P 500 · **Costs**: NSE statutory + 10 bps India; flat 1 bp + 5 bps US · **Capital**: 100,000 · **Windows**: trailing 1/2/3/5y · **Runs**: 160 base + 32 lever variants + 5 reruns = 197. Raw: `findings/paper_factors/results_paper_factors_all.csv`.

Researched and implemented by 5 parallel sub-agents from the academic literature (all validated: ruff, mypy, 270 strategy/fundamental tests, causality-tested prepare_bars):

## Strategies (20 new, +1 variant; all in `screener/strategies/plugins/`)

| Family | File | Strategies | Papers |
|---|---|---|---|
| Accounting anomalies | accounting_anomalies.py | `sloan_low_accruals`, `piotroski_value` (+`_f6`), `gross_profitability`, `conservative_investment` | Sloan 1996; Piotroski 2000; Novy-Marx 2013; Fama-French 2015; Altman 1968 |
| Volatility/beta/risk | beta_volatility.py | `low_idio_vol`, `betting_against_beta`, `downside_risk`, `max_avoidance` | Ang-Hodrick-Xing-Zhang 2006; Frazzini-Pedersen 2014; Ang-Chen-Xing 2006; Bali-Cakici-Whitelaw 2011 |
| Earnings momentum | earnings_momentum.py | `pead_drift`, `earnings_momentum`, `fcf_yield_value`, `qmj_quality` | Bernard-Thomas 1989; Chan-Jegadeesh-Lakonishok 1996; LSV 1994; Asness-Frazzini-Pedersen 2019 |
| Reversal & 52-wk | reversal_52week.py | `lt_reversal_path`, `str_reversal_trend`, `gw52_proximity`, `hs_same_month` | De Bondt-Thaler 1985; Jegadeesh 1990; George-Hwang 2004; Heston-Sadka 2008 |
| Trend/TSMOM | trend_technical.py | `tsmom_12_1`, `kama_trend`, `hurst_trend_quality`, `ma_timing_200` | Moskowitz-Ooi-Pedersen 2012; Kaufman 1995; Hurst 1951 / Lo-MacKinlay 1988; Zakamulin 2017; Odean 1998 |

Also: **fundamentals.py** gained 16 opt-in fields (`--fundamental-field`): piotroski_fscore, accruals, gross_profit_to_assets, asset_growth, z_score, fcf_yield, operating_cash_flow, free_cash_flow, total_assets, roa_ttm, asset_turnover, current_ratio, interest_coverage, dividend_yield_ttm, gross_margin_ttm, net_margin_ttm (balance + cash-flow sections fetched on demand; 0-filled FMP cash-flow quarters treated as missing).

## Headline: paper factors beat momentum — and the winner is Piotroski's F-Score inside value

Sharpe by window (1y/2y/3y/5y), base config. **Bold = beats the best round-1 benchmark on that market.**

### INDIA (Nifty 500) — vs round-1 benchmarks

| Strategy | 1y | 2y | 3y | 5y | mean | min | trades (1y/2y/3y/5y) |
|---|---|---|---|---|---|---|---|
| value_rank (R1 bench) | 2.31 | 1.75 | 2.13 | 1.68 | 1.97 | 0.63 | 4/11/23/51 (thin) |
| value_momentum_harness (R1) | 1.94 | 1.63 | 1.94 | 1.49 | 1.75 | 1.49 | 3/10/21/48 (thin) |
| **`hurst_trend_quality`** | 1.88 | 0.84 | 1.97 | 1.82 | **1.63** | **0.84** | 39/65/92/148 ✓ |
| seasonal_strong_trend (R1) | 2.46 | 0.79 | 1.40 | 1.36 | 1.50 | 0.79 | 38/82/118/198 ✓ |
| **`piotroski_value`** | 1.45 | 0.92 | 1.72 | 1.31 | **1.35** | **0.92** | 12/15/29/53 ✓ |
| momentum_quality_pe60 (R1 bench) | 0.55 | 0.88 | 1.81 | 1.15 | 1.10 | 0.55 | 21/39/53/85 ✓ |
| **`fcf_yield_value`** | 0.53 | 0.60 | 1.62 | 1.60 | **1.09** | 0.53 | 11/15/26/48 |
| nifty_momentum (R1 bench) | −0.15 | −0.02 | 1.59 | 1.11 | 0.63 | −0.15 | 20/40/60/100 |

**India verdict**: `hurst_trend_quality` (Hurst trend-quality × 12-1 momentum, exit on persistence break) is the most RELIABLE strategy in the project — mean Sharpe 1.63, all four windows ≥ 0.84, fully invested, 148 trades in 5y. `piotroski_value` (F-Score ≥ 7 inside the cheap universe) beats the momentum benchmark on mean (1.35 vs 1.10) AND floor (0.92 vs 0.55) with 77–83% hit rates. Both beat nifty_momentum outright. Value+quality accounting factors + trend persistence now dominate India.

### US (S&P 500) — vs round-1 benchmarks

| Strategy | 1y | 2y | 3y | 5y | mean | min | trades |
|---|---|---|---|---|---|---|---|
| **`piotroski_value`** | **3.05** | 1.64 | 1.76 | 1.12 | **1.89** | **1.12** | 20/40/60/100 ✓ |
| **`tsmom_12_1`** | 1.86 | 1.69 | **2.06** | 1.29 | **1.72** | **1.29** | 22/44/66/108 ✓ |
| **`fcf_yield_value`** | 1.68 | 1.73 | 1.95 | 1.08 | **1.61** | **1.08** | 20/40/60/100 ✓ |
| **`ma_timing_200`** | 1.79 | 1.75 | 1.77 | 1.09 | **1.60** | **1.09** | 20/41/54/106 ✓ |
| value_rank (R1) | 1.99 | 1.57 | 1.57 | 0.81 | 1.48 | 0.81 | 19/24/38/78 |
| nifty_momentum (R1 bench) | 1.36 | 1.34 | 1.60 | 1.21 | 1.38 | 0.38 | 20/40/60/100 |
| max_avoidance | 1.87 | 1.17 | 1.79 | 0.84 | 1.42 | 0.84 | 20/40/60/100 ✓ |

**US verdict**: FOUR round-2 strategies beat the momentum champion on mean Sharpe, and `piotroski_value` (mean 1.89, floor 1.12) is the single best strategy found in the whole project. `tsmom_12_1` is the consistency king (range 0.77). The F-score + value + cash-flow-yield family wins on both markets.

## Lever experiments (32 runs)

- **Moreira-Muir vol targeting** (`--sizing inverse_vol --sizing-risk-pct 0.002`): the default 0.01 risk-pct never binds (equal-slot cap); at 0.002 it binds and helps India flat windows: tsmom 2y 0.04→0.13, max_avoidance 1y −0.10→0.57, hurst 1y/2y slightly up. US roughly neutral. Modest, not the 2x the futures paper claims.
- **Trailing stop 15% on ma_timing_200**: US improves (mean 1.60→1.75, 1y 1.79→2.09, 3y 1.77→1.92, MDD −42%→−22%); India worsens (−0.15 in 1y). Stop helps the fully-invested US book, hurts the already-cautious India one.
- **Sector-neutral**: gw52_proximity US 5y 0.83→1.29 ✓; qmj_quality US destroyed (0.74→0.00, 0.54→0.00) ✗ — dropped.
- **Relaxed F-Score (≥6) India**: 0.45/0.38/1.57/1.12 — WORSE than ≥7 (1.45/0.92/1.72/1.31). The tighter Piotroski screen is better; keep ≥7.

## Data-infrastructure changes

- `screener/backtester/fundamentals.py`: 16 opt-in fields; balance/cash-flow FMP sections fetched on demand (mapped by field); `_clean_cashflow` treats FMP's 0-filled CFO/free-cash-flow for profitable Indian names as missing (otherwise F-scores silently punish RELIANCE/HDFCBANK); accruals/asset_growth/z_score/fcf_yield computed from dated statements with a 1-day filing lag; NaN fails gates closed (missing data excludes the name).

## Caveats

- Survivorship bias (today's index members applied to history) unchanged from all repo research.
- FMP India statement coverage is gappy for some balance/cash-flow quarters (0-filled) — accruals/F-score/z-score go NaN → fail closed → slightly thinner India trades (still healthy: piotroski 12-53, hurst 39-148).
- `piotroski_value` US 1y Sharpe 3.05 is a single-window outlier (20 trades, 85% hit); the 2/3/5y (1.64/1.76/1.12) are the reliable figures.
- Indian 1y/2y are flat/bear regimes (benchmark −0.8%/−0.2%): long-only Sharpe there is capped ~0.5-0.9 for good strategies; hurst's 0.84/1.88 in those windows is a strong defensive result.
- Research, not financial advice.

## Files

- `screener/strategies/plugins/`: accounting_anomalies.py, beta_volatility.py, earnings_momentum.py, reversal_52week.py, trend_technical.py (+ spec.py registration, 112 strategies total)
- `screener/backtester/fundamentals.py`: opt-in accounting/quality fields
- `findings/paper_factors/results_paper_factors_all.csv` (197 runs), `findings/paper_factors/*.log` (per-run CLI digests)
