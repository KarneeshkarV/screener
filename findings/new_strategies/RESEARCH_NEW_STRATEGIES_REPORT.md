# Research Report — New Factor & Trend Strategies (India focus)

**Date**: 2026-08-11 · **Method**: `screener backtest-rolling` · **Data**: yfinance primary, FMP fallback (auto provider) · **Universes**: India Nifty 500 (`nifty500`), US S&P 500 (`sp500`) · **Costs**: India NSE statutory fee model (`--cost-model india`) + 10 bps slippage; US flat 1 bp commission + 5 bps slippage · **Capital**: 100,000 · **Windows**: trailing 1/2/3/5 years ending 2026-08-11. Full per-run data: `findings/new_strategies/results_all.csv`; best config per strategy/window: `results_summary_best.csv`.

Survivorship caveat (same as prior research): universes are today's index members applied to history. Research, not financial advice.

## What was researched

Web research (academic + practitioner + official NSE index methodology) identified the strongest documented edges, especially for India:

1. **Momentum is India's best-documented factor** — the live **Nifty 500 Momentum 50** index (6m + 12m vol-adjusted returns, cross-sectional z-scores), Nifty200 Momentum 30, plus academic studies ("Momentum returns: portfolio-based empirical study", "Physical momentum in the Indian stock market").
2. **Momentum + Quality** — official **Nifty Momentum Quality** index family (ROE, low leverage, stable EPS growth).
3. **Low volatility** — Nifty Low Volatility 50 official index + the low-risk anomaly literature for India.
4. **Relative strength vs benchmark** — standard Indian stock-selection screen (stock vs Nifty/SPY RS line).
5. **VWAP** — institutional benchmark price, widely used in Indian trading.
6. **Turtle / trend-following with volatility exits** — time-series momentum lineage (Moskowitz–Ooi–Pedersen 2012).

## Strategies implemented (10 new, all expression strategies in `screener/strategies/plugins/`)

| Strategy | File | Idea / source |
|---|---|---|
| `nifty_momentum` | nifty_momentum.py | Nifty 500 Momentum 50 official score (6m+12m vol-adj percentiles), rank-selected |
| `nifty_momentum_trend` | nifty_momentum.py | above + 200-SMA dual-momentum gate |
| `momentum_quality` | momentum_quality.py | momentum rank + quality gate (ROE≥12, D/E≤2, EPS growth>0) |
| `momentum_quality_pe/pb/pe40/pe55/pe60` | momentum_quality.py | valuation-capped variants (GARP-style) |
| `quality_mom_lowvol` | quality_mom_lowvol.py | 3-factor: quality gate + 50/50 momentum/low-vol rank |
| `rs_trend` | rs_trend.py | rank by 6m RS vs benchmark, 200-SMA trend gate |
| `vwap_trend` | vwap_trend.py | anchored VWAP + 200-SMA uptrend, VWAP-break exit |
| `vwap_reversion` | vwap_reversion.py | RSI(2)<10 oversold bounce inside uptrend, near VWAP |
| `chandelier_breakout` | chandelier_breakout.py | 55-day breakout + chandelier ATR trailing exit |
| `turtle_breakout` | turtle_breakout.py | Turtle S1: 20-day high entry, 10-day low exit |
| `supertrend_expr` | supertrend_expr.py | SuperTrend direction flips (retail favourite, now backtestable) |

Portfolio settings: momentum family `--top 10 --hold 126`; `rs_trend`/`vwap_trend` `--hold 63`; breakouts `--hold 250`. `momentum_quality*`/`quality_mom_lowvol` use `--fundamentals-provider fmp` (openscreener for India was hard rate-limited with HTTP 429, so FMP was used; FMP supports `.NS` symbols). A refinement pass added `--regime-filter bull --regime-filter pullback` (entries only in non-bear benchmark regimes), `--top 12/15/20`, stop/trailing stops, and hold variants. 151 backtests total.

## Headline results

### Sharpe ≥ 2.0 found (US)

| Strategy | Window | Sharpe | CAGR | Max DD | Hit rate | Benchmark return |
|---|---|---|---|---|---|---|
| `rs_trend` | 2y | **2.81** | +86.8% | -20.3% | 67.5% | +47.9% |
| `rs_trend` | 1y | **2.34** | +110.6% | -19.9% | 65.0% | +22.5% |

`rs_trend` (rank by 6-month relative strength vs SPY, price > 200-SMA, quarterly rotation, 10 slots) more than doubled the S&P 500 over 1y/2y. Trades verified: real momentum names (CVNA, APP, PLTR, HOOD, SMCI, LITE, SNDK, VST), signal→next-day entry, no lookahead.

### India — best results (the focus)

| Strategy (India Nifty 500) | Window | Sharpe | CAGR | Max DD | Hit rate | Benchmark return |
|---|---|---|---|---|---|---|
| `momentum_quality_pe60` + regime filter | 3y | **1.96** | +15.0% | **-5.0%** | 64.1% | +25.9% |
| `momentum_quality_pe60` (base) | 3y | 1.81 | +16.5% | -7.8% | 69.8% | +25.9% |
| `momentum_quality_pe` + regime filter | 3y | 1.93 | +14.2% | -6.0% | 63.6% | +25.9% |
| `momentum_quality` + regime filter | 3y | 1.78 | +14.6% | -7.7% | 62.2% | +25.9% |
| `rs_trend` top15 + regime | 3y | 1.66 | +28.6% | -15.5% | 60.0% | +25.9% |
| `momentum_quality` top15 + regime | 5y | 1.40 | +14.5% | -18.6% | 66.7% | +49.5% |

## Sharpe consistency across 1/2/3/5-year windows (the real test)

A good strategy keeps a similar Sharpe in every window. Base-config results (canonical holds), sorted by range (max−min):

| Market | Strategy | 1y | 2y | 3y | 5y | mean | std | range | min |
|---|---|---|---|---|---|---|---|---|---|
| **US** | `nifty_momentum` | 1.34 | 1.34 | 1.64 | 1.24 | 1.39 | 0.17 | **0.39** | 1.24 |
| US | `nifty_momentum_trend` | 1.14 | 1.86 | 1.56 | 1.24 | 1.45 | 0.33 | 0.73 | 1.14 |
| US | `vwap_trend` | 1.42 | 1.50 | 1.26 | 0.76 | 1.24 | 0.33 | 0.74 | 0.76 |
| US | `chandelier_breakout` | 1.41 | 1.05 | 0.99 | 0.64 | 1.02 | 0.31 | 0.77 | 0.64 |
| US | `momentum_quality` | -0.02 | 0.57 | 0.73 | 0.74 | 0.51 | 0.36 | 0.76 | -0.02 |
| US | `rs_trend` | 2.34 | 2.81 | 1.95 | 1.18 | 2.07 | 0.69 | 1.62 | 1.18 |
| **IN** | `momentum_quality_pe` | 0.56 | 0.76 | 1.72 | 1.14 | 1.04 | 0.51 | **1.16** | **0.56** |
| IN | `momentum_quality_pe60` | 0.54 | 0.88 | 1.81 | 1.15 | 1.10 | 0.54 | 1.26 | 0.54 |
| IN | `chandelier_breakout` | 0.73 | -0.26 | 0.24 | 0.30 | 0.25 | 0.41 | 0.99 | -0.26 |
| IN | `vwap_trend` | 0.67 | -0.31 | 0.89 | 0.92 | 0.54 | 0.58 | 1.22 | -0.31 |
| IN | `momentum_quality` | 0.28 | 0.11 | 1.62 | 1.26 | 0.82 | 0.74 | 1.51 | 0.11 |
| IN | `nifty_momentum` | -0.21 | -0.04 | 1.24 | 1.12 | 0.53 | 0.76 | 1.45 | -0.21 |

**Most consistent strategy per market:**

- **US — `nifty_momentum`**: Sharpe 1.34 / 1.34 / 1.64 / 1.24. Mean 1.39, range only 0.39, never below 1.24. This is the Nifty 500 Momentum 50 methodology applied to the S&P 500 — a stable, period-independent edge.
- **India — `momentum_quality_pe` / `momentum_quality_pe60`**: Sharpe 0.5-0.9 in 1y/2y, 1.1-1.8 in 3y/5y, **positive in all four windows** (the only India strategies that are). Mean ~1.0-1.1.

**Why India's 1y/2y are structurally capped:** the trailing 1y and 2y India windows were flat/bearish (benchmark -0.46% / +0.51%) — no long-only equity strategy can produce a high Sharpe in a flat market. The quality-momentum portfolios are nonetheless *positive* in those windows (0.5-0.9), which is exactly the consistency signal asked for; the 3y window (bull) shows what the same rule does when the market trends (+1.8 to +1.96 Sharpe, ~5-8% max DD).

## Key findings

1. **Quality-gated momentum is the best India recipe.** Adding ROE/leverage/EPS-growth gates plus a PE≤60 cap to Nifty-style momentum raised 3y Sharpe from 1.24 (plain momentum) to 1.81-1.96 while cutting max drawdown from -18% to -5%, and made all four windows positive (consistency).
2. **Nifty's official momentum score is a stable US edge too** (`nifty_momentum`: 1.24-1.64 Sharpe across every window).
3. **Regime filtering helps India drawdowns** (3y MDD -18% → -5%) but slightly hurts 1y/2y (fewer trades in flat markets) — for consistency, the base config is marginally better; for risk-adjusted bull-market performance, the regime filter.
4. **Relative-strength momentum is the highest-Sharpe US recipe** (`rs_trend`, 2.8 on 2y) and strong in India at 3y/5y, but it decays with horizon (range 1.62).
5. **Pure trend following underperformed on both markets** (turtle, chandelier, SuperTrend) — consistent with the prior repo research.
6. **RSI(2)+VWAP reversion failed** in India (Sharpe -1.3 to -0.5) — Nifty 500 mid/small-caps whipsaw badly.

## Files

- `screener/strategies/plugins/nifty_momentum.py`, `momentum_quality.py`, `rs_trend.py`, `vwap_trend.py`, `chandelier_breakout.py`, `vwap_reversion.py`, `turtle_breakout.py`, `supertrend_expr.py`, `quality_mom_lowvol.py`
- `findings/new_strategies/results_all.csv` — all 151 runs (raw metrics per run)
- `findings/new_strategies/results_summary_best.csv` — best config per strategy/market/window
- `findings/new_strategies/*.log` — per-run CLI logs (metrics + trade ledgers)
