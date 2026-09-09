# Non-Momentum Strategy Research — do value/quality/flow/seasonal/volatility beat momentum?

**Date**: 2026-08-12 · **Method**: `screener backtest-rolling` (base config, no regime filter) · **Data**: yfinance + FMP fallback · **Universes**: India Nifty 500, US S&P 500 · **Costs**: India NSE statutory + 10 bps slippage; US flat 1 bp + 5 bps slippage · **Capital**: 100,000 · **Windows**: trailing 1/2/3/5y ending 2026-08-12 · **Runs**: 176 (20 new strategies + 2 momentum benchmarks × 2 markets × 4 windows). Full data: `findings/new_strategies/results_nonmomentum.csv` (raw), `results_nonmomentum_summary.csv` (per-strategy Sharpe table).

Explored via 5 parallel research sub-agents: VALUE & GARP, QUALITY & DEFENSIVE, PRICE-VOLUME & FLOW, CALENDAR & SEASONALITY, VOLATILITY & TECHNICAL REGIME.

## Strategies added (20 new, all in `screener/strategies/plugins/`)

| Family | Strategies (file) |
|---|---|
| Value/GARP | `value_rank`, `garp`, `deep_value`, `value_momentum_harness` (value_garp.py) — Nifty500 Value 50 (E/P+B/P blend + quality floor), PEG/GARP, cheap+quality+200d trend, Asness "value & momentum everywhere" harness |
| Quality/defensive | `quality_lowvol`, `quality_lowbeta`, `quality_stability`, `quality_value` (quality_defensive.py) — Nifty Quality 30 / AQLV-style: quality gate (ROE≥10, D/E≤2.5, EPS growth>0) ranked by lowest vol / lowest beta / stability / +valuation caps |
| Price-volume/flow | `delivery_accumulation`, `volume_surge`, `obv_flow_trend`, `cmf_flow_factor` (delivery_accumulation.py, volume_flow.py) — delivery% accumulation (India, optional), Gervais-Kaniel-Mingelgrin high-volume premium, Granville OBV, Chaikin Money Flow factor |
| Calendar/seasonality | `seasonal_strong_trend`, `tom_window_trend`, `pre_holiday_trend`, `nov_apr_trend` (calendar_seasonality.py) — strong months Apr/Jul/Nov/Dec (India), Ariel/McConnell turn-of-month trading-day window, pre-holiday effect, Halloween effect; all with 200-SMA trend overlay |
| Volatility/technical | `vcp_breakout`, `vol_expansion_breakout`, `vol_target_lowvol`, `keltner_squeeze_breakout` — Minervini VCP contraction breakout, ATR-expansion Donchian breakout, Moreira-Muir vol-managed low-vol, Carter TTM-squeeze |

## Headline answer: YES — non-momentum beats momentum on India

Sharpe by window (1y/2y/3y/5y), base config. **Bold = beats the best momentum benchmark on that market.**

### India (Nifty 500) — sorted by mean Sharpe

| Strategy | 1y | 2y | 3y | 5y | mean | range | min |
|---|---|---|---|---|---|---|---|
| **`value_rank`** | 2.31 | 1.75 | 2.13 | 1.68 | **1.97** | 0.63 | 1.68 |
| **`value_momentum_harness`** | 1.94 | 1.63 | 1.94 | 1.49 | **1.75** | **0.46** | 1.49 |
| **`quality_value`** | 1.50 | 1.18 | 1.94 | 1.48 | **1.53** | 0.76 | 1.18 |
| **`seasonal_strong_trend`** | 2.46 | 0.79 | 1.40 | 1.36 | **1.50** | 1.67 | 0.79 |
| `quality_lowbeta` | 0.98 | 0.43 | 1.97 | 1.56 | 1.23 | 1.55 | 0.43 |
| `momentum_quality_pe60` (momentum bench) | 0.55 | 0.88 | 1.81 | 1.15 | 1.10 | 1.26 | 0.55 |
| `nifty_momentum` (momentum bench) | −0.15 | −0.02 | 1.59 | 1.11 | 0.63 | 1.74 | −0.15 |

**India verdict**: the three value-family strategies and the India-seasonality strategy all beat BOTH momentum benchmarks on mean Sharpe. `value_momentum_harness` is the most period-stable non-momentum strategy (Sharpe 1.49–1.94, range 0.46 vs momentum's 1.26–1.74). `value_rank` has the highest mean (1.97) — its 1y/2y windows are thin (4/11 trades; FMP India fundamentals only let a handful of names through in short windows), while 3y/5y (23/51 trades) are solid. `quality_value` is the most broadly reliable (12/22/38/72 trades, all windows positive).

### US (S&P 500) — sorted by mean Sharpe

| Strategy | 1y | 2y | 3y | 5y | mean | range | min |
|---|---|---|---|---|---|---|---|
| **`value_rank`** | 1.99 | 1.57 | 1.57 | 0.81 | **1.48** | 1.18 | 0.81 |
| `deep_value` | 1.88 | 1.29 | 1.62 | 0.79 | 1.40 | 1.09 | 0.79 |
| `delivery_accumulation` | 1.72 | 1.68 | 1.24 | 0.91 | 1.39 | 0.81 | 0.81 |
| `nifty_momentum` (momentum bench) | 1.36 | 1.34 | 1.60 | 1.21 | **1.38** | **0.38** | 0.38 |
| `value_momentum_harness` | 1.97 | 1.32 | 1.26 | 0.75 | 1.32 | 1.23 | 0.75 |
| `obv_flow_trend` | 1.21 | 1.11 | 1.18 | 0.82 | 1.08 | 0.40 | 0.40 |

**US verdict**: `value_rank` edges momentum on mean Sharpe (1.48 vs 1.38), but `nifty_momentum` remains the most consistent strategy on US (range 0.38, min 0.38). Value works on both markets; momentum is still the consistency champion in the US.

## Key findings

1. **Value beats momentum on India.** The Nifty500 Value 50 methodology (`value_rank`) and Asness-style value+momentum harness deliver mean Sharpe 1.75–1.97 vs 0.63–1.10 for the momentum benchmarks, with tighter period-to-period consistency (`value_momentum_harness` range 0.46). Value was under-researched in the repo's earlier momentum study.
2. **Quality + valuation cap (`quality_value`) is the most reliable India strategy**: positive Sharpe in all four windows with healthy trade counts (12–72), mean 1.53, 3y Sharpe 1.94 with −10.7% max DD.
3. **India seasonality (`seasonal_strong_trend`)** — long Apr/Jul/Nov/Dec with a 200-SMA gate — is exceptional in the flat 1y window (Sharpe 2.46, MDD −2.2%, ~34% exposure) and solid at 3y/5y (1.40/1.36), confirming the well-documented April/December strength and March weakness in Indian markets.
4. **Volume/flow signals work but weaker**: high-volume premium (`volume_surge`) and delivery accumulation are positive in 3/4 windows; OBV/CMF are positive but thin in 1y/2y.
5. **Volatility/breakout family underperformed** (VCP, vol-expansion, squeeze, vol-target: mean Sharpe 0.08–0.53 India) — breakout timing is not an edge in these windows, consistent with the prior momentum study.
6. **Calendar micro-effects are fragile**: pre-holiday and turn-of-month are good 1y, decay by 5y (and fail on US tom_window). `nov_apr_trend` (Halloween) only works on US.

## Caveats

- India value 1y/2y trade counts are thin (4–11) for `value_rank`; trust 3y/5y. FMP India fundamentals coverage drives this.
- Survivorship bias: today's index members applied to history (same as all repo research).
- Fundamentals are point-in-time with filing lag (FMP 1d) — good, but FMP India ratios history is shorter (~2019+), so early-window names may be underrepresented.
- `delivery_accumulation` on US runs on pure OHLCV (delivery is India-only) — the US numbers are the VWAP+OBV leg.
- Research, not financial advice.

## Files

- `screener/strategies/plugins/`: value_garp.py, quality_defensive.py, calendar_seasonality.py, delivery_accumulation.py, volume_flow.py, vcp_breakout.py, vol_expansion_breakout.py, vol_target_lowvol.py, keltner_squeeze_breakout.py (all registered in spec.py)
- `findings/new_strategies/results_nonmomentum.csv` (176 runs), `results_nonmomentum_summary.csv`
- `findings/new_strategies/nonmomentum_full/*.log` (per-run CLI logs)
