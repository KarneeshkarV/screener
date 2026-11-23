# Bollinger Band Mean Reversion

**Strategy key:** `bollinger_mean_reversion`

## Research source

**Book:** Bollinger, *"Bollinger on Bollinger Bands"*, 2001, McGraw-Hill.

- Publisher page: https://www.bollingerbands.com/bollinger-on-bollinger-bands
- Amazon: https://www.amazon.com/Bollinger-Bands-John/dp/0071373683

John Bollinger's book is the primary reference for the bands (SMA20 ± 2σ with
20-day *population* standard deviation). The mean-reversion interpretation is
statistical: roughly 95% of closes fall inside the bands, so a close below the
lower band marks a two-standard-deviation overreaction that tends to snap back
to the mean (the middle band). Bollinger himself stressed the bands are a
"relative definition of high and low" — the mean-reversion trade is one of the
two canonical band strategies (the other, band breakout, already exists in the
repo as `bb_breakout`).

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `close < bb_lower` — close below `sma(close,20) − 2σ` (population σ) |
| Exit | `close > bb_mid` — revert to the 20-day mean |

## Implementation notes

- The engine's Pine has no `stdev`, so `prepare_bars` precomputes
  `bb_lower`/`bb_mid` with pandas rolling stats. Population std (`ddof=0`)
  matches the repo's `bollinger_bands` indicator exactly.
- `required_lookback = 20` bars.
- Suggested holding semantics: mean reversion — default `--hold` 20 is fine.

## Expected behaviour

- Frequent trades, high win rate, small mean win.
- Classic failure mode: buying the lower band in a strong downtrend where
  closes keep printing below the band (bands "walk down"). The exit at the
  middle band caps upside in runaway rallies (the move from band to band is
  the whole trade).
- Complements the existing `bb_breakout` (which trades the *continuation*
  interpretation); this trades the *reversion* interpretation of the same
  bands.
