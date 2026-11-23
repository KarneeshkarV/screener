# ADX Trend Filter — J. Welles Wilder

**Strategy key:** `adx_trend`

## Research source

**Book:** Wilder, *"New Concepts in Technical Trading Systems"*, 1978, Trend Research.

- https://www.amazon.com/New-Concepts-Technical-Trading-Systems/dp/0894590278
- Modern reference: https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx

Wilder's Directional Movement system is the original trend-strength framework. +DI/−DI measure the relative power of up- vs down-moves; ADX smooths their difference into a 0-100 trend-strength gauge. Wilder's guidance: only trade when ADX is high (trend present) and in the direction of the dominant DI.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `adx_14 > 25 and di_plus > di_minus` — strong up-trend confirmed |
| Exit | `di_plus < di_minus` — direction flips |

with Wilder RMA smoothing (α = 1/14) on true range and directional movement, `+DI = 100·rma(+DM)/rma(TR)`, `−DI = 100·rma(−DM)/rma(TR)`, `ADX = rma(100·|+DI−−DI|/(+DI+−DI), 14)`.

## Implementation notes

- Not expressible in the engine's Pine (Wilder iterative smoothing), so `prepare_bars` computes `adx_14`/`di_plus`/`di_minus` using the repo's `rma` indicator.
- `required_lookback = 29` (2×14 + 1).
- Suggested sizing: trend following — 10 slots, longer holds.

## Expected behaviour

- Only trades when a trend is actually strong (ADX > 25) — filters chop that breaks EMA rules.
- Tends to enter late (ADX confirms after the move starts) and exit on direction flip — expect medium holding periods, decent hit rate.
- Complements the repo's trend rules (`ema_trend`, `supertrend`) by adding an explicit trend-strength gate.
