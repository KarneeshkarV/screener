# Turn-of-Month Effect — Ariel (1987)

**Strategy key:** `turn_of_month`

## Research source

**Paper:** Ariel, *"A Monthly Effect in Stock Returns"*, Journal of Financial Economics 18(1), 1987.

- DOI: https://doi.org/10.1016/0304-405X(87)90066-3
- Elsevier page: https://www.sciencedirect.com/science/article/abs/pii/0304405X87900663

**India evidence:** *"Semi-monthly effect in stock returns: new evidence from Bombay Stock Exchange"* (2017). https://doi.org/10.21511/imfi.14(3-1).2017.01

Ariel found that virtually the entire monthly equity premium is earned in the ~4 days around the turn of the month (last trading day of month through the first ~3 of the next); Lakonishok & Smidt (1988, *Are Seasonal Anomalies Real?*) confirmed the same concentration on 90 years of Dow data. The TOM effect is one of the most robust calendar anomalies and has been documented internationally, including on the BSE.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `day_of_month >= 28 or day_of_month <= 3` — into the TOM window |
| Exit | `day_of_month >= 4 and day_of_month <= 27` — out of the window |

## Implementation notes

- `prepare_bars` adds `day_of_month` from the bar index; no indicator math, no lookback required.
- Suggested sizing: this is an index-timing overlay — 20 slots; exposure ≈ 1/3 of trading days.

## Expected behaviour

- Low absolute return (flat ~2/3 of the month) but high per-exposure return if the effect holds.
- The benchmark comparison is unfair unless scaled by exposure — evaluate on Sharpe/Calmar and per-exposure return, not raw CAGR.
- Expected to be the "boring but robust" strategy: small, consistent wins month after month.
