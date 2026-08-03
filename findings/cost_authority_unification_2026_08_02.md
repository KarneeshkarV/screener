# Cost authority unification: what moved and why

Date: 2026-08-02.
Scope: the change that made `Portfolio` the single cost and budget authority, unified the
force-close window, fixed the NaN entry-filter divergence, unified warmup, and made
`commission_bps` mean per-fill everywhere.

## Summary

Backtest numbers moved in two places, both deliberately.

1. Equity backtests move **only when a size-dependent slippage model is configured**.
2. PEAD results move for every run, because `commission_bps` changed meaning.

Everything else is unchanged, including `tests/test_rolling_characterization.py`, whose expected
ledger and final equity are untouched.

## 1. Equity engines: only the size-dependent path moves

Measured with `scripts/backtest_delta.py` across a 48-cell matrix
(2 engines x 3 cost models x 2 sizing rules x 2 intervals x 2 slippage models), comparing the
commit before this change against the commit after.

| Slippage model | Cells | Cells that moved |
|---|---|---|
| `fixed_bps` | 24 | **0** |
| `volume_impact` | 24 | **24** |

That split is the whole point, and it is the expected signature rather than a surprise.

`FixedBpsSlippage` ignores order size, so the entry budget and the configured cost model never reach
the fill price and nothing can change. `VolumeImpactSlippage` is size-dependent, so both do reach it.
Before this change, `fills.py` sized the order using a hardcoded flat commission while `portfolio.py`
charged the configured schedule. With `--cost-model india` or `us_vested` the two disagreed.

Magnitudes are small but systematic, for example
`historical|india|equal_slot|1d|volume_impact`:

```
metrics.alpha_annual:      -0.014222093 -> -0.014218198  (+3.9e-06)
metrics.beta:               0.111168927 ->  0.111174538  (+5.6e-06)
metrics.avg_trade_return:  -0.001048584 -> -0.001048513  (+7.1e-08)
```

Note the `flat` cells moved too. That is not the cost-model fix; it is the sizing convention. Entry
shares are now computed once from the pre-slippage reference and carried into `Portfolio.open`
rather than recomputed there from the impacted fill. See `docs/adr/0001-pre-impact-entry-sizing.md`.

**A caveat that is worth recording.** The original matrix had no size-dependent slippage cell at all,
so it reported 24 of 24 cells identical and would have shown this fix as a no-op. The
`volume_impact` dimension was added specifically because the instrument could not otherwise see the
defect it was built to measure. A harness that only exercises size-independent slippage cannot
detect a cost-authority regression.

## 2. PEAD: commission_bps changed meaning

`commission_bps` is now per-fill everywhere. It previously meant a single round-trip total in PEAD,
which reached the number through `net_round_trip_return`, while the equity engines charged it per
fill and the earnings engine halved it to compensate.

At 10 bps:

| Case | Before | After |
|---|---|---|
| fixed hold | 4.3410% | 4.2410% |
| dynamic hold | 9.7901% | 9.6901% |

The 10 bps now applies on entry and on exit rather than once per round trip, so each result loses
roughly 10 bps more. Any PEAD figure quoted from an earlier run was understating costs relative to
the equity engines.

See `docs/adr/0002-commission-bps-is-per-fill.md`.

## 3. Force-close: a class of silently missing trades

The two engines force-closed open slots with the same body but different windows. The historical
engine used a strictly-greater, unbounded window, so a position entering on the last bar of its
frame produced an empty tail, was skipped, and never reached `portfolio.closed_trades()`.

Because the historical result is built from `closed_trades()`, such a position was **silently erased
from both the trade ledger and the equity curve**. It did not appear as a zero-return trade; it did
not appear at all.

Both engines now use the shared inclusive window. `_TERMINAL_EOD_TRIM` in the cross-engine
reconciliation suite remains 0.

## 4. NaN entry filters

The scalar and panel filters agreed for every ordered value including the boundary, and disagreed on
`NaN`: `NaN < min_price` is false so the scalar path passed a NaN close, while `NaN >= min_price` is
false so the panel path failed it. The ADV leg did not rescue it, because `.mean()` skips NaN and
still returns a finite number.

The policy is now explicit in both paths: a non-finite close never passes the price filter. A
non-finite `min_price` is also rejected. Neither case had test coverage before.
