# Plan: one definition for screen and backtest

Status: complete. Stages 0-7 done.
Branch: `feat/unify-screen-backtest`, cut from `feat/unify-score-layer` (PR #134).

## Goal

A strategy is defined in one file.
That one definition drives the screen, the rolling backtest, and the optimizer.

## Success criteria

All must pass before this work is done.

1. **Met.** A CI test driven off `screener.strategies.spec.registry.items()`, so it cannot be forgotten, asserts for every registered strategy: on a golden fixture panel and a fixed as-of date, the screen's candidate set equals the rolling engine's candidate rows for that date.
2. **Met.** A second CI test asserts the TradingView prefilter never drops a name the bar rules would have kept, on those fixtures. It found one violation, in `breakout`, which is fixed.
3. **Met, to the limit of the residuals below.** Exactly one implementation exists for each of `breakout`, `momentum_12_1`, and the Minervini trend template rule expression. `minervini.py:evaluate_symbol` and `historical.py` keep their own, frozen by D11 and D12.
4. **Met.** `just ci` passes.
5. **Forward guard only.** No baseline was pinned before stage 1, so the zero-movement check could not be run across this work. One is now pinned at `scripts/backtest_delta_baseline.json.gz`, cut after the flip, and later work compares against it:

       uv run python scripts/backtest_delta.py --compare scripts/backtest_delta_baseline.json.gz

## Settled decisions

| Ref | Decision |
|---|---|
| D1 | Goal is one-file-gets-everything, plus a proven screen-equals-backtest guarantee. |
| D2 | Local bars are canonical. TradingView stops being a rule source. |
| D3 | Fundamentals ride in one registry with price factors, via an extended `BarFeatures`. |
| D4 | A 60 second uncached screen is acceptable. |
| D5 | CLI flags and the `history.db` schema are frozen. Criterion names become aliases. |
| D6 | Equality is asserted at the candidate layer. Ranked top-N is a derived view. |
| D7 | The rolling engine is the reference. |
| D8 | Fundamentals come from the point-in-time fetchers in `backtester/fundamentals.py`, never from TradingView. |
| D9 | Two universe modes. The TradingView prefilter is the default. `--universe` is the exact path. |
| D10 | Callables convert to expressions over precomputed indicator columns. The Pine grammar does not grow. |
| D11 | The `garp`, `conviction`, `rs_breakout` and `minervini` commands are frozen and out of scope. Do not add logic to them. |
| D12 | `historical.py` is untouched and gets no guard test. It is deleted in later work. |
| D13 | The plugin declares a full `StrategyProfile`, not just a rule. |
| D14 | `entry` stays required and non-empty. Fundamental-only rules register `entry="true"`. |
| D15 | Golden-fixture tests gate CI. Live triples opt in behind `SCREENER_LIVE_TESTS=1` and the `network` marker. |
| D16 | Built on PR #134's `screener/factors/` layer. This branch lands together with it. |
| D17 | At cutover the `runs.criteria` label changes, so diffs never cross the semantics change. |
| D18 | Staged. Nothing observable changes until stage 6. **Breached by stage 3**: converting a callable to an expression changed which names some strategies select, before the flip. The staging held for the gates and the labels, not for the rules. |
| D19 | Keep the word `strategy`. Add `StrategyProfile`. Rewrite the `CONTEXT.md` collision entries. Add an ADR. |
| D20 | Port the `RSI`, `relative_volume_10d_calc` and `Perf.Y` scorers to bar recipes. |
| D21 | The criterion's TradingView filters survive as a per-strategy declared prefilter. |
| D22 | One resolver owns the gates. `resolve_strategy_profile(spec, overrides, market=...)` is the only place a `StrategyProfile` becomes the gates a run uses, for the screen and the rolling backtest alike. |
| D23 | A flag left at its option default is not given. Only a flag the user typed overrides the declared profile. An explicit `0` means "disable this gate", which is not the same as saying nothing. |
| D24 | The market floor applies on both paths. An unset `min_price` or `min_avg_dollar_volume` falls to the venue minimum from `markets.py`. The resolver uses `get_market`, so an unknown market name raises rather than quietly producing an unfloored screen. |
| D25 | The shared flags live in `screener/gate_options.py`, a module that imports neither the screen nor the backtester. Both commands build their options from it, so a gate cannot be added to one command and forgotten on the other. `signal_panel.py` asserts the same partition at import time on the data side. |
| D26 | `--min-score` is a candidate-layer gate, not a presentation filter. The percentile is taken over the day's eligible field, before `exclude` and before `limit`, so the score means the same thing in a screen and in a backtest day. |
| D27 | `--earnings-buffer` stays screen-only. It is a presentation-stage filter on result rows, a different stage from `--earnings-blackout`, which suppresses entry signals. Sharing the name would hide that. |
| D28 | The screen's run label carries a settings fingerprint (`<criteria>@<universe>#<8 hex>`). The gates are part of the question, so history never diffs a run made with `--min-price 50` against one made without it. |
| D29 | The screen's panel lookback is `max(program.lookback, avg_dollar_volume_window)`. A 15-bar window judged a 20-bar ADV mean on too few bars, which is a screen-only defect this work fixes. |
| D30 | Bar-path flags are refused, never ignored. A gate flag, `--interval`, or `--max-universe` on a criterion that names TradingView filters only raises `UnscreenableStrategyError`. A non-`1d` interval with a fundamental fetcher or an earnings blackout raises `IntervalNotScreenableError`. |

## Target architecture

### One candidate layer, three callers

`screener/backtester/signal_panel.py` already owns entry and exit parsing, filter signals, the regime gate, the earnings blackout, sector neutralisation, and the candidate matrices.
It gains a one-day entry point.
It then has three callers: the rolling engine, the screen, and later the historical engine.

Today only `rolling_simulation.py` calls it.
`historical.py` builds candidates its own way through `core._precompute_entry_signals` and `select_candidates`.
That stays untouched, per D12.

### One score layer, extended

PR #134's `screener/factors/` stays as built: a registry of causal, bars-only recipes consumed by two thin adapters, `strategies/factor_adapter.py` writing `rank_score` and `scoring/bar_scores.py` writing `setup_score`, both calling `score_bars`.

`BarFeatures` grows point-in-time fundamental columns.
Those columns enter only through `merge_fundamentals_into_bars` with `fundamental_filing_lag_days` applied.
Direct assignment of a fundamental column onto bars is a bug.
`tests/correctness/test_lookahead_blindness.py` gets a case per fundamental column, because that join is the one place silent lookahead can enter.

### The plugin declares a profile

`StrategyProfile` mirrors `SignalPanelInputs` field for field, so it cannot omit a gate.
Fields: `entry`, `exit`, `min_price`, `min_avg_dollar_volume`, `avg_dollar_volume_window`, `regime_filter`, `earnings_blackout_days`, `sector_neutral`, the fundamental stage, and an optional `tv_prefilter`.
Screen and backtest both load it.
Without this the two paths still drift, by config instead of by code.

Not built: per-gate CLI overrides.
No flag on `screen` or `backtest` maps to a profile gate today, and D5 freezes the CLI surface, so adding one would contradict a settled decision to serve a sentence.
The profile is the single place a gate is set.

### Two universe modes, recorded in the run

The default keeps the TradingView query as a field cutter, which is what `scoring/bar_scores.py` already assumes.
`--universe` resolves from `universes.py` and applies no prefilter.
The mode is written into the `runs.criteria` label so a diff never crosses modes.

## Stages

Pin a baseline before stage 1 with `uv run python scripts/backtest_delta.py --out /tmp/unify-baseline.json`.
Run `--compare /tmp/unify-baseline.json` at the end of every stage.

- **Stage 0.** Branch off `feat/unify-score-layer`. **Done.**
- **Stage 1. Done.** `StrategyProfile` on the spec, derived from `SignalPanelInputs`. Every existing plugin gets its current effective defaults. No behaviour change.
- **Stage 2. Done.** `signal_panel` gains the one-day entry point. Nothing calls it yet. No behaviour change.
- **Stage 3. Done.** Callables convert. Indicator-registry columns are precomputed into bars and referenced as plain series names in the expression. Non-convertible callables stay and are rejected at screen time with a clear message, in the style of `ensure_backtestable_scorer`. No backtest behaviour change.
- **Stage 4. Done.** `BarFeatures` extended with point-in-time fundamentals. The `RSI`, `relative_volume_10d_calc` and `Perf.Y` scorers port to bar recipes. `market_cap_basic` stays snapshot, which is acceptable because the default path still queries TradingView. No behaviour change until stage 6.
- **Stage 5. Done.** Reconciliation tests land, against the not-yet-default exact path.
- **Stage 6. Done.** The flip, in one reviewable commit. Criterion names become aliases onto the unified registry. Criterion filter functions become `tv_prefilter` declarations. `--universe` mode goes live. `runs.criteria` labels change. This is the only stage that moves numbers.
- **Stage 7. Done.** `CONTEXT.md` collision entries rewritten. ADR added under `docs/adr/` recording that TradingView is no longer a rule or fundamental source, and why.

## Stage 3 outcome

14 of 18 callable strategies converted. Bucket C (needs a new indicator) was empty:
every indicator required already existed in `screener/indicators/`.

Four remain callable-only because their trade generation is not a per-bar boolean:
`heikin_ashi` (recursive `ha_open` plus a `cumsum` cap), `shooting_star`,
`bb_pattern` and `rsi_pattern` (nested backward searches carrying state).
Stage 6 rejects these at screen time with a clear message.

Two mechanisms made this work, both of which serve the plan directly.
`screener/strategies/registry.py` now synthesises the pine_runner's callable
from an expression, so converting a name no longer drops it out of
`STRATEGIES`, which the registry tests pin as a breaking change.
`bar_columns` on the spec declares pure bar-local derived columns, so a new
indicator becomes a column rather than a new function in the Pine parser.

## Defect found, not fixed here

`screener/strategies/plugins/shooting_star.py` computes
`mean_body = np.mean(np.abs(op - cl))` over the entire series, including future
bars, then uses it as a per-bar threshold. Every historical signal it produces
is contaminated by data from after that bar. It is bucket D so this work does
not touch it, but its results should not be trusted.

## Stage 5 outcome

The equality sweep covers 39 expression strategies, 20 of which produce a
non-empty candidate set on the fixture, all agreeing exactly with the rolling
engine on ticker order, rank, role, signal index and the resolved as-of bar.
Four callable-only strategies are excluded by kind, pinned by a test so the
exclusion cannot widen by name.

The prefilter sweep found one real violation. `breakout` fronted
`close >= highest(close, 252) * 0.9` with `price_52_week_high`, the 52-week
extreme of *highs*, which is never below the extreme of closes. The vendor
threshold therefore sat at or above the rule's and dropped names inside the
band - 8 name-days on the fixture. `breakout` now declares only the volume
leg, as the `above_avg_volume` criterion. The entry expression is untouched,
so no backtest number moves; the default screen widens, which is the sound
direction.

## Accepted residuals

These are known and are not fixed by this work.

- The Minervini trend template has three implementations and this work removes one.
  `criteria/plugins/technical.py` is the lossy TradingView version and becomes a prefilter.
  `MINERVINI_ENTRY_EXPR` is the Pine version and becomes canonical.
  `minervini.py:evaluate_symbol` is a hand-written Python re-implementation with all ten checks inline, and D11 freezes it.
- `historical.py` keeps its own candidate definition, so `screener backtest --as-of` answers a different question than `screener screen` for the same strategy and day. D12 accepts this. It is scheduled for deletion, not for a fence.
- The default path still has a field cut that TradingView does not declare. The stage 5 prefilter test bounds it on fixtures only, not live.

## Gate parity outcome

`resolve_screen_gates` and `resolve_rolling_gates` both call
`resolve_strategy_profile`, and `tests/correctness/test_screen_backtest_reconciliation.py`
asserts they agree field for field over the whole strategy registry against both
markets, with and without typed overrides. A second test drives both commands'
`--help` from `GATE_OPTION_NAMES`, so a new shared flag that reaches only one
command fails CI rather than shipping.

Checked end to end against live nifty50 bars:

```
screener screen -m india -c breakout --universe nifty50 --min-price 1000 --min-score 50
screener backtest-rolling -m india --universe nifty50 --strategy breakout --candidates --min-price 1000 --min-score 50
```

Both drop the sub-₹1000 names, both recompute the percentile over what survived
the price gate, and both keep the top half. The residual difference between the
two lists is the signal bar: the screen answers for the newest bar, the backtest
for the last bar in its window.

## Known gaps, deliberate

- `backtest-historical` still ignores strategy profiles and keeps its own
  candidate definition. D12 stands: it is scheduled for deletion, not for a fence.
- Freshness defaults differ by design. The screen caches TradingView for 15m;
  the backtest reads bars for a closed window. `--refresh` is on both.
- `--max-universe` is run-scoped, not a gate. It caps the field before bars are
  fetched, so it is not part of the profile and not part of the fingerprint.
