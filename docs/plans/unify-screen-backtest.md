# Plan: one definition for screen and backtest

Status: in progress.
Branch: `feat/unify-screen-backtest`, cut from `feat/unify-score-layer` (PR #134).

## Goal

A strategy is defined in one file.
That one definition drives the screen, the rolling backtest, and the optimizer.

## Success criteria

All must pass before this work is done.

1. A CI test driven off `screener.strategies.spec.registry.items()`, so it cannot be forgotten, asserts for every registered strategy: on a golden fixture panel and a fixed as-of date, the screen's candidate set equals the rolling engine's candidate rows for that date.
2. A second CI test asserts the TradingView prefilter never drops a name the bar rules would have kept, on those fixtures.
3. Exactly one implementation exists for each of `breakout`, `momentum_12_1`, and the Minervini trend template rule expression.
4. `just ci` passes.
5. `uv run python scripts/backtest_delta.py --compare <baseline>` reports zero movement, except where a stage note below says otherwise and names the reason.

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
| D18 | Staged. Nothing observable changes until stage 6. |
| D19 | Keep the word `strategy`. Add `StrategyProfile`. Rewrite the `CONTEXT.md` collision entries. Add an ADR. |
| D20 | Port the `RSI`, `relative_volume_10d_calc` and `Perf.Y` scorers to bar recipes. |
| D21 | The criterion's TradingView filters survive as a per-strategy declared prefilter. |

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
CLI flags override it, and every override is printed.
Without this the two paths still drift, by config instead of by code.

### Two universe modes, recorded in the run

The default keeps the TradingView query as a field cutter, which is what `scoring/bar_scores.py` already assumes.
`--universe` resolves from `universes.py` and applies no prefilter.
The mode is written into the `runs.criteria` label so a diff never crosses modes.

## Stages

Pin a baseline before stage 1 with `uv run python scripts/backtest_delta.py --out /tmp/unify-baseline.json`.
Run `--compare /tmp/unify-baseline.json` at the end of every stage.

- **Stage 0.** Branch off `feat/unify-score-layer`. Done.
- **Stage 1.** `StrategyProfile` on the spec, derived from `SignalPanelInputs`. Every existing plugin gets its current effective defaults. No behaviour change.
- **Stage 2.** `signal_panel` gains the one-day entry point. Nothing calls it yet. No behaviour change.
- **Stage 3.** Callables convert. Indicator-registry columns are precomputed into bars and referenced as plain series names in the expression. Non-convertible callables stay and are rejected at screen time with a clear message, in the style of `ensure_backtestable_scorer`. No backtest behaviour change.
- **Stage 4.** `BarFeatures` extended with point-in-time fundamentals. The `RSI`, `relative_volume_10d_calc` and `Perf.Y` scorers port to bar recipes. `market_cap_basic` stays snapshot, which is acceptable because the default path still queries TradingView. No behaviour change until stage 6.
- **Stage 5.** Reconciliation tests land, against the not-yet-default exact path.
- **Stage 6.** The flip, in one reviewable commit. Criterion names become aliases onto the unified registry. Criterion filter functions become `tv_prefilter` declarations. `--universe` mode goes live. `runs.criteria` labels change. This is the only stage that moves numbers.
- **Stage 7.** `CONTEXT.md` collision entries rewritten. ADR added under `docs/adr/` recording that TradingView is no longer a rule or fundamental source, and why.

## Accepted residuals

These are known and are not fixed by this work.

- The Minervini trend template has three implementations and this work removes one.
  `criteria/plugins/technical.py` is the lossy TradingView version and becomes a prefilter.
  `MINERVINI_ENTRY_EXPR` is the Pine version and becomes canonical.
  `minervini.py:evaluate_symbol` is a hand-written Python re-implementation with all ten checks inline, and D11 freezes it.
- `historical.py` keeps its own candidate definition, so `screener backtest --as-of` answers a different question than `screener screen` for the same strategy and day. D12 accepts this. It is scheduled for deletion, not for a fence.
- The default path still has a field cut that TradingView does not declare. The stage 5 prefilter test bounds it on fixtures only, not live.
