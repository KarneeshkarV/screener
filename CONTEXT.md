# Domain glossary

The vocabulary this codebase actually uses, recovered from the code.
Terms here are the names to use in commit messages, docstrings, and design discussion.

Where a word currently means more than one thing, that is recorded as a collision rather than hidden.
Collisions are defects in the shared language, and each one is a candidate for renaming.

## Core terms

**run**
One execution of a simulation over one config.
`_RunCaches` is per-run memoisation; `PreparedRollingBacktest` is the reusable half of a run, guarded by a config fingerprint.

**slot**
A capital compartment, not a position.
`slot_capital = initial_capital / slot_count`.
Slot state is keyed by an integer `slot_id`, and slots are not compounded.

**candidate**
A ticker whose entry signal fired and which passed the entry filters on a given day.
Produced per day by the rolling engine's candidate matrices.

**criterion**
A name passed to `screen --criteria`.
Three of them - `breakout`, `mark_minervini`, `momentum_12_1` - are aliases onto the strategy of the same name, so the screen judges them with that strategy's bar rule.
The rest are TradingView filter sets, judged by the vendor.

**profile**
`StrategyProfile`: the eligibility gates a strategy declares next to its rule, mirroring `SignalPanelInputs` field for field so a gate cannot be omitted.
Screen and backtest both load it, which is what makes the two paths agree.

**prefilter**
`StrategyProfile.tv_prefilter`: the name of a criterion whose TradingView filters cut the field before bars are downloaded.
It is an optimisation, not a gate.
It may only remove names the bar rule would have removed anyway, which `tests/correctness/test_screen_backtest_reconciliation.py` pins.
`--universe` is the exact path and applies none.

**role**
`active` or `reserve`.
Actives fill slots at `as_of`; reserves wait in a queue sized `top * reserve_multiple`.
Only the historical engine uses reserves; the rolling engine emits `active` unconditionally.

**universe**
The tradeable name set for a run.
Kinds: static (`--tickers`, `--universe-file`), named index (`--universe sp500`), point-in-time snapshot (membership windows with effective dates), and dynamic (top-N by lagged average dollar volume, rebalanced on a schedule).

**regime**
Benchmark trend classification used to gate entries.
Only the rolling engine gates on it.

**blackout**
Suppression of entries within N days of a known earnings date.

**pyramiding**
Concurrent lots on one ticker, keyed by `(ticker, open_seq)`.

**sleeve / tier**
Partial-exit vocabulary.
A partial exit sells a fraction at a target and the remainder is the remaining sleeve.

**fill**
The executed price, as distinct from the signal price.
`gap_fills` means an adverse open past the trigger is honoured rather than assumed away.

**rank_score**
A cross-sectional factor score written into bars by a strategy's `prepare_bars`.
Its presence flips candidate ranking from dollar volume to factor order.

**master calendar**
The sorted union of all tickers' bar timestamps in the window.
It is the equity curve's index.

## Symbol vocabularies

Four symbol spellings coexist and are not interchangeable.

- TradingView: `NSE:RELIANCE`, `M_M`. The canonical key a name is addressed by, including in the bar cache. It is not where the bars come from.
- yfinance: `RELIANCE.NS`, `M&M.NS`. Keys the price panel.
- NSE bhavcopy: `M&M`.
- FMP: `RELIANCE.NS` for India, bare symbol for US.

`tv_to_yf` and `tv_to_nse` translate three of them.

## Known collisions

These words currently mean more than one thing.
Prefer the qualified name in new code.

**panel** means four things.
1. `price_panel`: a dict of yfinance symbol to OHLCV frame.
2. `PanelBars`: column-per-ticker wide frames sharing one index, used by the expression evaluator.
3. An on-disk accumulating snapshot parquet.
4. An options chain panel.

**run** means two things.
A backtest execution, and a persisted screen result row addressed by `run_id` and replayable via `--from-run`.

**trade** has one neutral lifecycle base in `screener.ledger.Trade`.
`EquityLedgerTrade`, `ResearchTrade`, `ExecutedEventTrade`, and `OptionPositionTrade` extend it with their accounting, research, event, and multi-leg fields.
The extensions keep their distinct persisted schemas and return units.

**ExitReason** is one neutral literal set in `screener.ledger`.
It preserves every existing equity and options serialized value.
`rank` marks a rolling-engine position closed because it left the top `rank_universe_size` of the prior completed bar's candidate ranking (`--rank-exit`).

**tearsheet** means two things with zero shared code.
An HTML equity and trade report for a backtest result, and the factor IC and quantile report.

**provider** spans three unrelated vocabularies.
A circuit-breaker name, a price provider selector, and a fundamentals provider selector.

**strategy** spans five populations.
Callable specs, expression specs, earnings scorers, vbt signal builders, and a vbt config literal.
The first two have stopped being a collision inside the screen/backtest path: an expression spec is the one definition that drives the screen, the rolling backtest and the optimizer.
Four callable specs remain, whose trade generation is not a per-bar boolean, and the screen rejects them by kind with a clear message.

**screen** means the `screen` command, the standalone feature commands that are not that command, an operator labeller, and an options criterion.
Within the `screen` command the word has one meaning again: for an aliased criterion it asks the same question the rolling backtest asks, on the same day, off the same bars.
The standalone commands (`garp`, `conviction`, `rs_breakout`, `mark-minervini`) still carry their own hand-written definitions and are outside that guarantee.

**sharpe** is computed two incomparable ways.
`equity_curve_sharpe` annualises a daily equity curve.
`trade_return_sharpe_by_holding_period` annualises per-trade returns by average holding period.
The two numbers are not comparable.

## Conventions

Timestamps are canonical naive UTC across providers for intraday data, and tz-naive for daily.
Intraday caches are namespaced per interval so daily parquet files are never polluted.

A non-finite close must never pass an entry filter.
This is stated explicitly because the two filter implementations historically disagreed on `NaN`.
