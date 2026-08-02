"""Historical backtest orchestration."""

from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd

from screener.backtester.core import (
    _RunCaches,
    _SlotState,
    _bar_label,
    _benchmark_series_from_panel,
    _eligible_reserve_signal_idx,
    _make_slot_state,
    _passes_entry_filters,
    prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.costs import cost_model_from_config
from screener.backtester.data import PriceFetcher
from screener.backtester.day_loop import (
    DayLoop,
    FreedSlot,
    _force_close_open_slots,
    run_day_loop,
)
from screener.backtester.fills import FillModel
from screener.backtester.metrics import (
    compute_cost_metrics,
    compute_metrics,
    compute_regime_metrics,
    no_trades_result,
    periods_per_year_for_interval,
)
from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
)
from screener.backtester.historical_cli import backtest_historical
from screener.backtester.pine import PineError, parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.sizing import entry_budget_for
from screener.backtester.warmup import _warmup_days_for_interval
from screener.options.backtest import merge_referenced_options


def select_candidates(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    as_of: pd.Timestamp,
    top_n: int,
    lookback_required: int,
    cfg: BacktestConfig | None = None,
    caches: _RunCaches | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate entry AST at ``as_of`` for each ticker and rank survivors.

    The entry AST is evaluated once over each ticker's full frame and indexed
    positionally at the ``as_of`` bar (every Pine op is causal, so this matches
    the truncated-history evaluation — the same discipline the rolling engine's
    ``_precompute_entry_signals`` relies on). Pass ``caches`` to share those
    evaluations with the reserve-rotation simulator.
    """
    rows = []
    warnings: list[str] = []
    filtered_count = 0
    reserve_multiple = cfg.reserve_multiple if cfg is not None else 1
    pool_limit = max(top_n * max(reserve_multiple, 1), top_n)
    if caches is None:
        caches = _RunCaches()
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            warnings.append(f"no data: {ticker}")
            continue
        # bars.index is a sorted DatetimeIndex; searchsorted mirrors the
        # previous ``bars.index <= as_of`` mask without copying the history.
        pos = int(bars.index.searchsorted(as_of, side="right")) - 1
        if pos + 1 < lookback_required + 1:
            warnings.append(f"insufficient lookback ({pos + 1} bars): {ticker}")
            continue
        if cfg is not None:
            passes, _reason = _passes_entry_filters(bars, as_of, cfg)
            if not passes:
                filtered_count += 1
                continue
        signal = caches.entry_signal(ticker, bars, entry_ast)
        if isinstance(signal, PineError):
            warnings.append(f"entry eval failed: {ticker}: {signal}")
            continue
        if signal.empty:  # pragma: no cover - bars are non-empty here
            continue
        last = signal.iloc[pos]
        if pd.isna(last) or not bool(last):
            continue
        last_bar = bars.iloc[pos]
        close = float(last_bar["close"])
        volume = float(last_bar["volume"])
        rows.append(
            {
                "ticker": ticker,
                "as_of_close": close,
                "as_of_volume": volume,
                "as_of_dollar_vol": close * volume,
            }
        )
    if filtered_count:
        warnings.append(f"filtered {filtered_count} tickers on price/liquidity filters")
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "as_of_close",
                "as_of_volume",
                "as_of_dollar_vol",
                "rank",
                "role",
            ]
        ), warnings
    df = (
        pd.DataFrame(rows)
        .sort_values("as_of_dollar_vol", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    df = df.head(pool_limit).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["role"] = ["active" if i < top_n else "reserve" for i in range(len(df))]
    return df, warnings


class _ReserveRotationSource:
    """Historical :class:`~screener.backtester.day_loop.CandidateSource` adapter.

    Owns the historical fill half: a fixed reserve queue selected once at
    ``as_of`` plus optional same-ticker re-entry into a just-vacated slot.
    ``before_exits`` promotes any pending re-entries whose entry signal has
    re-fired; ``after_exits`` schedules re-entries for the slots that freed this
    day and rotates the reserve queue into the rest. All state (slot maps,
    portfolio, queues) is shared by reference with the driver.
    """

    def __init__(
        self,
        *,
        portfolio: Portfolio,
        cfg: BacktestConfig,
        entry_ast,
        exit_ast,
        lookback: int,
        fill_model: FillModel,
        bars_by_tv: dict[str, pd.DataFrame],
        slot_states: dict[int, _SlotState | None],
        slot_bars: dict[int, pd.DataFrame],
        reentries_left: dict[int, int],
        pending_reentry: dict[int, str],
        reserve_queue: deque[dict],
        taken: set[str],
        warnings: list[str],
        caches: _RunCaches,
    ) -> None:
        self.portfolio = portfolio
        self.cfg = cfg
        self.entry_ast = entry_ast
        self.exit_ast = exit_ast
        self.lookback = lookback
        self.fill_model = fill_model
        self.bars_by_tv = bars_by_tv
        self.slot_states = slot_states
        self.slot_bars = slot_bars
        self.reentries_left = reentries_left
        self.pending_reentry = pending_reentry
        self.reserve_queue = reserve_queue
        self.taken = taken
        self.warnings = warnings
        self.caches = caches

    def before_exits(self, day: pd.Timestamp) -> None:
        cfg = self.cfg
        portfolio = self.portfolio
        if self.pending_reentry:
            for slot_id, ticker in list(self.pending_reentry.items()):
                slot_frame = self.slot_bars.get(slot_id)
                if (  # pragma: no cover - freed slots always have slot_bars set
                    slot_frame is None or slot_frame.empty
                ):
                    del self.pending_reentry[slot_id]
                    continue
                reentry_signal_idx = _eligible_reserve_signal_idx(
                    slot_frame,
                    day,
                    cfg,
                    self.entry_ast,
                    self.lookback,
                    ticker=ticker,
                    caches=self.caches,
                )
                if reentry_signal_idx is None:
                    continue
                new_rank = portfolio._ranks.get(ticker, 0)
                entry_budget = entry_budget_for(
                    cfg, portfolio, slot_frame, reentry_signal_idx
                )
                state, warn = _make_slot_state(
                    ticker,
                    slot_frame,
                    reentry_signal_idx,
                    cfg,
                    self.exit_ast,
                    new_rank,
                    self.fill_model,
                    caches=self.caches,
                    entry_budget=entry_budget,
                )
                if state is None:
                    if warn:
                        self.warnings.append(f"{ticker} re-entry: {warn}")
                    del self.pending_reentry[slot_id]
                    continue
                portfolio.assign(ticker, new_rank, _bar_label(day, cfg))
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    budget=entry_budget,
                    shares=state.entry_shares,
                )
                self.slot_states[slot_id] = state
                del self.pending_reentry[slot_id]

    def after_exits(self, day: pd.Timestamp, freed_slots: list[FreedSlot]) -> None:
        cfg = self.cfg
        portfolio = self.portfolio
        freed: list[int] = []
        for freed_slot in freed_slots:
            slot_id = freed_slot.slot_id
            freed.append(slot_id)
            if cfg.allow_reentry and self.reentries_left.get(slot_id, 0) > 0:
                self.reentries_left[slot_id] -= 1
                self.pending_reentry[slot_id] = freed_slot.state.ticker

        if not cfg.reinvest or not freed:
            return

        for slot_id in freed:
            if slot_id in self.pending_reentry:
                continue
            while self.reserve_queue:
                reserve = self.reserve_queue.popleft()
                ticker = str(reserve["ticker"])
                if ticker in self.taken:
                    continue
                reserve_bars = self.bars_by_tv.get(ticker, pd.DataFrame())
                if reserve_bars is None or reserve_bars.empty:
                    continue
                reserve_signal_idx = _eligible_reserve_signal_idx(
                    reserve_bars,
                    day,
                    cfg,
                    self.entry_ast,
                    self.lookback,
                    ticker=ticker,
                    caches=self.caches,
                )
                if reserve_signal_idx is None:
                    continue
                entry_budget = entry_budget_for(
                    cfg, portfolio, reserve_bars, reserve_signal_idx
                )
                state, warn = _make_slot_state(
                    ticker,
                    reserve_bars,
                    reserve_signal_idx,
                    cfg,
                    self.exit_ast,
                    int(reserve["rank"]),
                    self.fill_model,
                    caches=self.caches,
                    entry_budget=entry_budget,
                )
                if state is None:
                    if warn:
                        self.warnings.append(f"{ticker}: {warn}")
                    continue
                portfolio.assign(ticker, int(reserve["rank"]), _bar_label(day, cfg))
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    budget=entry_budget,
                    shares=state.entry_shares,
                )
                self.slot_states[slot_id] = state
                self.slot_bars[slot_id] = reserve_bars
                self.taken.add(ticker)
                break


def _run_event_driven_sim(
    *,
    portfolio: Portfolio,
    actives_df: pd.DataFrame,
    reserves_df: pd.DataFrame,
    bars_by_tv: dict[str, pd.DataFrame],
    as_of_ts: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    exit_ast,
    lookback: int,
    warnings: list[str],
    caches: _RunCaches | None = None,
) -> list[pd.Timestamp]:
    """Chronological event-driven simulator with optional reserve rotation.

    Returns ``master_dates``, the full cross-universe trading-day calendar the
    simulation ran over, so the caller can seed the equity-curve calendar with
    every real trading day in the horizon (not just trade-covered days).
    """
    if caches is None:
        caches = _RunCaches()
    fill_model = FillModel(cfg, cost_model=portfolio.cost_model)
    slot_states: dict[int, _SlotState | None] = {}
    slot_bars: dict[int, pd.DataFrame] = {}
    reentries_left: dict[int, int] = {}
    pending_reentry: dict[int, str] = {}

    for raw_slot_id, row in actives_df.iterrows():
        slot_id = int(str(raw_slot_id))
        ticker = row["ticker"]
        bars = bars_by_tv.get(ticker, pd.DataFrame())
        if bars is None or bars.empty:
            warnings.append(f"no data during sim: {ticker}")
            slot_states[slot_id] = None
            continue
        mask = bars.index <= as_of_ts
        if not mask.any():
            warnings.append(f"no history at as_of: {ticker}")
            slot_states[slot_id] = None
            continue
        signal_idx = int(np.where(mask)[0][-1])
        entry_budget = entry_budget_for(cfg, portfolio, bars, signal_idx)
        state, warn = _make_slot_state(
            ticker,
            bars,
            signal_idx,
            cfg,
            exit_ast,
            int(row["rank"]),
            fill_model,
            caches=caches,
            entry_budget=entry_budget,
        )
        if state is None:
            if warn:
                warnings.append(f"{ticker}: {warn}")
            slot_states[slot_id] = None
            continue
        portfolio.assign(ticker, int(row["rank"]), cfg.as_of)
        portfolio.open(
            ticker=ticker,
            entry_date=state.entry_date,
            entry_price=state.entry_fill,
            budget=entry_budget,
            shares=state.entry_shares,
        )
        slot_states[slot_id] = state
        slot_bars[slot_id] = bars
        reentries_left[slot_id] = cfg.max_reentries if cfg.allow_reentry else 0

    taken = {state.ticker for state in slot_states.values() if state is not None}
    reserve_queue: deque[dict] = deque(reserves_df.to_dict("records"))

    horizon_end = as_of_ts + pd.Timedelta(days=max(cfg.hold * 3 + 60, 90))
    day_set: set[pd.Timestamp] = set()
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        idx = bars.index
        day_set.update(idx[(idx > as_of_ts) & (idx <= horizon_end)])
    master_dates = sorted(day_set)

    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )

    source = _ReserveRotationSource(
        portfolio=portfolio,
        cfg=cfg,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        lookback=lookback,
        fill_model=fill_model,
        bars_by_tv=bars_by_tv,
        slot_states=slot_states,
        slot_bars=slot_bars,
        reentries_left=reentries_left,
        pending_reentry=pending_reentry,
        reserve_queue=reserve_queue,
        taken=taken,
        warnings=warnings,
        caches=caches,
    )
    run_day_loop(master_dates, day_loop, source)

    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=horizon_end,
        fill_model=fill_model,
    )

    return master_dates


def run_backtest(cfg: BacktestConfig, fetcher: PriceFetcher) -> BacktestResult:
    warnings: list[str] = []
    as_of_ts = pd.Timestamp(cfg.as_of)

    entry_ast = parse(cfg.entry_expr)
    exit_ast = parse(cfg.exit_expr) if cfg.exit_expr else None
    lookback = required_lookback(entry_ast)
    if exit_ast is not None:
        lookback = max(lookback, required_lookback(exit_ast))

    from screener.backtester.data import tv_to_yf

    tv_symbols, univ_warnings = _resolve_universe(cfg)
    warnings.extend(univ_warnings)

    yf_by_tv = {tv: tv_to_yf(tv, cfg.market) for tv in tv_symbols}
    yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [cfg.benchmark]))

    start = (
        as_of_ts - pd.Timedelta(days=_warmup_days_for_interval(lookback, cfg.interval))
    ).date()
    end = (as_of_ts + pd.Timedelta(days=cfg.hold * 2 + 30)).date()
    price_panel = fetcher.fetch(yf_symbols, start, end)

    if cfg.price_adjustment == "splits_only":
        from screener.backtester.data import (
            apply_splits_only_adjustment,
            warn_unadjustable_fmp_frames,
        )

        warn_unadjustable_fmp_frames(price_panel)
        price_panel = apply_splits_only_adjustment(price_panel)

    # See rolling_simulation: dict.get's default is eager, so the comprehension
    # form built one throwaway DataFrame per symbol.
    bars_by_tv = {}
    for tv in tv_symbols:
        panel_bars = price_panel.get(yf_by_tv[tv])
        bars_by_tv[tv] = pd.DataFrame() if panel_bars is None else panel_bars
    bars_by_tv, strategy_lookback = prepare_strategy_bars(
        cfg.strategy_name,
        bars_by_tv,
        price_panel,
        tv_symbols,
        start,
        end,
        fetcher,
        warnings,
        market=cfg.market,
        benchmark=cfg.benchmark,
    )
    lookback = max(lookback, strategy_lookback)
    bars_by_tv = merge_referenced_options(
        bars_by_tv,
        market=cfg.market,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        warnings=warnings,
    )

    caches = _RunCaches()
    selection, sel_warnings = select_candidates(
        bars_by_tv, entry_ast, as_of_ts, cfg.top, lookback, cfg, caches=caches
    )
    warnings.extend(sel_warnings)

    if selection.empty:
        return no_trades_result(
            cfg,
            calendar=pd.date_range(
                as_of_ts, as_of_ts + pd.Timedelta(days=cfg.hold * 2), freq="B"
            ),
            benchmark=_benchmark_series_from_panel(price_panel, cfg.benchmark),
            warnings=warnings,
            selection=selection,
        )

    actives_df = selection[selection["role"] == "active"].reset_index(drop=True)
    reserves_df = selection[selection["role"] == "reserve"].reset_index(drop=True)
    slot_count = max(cfg.top, len(actives_df))
    portfolio = Portfolio(
        cfg.initial_capital, slot_count, cost_model=cost_model_from_config(cfg)
    )

    master_dates = _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives_df,
        reserves_df=reserves_df,
        bars_by_tv=bars_by_tv,
        as_of_ts=as_of_ts,
        cfg=cfg,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        lookback=lookback,
        warnings=warnings,
        caches=caches,
    )

    trades = portfolio.closed_trades()
    # Seed the equity-curve calendar with the full cross-universe trading-day
    # horizon (``master_dates``), not just trade-covered days. Otherwise any day
    # where every slot is simultaneously flat is dropped from the index, and the
    # bar-count annualization in metrics.py silently compresses elapsed time,
    # wildly inflating cagr/sharpe/vol. Mirrors rolling.py's _assemble_results.
    date_set: set[pd.Timestamp] = set(master_dates)
    date_set.add(as_of_ts.normalize())
    for trade in trades:
        frame = bars_by_tv.get(trade.ticker)
        if frame is None or frame.empty:  # pragma: no cover - traded ticker has bars
            continue
        # frame.index is a sorted DatetimeIndex; slice by position instead of
        # building two boolean masks per trade.
        lo = frame.index.searchsorted(pd.Timestamp(trade.entry_date), side="left")
        hi = frame.index.searchsorted(pd.Timestamp(trade.exit_date), side="right")
        date_set.update(frame.index[lo:hi].tolist())
    if not date_set:  # pragma: no cover - date_set always seeded with as_of
        date_set.update(
            pd.date_range(
                as_of_ts, as_of_ts + pd.Timedelta(days=cfg.hold * 2), freq="B"
            ).tolist()
        )
    calendar = pd.DatetimeIndex(sorted(date_set))
    equity = build_equity_curve(
        calendar,
        trades,
        bars_by_tv,
        cfg.initial_capital,
        price_adjustment=cfg.price_adjustment,
    )

    benchmark = _benchmark_series_from_panel(price_panel, cfg.benchmark)
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(
        equity,
        benchmark_aligned,
        trades,
        slot_count,
        periods_per_year=periods_per_year_for_interval(cfg.interval),
    )
    metrics.update(compute_regime_metrics(benchmark, trades))
    metrics.update(
        compute_cost_metrics(
            portfolio.fees_paid,
            cfg.initial_capital,
            sum(float(t.pnl) for t in trades),
        )
    )

    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark_aligned,
        metrics=metrics,
        warnings=warnings,
        selection=selection,
    )


__all__ = [
    "_ReserveRotationSource",
    "_run_event_driven_sim",
    "_warmup_days_for_interval",
    "backtest_historical",
    "run_backtest",
    "select_candidates",
]
