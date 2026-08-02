"""Rolling backtest simulation and execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.core import (
    _active_or_pending_tickers,
    _FrameCache,
    _RunCaches,
    _SlotState,
    _bar_label,
    _benchmark_series_from_panel,
    _make_slot_state,
    _precompute_entry_signals,
    _precompute_filter_signals,
    prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.costs import cost_model_from_config
from screener.backtester.day_loop import (
    DayLoop,
    FreedSlot,
    _force_close_open_slots,
    run_day_loop,
)
from screener.backtester.fills import FillModel
from screener.backtester.data import PriceFetcher
from screener.backtester.fundamentals import (
    FundamentalFetcher,
    merge_fundamentals_into_bars,
)
from screener.backtester.metrics import (
    compute_cost_metrics,
    compute_metrics,
    compute_regime_metrics,
    periods_per_year_for_interval,
)
from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
)
from screener.backtester.pine import evaluate_panel_many, parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.sizing import entry_budget_for
from screener.backtester.warmup import _warmup_days_for_interval
from screener.regime import classify_regimes
from screener.options.backtest import merge_referenced_options
from screener.backtester.rolling_candidates import (
    _RollingCandidateMatrices,
    _build_rolling_candidate_matrices,
    _candidate_rows_for_day,
)


@dataclass(frozen=True)
class _RollingSimulationSetup:
    """Once-per-run setup for the rolling day loop and result assembly.

    When ``early_result`` is set, no trading day had price data and the caller
    returns it directly without running the day loop; the remaining fields are
    unused in that case.
    """

    early_result: BacktestResult | None
    master_dates: list[pd.Timestamp]
    candidate_matrices: _RollingCandidateMatrices | None
    bars_by_tv: dict[str, pd.DataFrame]
    benchmark: pd.Series
    exit_ast: object
    exit_signals: dict[str, pd.Series | str]
    portfolio: Portfolio | None
    slot_states: dict[int, _SlotState | None]
    slot_bars: dict[int, pd.DataFrame]
    selection_rows: list[dict]
    fill_model: FillModel | None
    day_loop: DayLoop | None


_PREPARATION_CONFIG_FIELDS = {
    "market",
    "as_of",
    "benchmark",
    "tickers",
    "universe_file",
    "membership_added",
    "membership_windows",
    "dynamic_universe_size",
    "dynamic_universe_lookback",
    "dynamic_universe_rebalance",
    "max_universe",
    "entry_expr",
    "exit_expr",
    "strategy_name",
    "regime_filter",
    "earnings_blackout_days",
    "fundamentals_provider",
    "fundamental_fields",
    "fundamental_lag_days",
    "sector_neutral",
    "interval",
    "price_adjustment",
    "min_price",
    "min_avg_dollar_volume",
    "avg_dollar_volume_window",
}


def _preparation_fingerprint(cfg: BacktestConfig) -> dict:
    return cfg.model_dump(mode="json", include=_PREPARATION_CONFIG_FIELDS)


@dataclass(frozen=True)
class PreparedRollingBacktest:
    """Reusable data, signals and memoized frame primitives for simulations.

    Runtime-only settings such as hold period, stops, sizing, costs, top-N and
    initial capital may change between simulations. Any setting that affects
    fetched/prepared bars, signals, filters or eligibility matrices is guarded
    by ``config_fingerprint``. The internal frame-cache dictionary is populated
    lazily by sequential runs; callers should not mutate it directly.
    """

    config_fingerprint: dict
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    master_dates: tuple[pd.Timestamp, ...]
    candidate_matrices: _RollingCandidateMatrices | None
    bars_by_tv: dict[str, pd.DataFrame]
    benchmark: pd.Series
    exit_ast: object
    exit_signals: dict[str, pd.Series | str]
    frame_caches: dict[str, _FrameCache]
    warnings: tuple[str, ...]
    early_result: BacktestResult | None = None

    def supports(self, cfg: BacktestConfig) -> bool:
        """Whether ``cfg`` can reuse these prepared bars and signals."""
        return _preparation_fingerprint(cfg) == self.config_fingerprint


def _window_bounds(
    cfg: BacktestConfig, start_date: date, end_date: date
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    if cfg.interval == "1d":
        end_ts = pd.Timestamp(end_date).normalize()
    else:
        end_ts = (
            pd.Timestamp(end_date).normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(1, "ns")
        )
    if end_ts < start_ts:
        raise ValueError("end_date must be >= start_date")
    return start_ts, end_ts


def _prepare_simulation(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warnings: list[str],
    earnings_blackout: dict[str, list[date]] | None = None,
    fundamental_fetcher: FundamentalFetcher | None = None,
) -> _RollingSimulationSetup:
    """Fetch data, precompute signals/matrices and build the slot/portfolio state."""
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

    # Warmup is measured in BARS (enough history for the longest indicator).
    # For daily bars one bar ~ one calendar day, so the legacy day-based padding
    # stands. For intraday, convert the required warmup bars into calendar days
    # via bars-per-session (with slack for weekends/holidays) so we don't request
    # ~365 days of minute data — which both blows past yfinance's intraday cap
    # and is unnecessary. Chunking longer intraday windows is Phase 2.
    warmup_days = _warmup_days_for_interval(lookback, cfg.interval)
    fetch_start = (start_ts - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_ts.date()
    price_panel = fetcher.fetch(yf_symbols, fetch_start, fetch_end)

    if cfg.price_adjustment == "splits_only":
        from screener.backtester.data import (
            apply_splits_only_adjustment,
            warn_unadjustable_fmp_frames,
        )

        warn_unadjustable_fmp_frames(price_panel)
        price_panel = apply_splits_only_adjustment(price_panel)

    # dict.get's default is eager, so the dict-comprehension form built one
    # throwaway DataFrame per symbol. Only materialise the empty frame for
    # symbols the panel is actually missing.
    bars_by_tv = {}
    for tv in tv_symbols:
        panel_bars = price_panel.get(yf_by_tv[tv])
        bars_by_tv[tv] = pd.DataFrame() if panel_bars is None else panel_bars
    bars_by_tv, strategy_lookback = prepare_strategy_bars(
        cfg.strategy_name,
        bars_by_tv,
        price_panel,
        tv_symbols,
        fetch_start,
        fetch_end,
        fetcher,
        warnings,
        market=cfg.market,
        benchmark=cfg.benchmark,
    )
    lookback = max(lookback, strategy_lookback)

    if fundamental_fetcher is not None:
        fundamentals = fundamental_fetcher.fetch(
            yf_by_tv.values(), fetch_start, fetch_end
        )
        bars_by_tv = merge_fundamentals_into_bars(bars_by_tv, fundamentals, yf_by_tv)

    bars_by_tv = merge_referenced_options(
        bars_by_tv,
        market=cfg.market,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        warnings=warnings,
    )

    exit_signals_by_tv: dict[str, pd.Series | str] = {}
    if exit_ast is None:
        entry_signals_by_tv = _precompute_entry_signals(bars_by_tv, entry_ast, warnings)
    else:
        evaluated_entry, evaluated_exit = evaluate_panel_many(
            (entry_ast, exit_ast), bars_by_tv
        )
        entry_signals_by_tv = _precompute_entry_signals(
            bars_by_tv,
            entry_ast,
            warnings,
            evaluated=evaluated_entry,
        )
        signal_cache = _RunCaches()
        signal_cache.prewarm_exit_signals(
            bars_by_tv,
            exit_ast,
            evaluated=evaluated_exit,
        )
        exit_signals_by_tv = signal_cache.exit_signals
    filter_signals_by_tv = _precompute_filter_signals(bars_by_tv, cfg)

    # Reuse the benchmark already fetched into ``price_panel`` (it is included in
    # ``yf_symbols`` above and split-adjusted alongside the portfolio symbols in
    # ``splits_only`` mode). Fetching it raw here would reintroduce the phantom
    # split jump into the regime gate, the aligned curve, and regime metrics.
    benchmark = _benchmark_series_from_panel(price_panel, cfg.benchmark)
    regime_allowed: pd.Series | None = None
    if cfg.regime_filter:
        regime_allowed = classify_regimes(benchmark).isin(set(cfg.regime_filter))

    day_arrays: list[np.ndarray] = []
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        idx = bars.index
        mask = (idx >= start_ts) & (idx <= end_ts)
        if mask.any():
            day_arrays.append(idx[mask].to_numpy())
    if not day_arrays:
        calendar = pd.bdate_range(start_ts, end_ts)
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
        benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
        metrics = compute_metrics(
            equity,
            benchmark_aligned,
            [],
            max(cfg.top, 1),
            periods_per_year=periods_per_year_for_interval(cfg.interval),
        )
        metrics["unique_tickers"] = 0
        early_result = BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=benchmark_aligned,
            metrics=metrics,
            warnings=warnings + ["no trading days with price data in rolling window"],
            selection=pd.DataFrame(),
        )
        return _RollingSimulationSetup(
            early_result=early_result,
            master_dates=[],
            candidate_matrices=None,
            bars_by_tv=bars_by_tv,
            benchmark=benchmark,
            exit_ast=exit_ast,
            exit_signals={},
            portfolio=None,
            slot_states={},
            slot_bars={},
            selection_rows=[],
            fill_model=None,
            day_loop=None,
        )

    master_dates = list(pd.DatetimeIndex(np.unique(np.concatenate(day_arrays))))
    sector_by_tv: dict[str, str] | None = None
    if cfg.sector_neutral:
        from screener.sectors import sector_by_ticker

        sector_by_tv = sector_by_ticker(tv_symbols, cfg.market)

    # Resolve earnings blackout map when the gate is configured. Prefer an
    # injected mapping (tests); otherwise collect via the market-aware earnings
    # date collectors (already disk-cached). Keys must be TV symbols to match
    # bars_by_tv / signal_mat columns.
    resolved_earnings_blackout = earnings_blackout
    if cfg.earnings_blackout_days is not None and resolved_earnings_blackout is None:
        from screener.earnings_backtest.earnings_dates import load_earnings_dates_map

        yf_tickers = list(dict.fromkeys(yf_by_tv.values()))
        span_years = max(
            3,
            int((end_ts.normalize() - start_ts.normalize()).days / 365) + 2,
        )
        yf_map = load_earnings_dates_map(yf_tickers, cfg.market, years=span_years)
        # Invert yf -> list[tv] so multi-mapped symbols still gate correctly.
        resolved_earnings_blackout = {}
        for tv, yf_sym in yf_by_tv.items():
            dates = yf_map.get(yf_sym)
            if dates:
                resolved_earnings_blackout[tv] = dates

    candidate_matrices = _build_rolling_candidate_matrices(
        bars_by_tv,
        entry_signals_by_tv,
        filter_signals_by_tv,
        master_dates,
        lookback,
        membership_added=dict(cfg.membership_added) or None,
        membership_windows=cfg.membership_windows,
        dynamic_universe_size=cfg.dynamic_universe_size,
        dynamic_universe_lookback=cfg.dynamic_universe_lookback,
        dynamic_universe_rebalance=cfg.dynamic_universe_rebalance,
        regime_allowed=regime_allowed,
        earnings_blackout=resolved_earnings_blackout,
        earnings_blackout_days=cfg.earnings_blackout_days,
        warnings=warnings,
        sector_neutral=cfg.sector_neutral,
        sector_by_tv=sector_by_tv,
    )
    portfolio = Portfolio(
        cfg.initial_capital,
        max(cfg.top, 1),
        cost_model=cost_model_from_config(cfg),
    )
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    selection_rows: list[dict] = []

    fill_model = FillModel(cfg, cost_model=portfolio.cost_model)
    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )
    return _RollingSimulationSetup(
        early_result=None,
        master_dates=master_dates,
        candidate_matrices=candidate_matrices,
        bars_by_tv=bars_by_tv,
        benchmark=benchmark,
        exit_ast=exit_ast,
        exit_signals=exit_signals_by_tv,
        portfolio=portfolio,
        slot_states=slot_states,
        slot_bars=slot_bars,
        selection_rows=selection_rows,
        fill_model=fill_model,
        day_loop=day_loop,
    )


class _DailyRankingSource:
    """Rolling :class:`~screener.backtester.day_loop.CandidateSource` adapter.

    Owns the rolling fill half: after the shared exit sweep, every slot that is
    now empty (whether idle since setup or freed today) is refilled from that
    day's freshly ranked candidate scan. There is no pre-exit work, so
    ``before_exits`` is a no-op. All state (slot maps, portfolio, selection rows,
    warnings) is shared by reference with the driver.
    """

    def __init__(
        self,
        *,
        candidate_matrices: _RollingCandidateMatrices,
        bars_by_tv: dict[str, pd.DataFrame],
        cfg: BacktestConfig,
        exit_ast,
        fill_model: FillModel,
        portfolio: Portfolio,
        slot_states: dict[int, _SlotState | None],
        slot_bars: dict[int, pd.DataFrame],
        end_ts: pd.Timestamp,
        selection_rows: list[dict],
        warnings: list[str],
        exit_signals: dict[str, pd.Series | str],
        frame_caches: dict[str, _FrameCache],
    ) -> None:
        self.candidate_matrices = candidate_matrices
        self.bars_by_tv = bars_by_tv
        self.cfg = cfg
        self.exit_ast = exit_ast
        self.fill_model = fill_model
        self.portfolio = portfolio
        self.slot_states = slot_states
        self.slot_bars = slot_bars
        self.end_ts = end_ts
        self.selection_rows = selection_rows
        self.warnings = warnings
        # Per-run memo: exit-AST evaluations and frame primitives are computed
        # once per ticker instead of once per slot open. Exit signals are filled
        # up front in one panel pass rather than one interpreted AST walk per
        # first-traded ticker.
        self.caches = _RunCaches(
            exit_signals=dict(exit_signals),
            frames=frame_caches,
        )

    def before_exits(self, day: pd.Timestamp) -> None:
        return None

    def after_exits(self, day: pd.Timestamp, freed: list[FreedSlot]) -> None:
        """Refill freed slots from the day's candidate ranking.

        Mutates ``slot_states``, ``slot_bars``, ``portfolio``, ``selection_rows``
        and ``warnings`` in place, matching the original interleaved loop body.
        """
        cfg = self.cfg
        portfolio = self.portfolio
        slot_states = self.slot_states
        # Treat every slot that is now empty (whether already idle or freed
        # today) as available for refill. Order is slot-id ascending, matching
        # the original interleaved loop.
        free_slots: list[int] = [
            slot_id for slot_id, state in slot_states.items() if state is None
        ]

        if not free_slots:
            return

        candidates, day_warnings = _candidate_rows_for_day(
            day,
            self.candidate_matrices,
            exclude=_active_or_pending_tickers(slot_states),
        )
        self.warnings.extend(day_warnings)
        if not candidates:
            return
        candidate_queue: deque[dict] = deque(candidates)

        for slot_id in free_slots:
            opened = False
            while candidate_queue and not opened:
                row = candidate_queue.popleft()
                ticker = str(row["ticker"])
                if (
                    ticker
                    in _active_or_pending_tickers(  # pragma: no cover - candidates pre-excluded
                        slot_states
                    )
                ):
                    continue
                # No default: dict.get evaluates its default eagerly, so passing
                # pd.DataFrame() built and threw away a frame on every candidate
                # popped. The guard below already treats missing as empty.
                bars = self.bars_by_tv.get(ticker)
                if (
                    bars is None or bars.empty
                ):  # pragma: no cover - only valid tickers ranked
                    continue
                entry_budget = entry_budget_for(
                    cfg, portfolio, bars, int(row["signal_idx"])
                )
                state, warn = _make_slot_state(
                    ticker,
                    bars,
                    int(row["signal_idx"]),
                    cfg,
                    self.exit_ast,
                    int(row["rank"]),
                    self.fill_model,
                    caches=self.caches,
                    entry_budget=entry_budget,
                )
                if (
                    state is None
                ):  # pragma: no cover - lookback_ok guarantees a post-signal bar
                    if warn:
                        self.warnings.append(f"{ticker}: {warn}")
                    continue
                if (
                    pd.Timestamp(state.entry_date) > self.end_ts
                ):  # pragma: no cover - fetch_end == end_ts
                    continue
                portfolio.assign(ticker, int(row["rank"]), _bar_label(day, cfg))
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    budget=entry_budget,
                    shares=state.entry_shares,
                )
                slot_states[slot_id] = state
                self.slot_bars[slot_id] = bars
                self.selection_rows.append(
                    {
                        "ticker": ticker,
                        "signal_date": _bar_label(day, cfg),
                        "as_of_close": row["as_of_close"],
                        "as_of_volume": row["as_of_volume"],
                        "as_of_dollar_vol": row["as_of_dollar_vol"],
                        "rank": row["rank"],
                        "role": "active",
                    }
                )
                opened = True


def _assemble_results(
    *,
    portfolio: Portfolio,
    master_dates: list[pd.Timestamp],
    bars_by_tv: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    benchmark: pd.Series,
    selection_rows: list[dict],
    warnings: list[str],
) -> BacktestResult:
    """Assemble the trade ledger, equity curve, metrics and selection frame."""
    trades = portfolio.closed_trades()

    # ``master_dates`` was built as the sorted union of every ticker's bars in
    # the simulation window. Every entry, exit and held bar is therefore
    # already present; rebuilding the same union once per trade only repeats
    # thousands of Index.searchsorted/slice/tolist operations.
    calendar = pd.DatetimeIndex(master_dates)
    equity = build_equity_curve(
        calendar,
        trades,
        bars_by_tv,
        cfg.initial_capital,
        price_adjustment=cfg.price_adjustment,
    )
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(
        equity,
        benchmark_aligned,
        trades,
        max(cfg.top, 1),
        periods_per_year=periods_per_year_for_interval(cfg.interval),
    )
    metrics["unique_tickers"] = len({trade.ticker for trade in trades})
    metrics.update(compute_regime_metrics(benchmark, trades))
    metrics.update(
        compute_cost_metrics(
            portfolio.fees_paid,
            cfg.initial_capital,
            sum(float(t.pnl) for t in trades),
        )
    )

    selection = pd.DataFrame(
        selection_rows,
        columns=[
            "ticker",
            "signal_date",
            "as_of_close",
            "as_of_volume",
            "as_of_dollar_vol",
            "rank",
            "role",
        ],
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


def prepare_rolling_backtest(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_date: date,
    end_date: date,
    earnings_blackout: dict[str, list[date]] | None = None,
    fundamental_fetcher: FundamentalFetcher | None = None,
) -> PreparedRollingBacktest:
    """Fetch and precompute the immutable half of a rolling backtest.

    The returned object can drive many simulations whose changes are confined
    to execution/portfolio settings (for example hold, stops, sizing, costs,
    top-N or initial capital).
    """
    if cfg.fundamentals_provider and fundamental_fetcher is None:
        raise ValueError("fundamentals_provider requires a resolved FundamentalFetcher")
    warnings: list[str] = []
    start_ts, end_ts = _window_bounds(cfg, start_date, end_date)
    setup = _prepare_simulation(
        cfg,
        fetcher,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        earnings_blackout=earnings_blackout,
        fundamental_fetcher=fundamental_fetcher,
    )
    prepared_warnings = (
        tuple(setup.early_result.warnings)
        if setup.early_result is not None
        else tuple(warnings)
    )
    return PreparedRollingBacktest(
        config_fingerprint=_preparation_fingerprint(cfg),
        start_ts=start_ts,
        end_ts=end_ts,
        master_dates=tuple(setup.master_dates),
        candidate_matrices=setup.candidate_matrices,
        bars_by_tv=setup.bars_by_tv,
        benchmark=setup.benchmark,
        exit_ast=setup.exit_ast,
        exit_signals=setup.exit_signals,
        frame_caches={},
        warnings=prepared_warnings,
        early_result=setup.early_result,
    )


def run_prepared_rolling_backtest(
    prepared: PreparedRollingBacktest,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Run one mutable portfolio simulation over reusable prepared data."""
    if not prepared.supports(cfg):
        raise ValueError(
            "configuration changes prepared-data fields; build a new "
            "PreparedRollingBacktest"
        )
    warnings = list(prepared.warnings)
    if prepared.early_result is not None:
        calendar = prepared.early_result.equity_curve.index
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
        benchmark_aligned = prepared.benchmark.reindex(
            calendar, method="ffill"
        ).dropna()
        metrics = compute_metrics(
            equity,
            benchmark_aligned,
            [],
            max(cfg.top, 1),
            periods_per_year=periods_per_year_for_interval(cfg.interval),
        )
        metrics["unique_tickers"] = 0
        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=benchmark_aligned,
            metrics=metrics,
            warnings=warnings,
            selection=pd.DataFrame(),
        )

    assert prepared.candidate_matrices is not None
    portfolio = Portfolio(
        cfg.initial_capital,
        max(cfg.top, 1),
        cost_model=cost_model_from_config(cfg),
    )
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    selection_rows: list[dict] = []
    fill_model = FillModel(cfg, cost_model=portfolio.cost_model)
    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )

    source = _DailyRankingSource(
        candidate_matrices=prepared.candidate_matrices,
        bars_by_tv=prepared.bars_by_tv,
        cfg=cfg,
        exit_ast=prepared.exit_ast,
        fill_model=fill_model,
        portfolio=portfolio,
        slot_states=slot_states,
        slot_bars=slot_bars,
        end_ts=prepared.end_ts,
        selection_rows=selection_rows,
        warnings=warnings,
        exit_signals=prepared.exit_signals,
        frame_caches=prepared.frame_caches,
    )
    run_day_loop(prepared.master_dates, day_loop, source)

    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=prepared.end_ts,
        fill_model=fill_model,
    )

    return _assemble_results(
        portfolio=portfolio,
        master_dates=list(prepared.master_dates),
        bars_by_tv=prepared.bars_by_tv,
        cfg=cfg,
        benchmark=prepared.benchmark,
        selection_rows=selection_rows,
        warnings=warnings,
    )


def run_rolling_backtest(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_date: date,
    end_date: date,
    earnings_blackout: dict[str, list[date]] | None = None,
    fundamental_fetcher: FundamentalFetcher | None = None,
) -> BacktestResult:
    """Run a daily rolling simulation over ``[start_date, end_date]``."""
    prepared = prepare_rolling_backtest(
        cfg,
        fetcher,
        start_date=start_date,
        end_date=end_date,
        earnings_blackout=earnings_blackout,
        fundamental_fetcher=fundamental_fetcher,
    )
    return run_prepared_rolling_backtest(prepared, cfg)
