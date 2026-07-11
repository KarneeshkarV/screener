"""Rolling backtest simulation and execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.core import (
    _active_or_pending_tickers,
    _RunCaches,
    _SlotState,
    _bar_label,
    _benchmark_series_from_panel,
    _force_close_open_slots,
    _make_slot_state,
    _precompute_entry_signals,
    _precompute_filter_signals,
    _prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.day_loop import DayLoop, FreedSlot, run_day_loop
from screener.backtester.fills import FillModel
from screener.backtester.data import PriceFetcher
from screener.backtester.fundamentals import (
    build_fundamental_fetcher,
    merge_fundamentals_into_bars,
)
from screener.backtester.metrics import (
    compute_metrics,
    compute_regime_metrics,
    periods_per_year_for_interval,
)
from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
)
from screener.backtester.pine import parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.regime import classify_regimes
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
    portfolio: Portfolio | None
    slot_states: dict[int, _SlotState | None]
    slot_bars: dict[int, pd.DataFrame]
    selection_rows: list[dict]
    fill_model: FillModel | None
    day_loop: DayLoop | None


def _prepare_simulation(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warnings: list[str],
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
    warmup_bars = lookback * 3 + 30
    if cfg.interval == "1d":
        warmup_days = max(warmup_bars, 365)
    else:
        bars_per_day = max(periods_per_year_for_interval(cfg.interval) // 252, 1)
        warmup_days = int(np.ceil(warmup_bars / bars_per_day) * 1.6) + 5
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

    bars_by_tv = {
        tv: price_panel.get(yf_by_tv[tv], pd.DataFrame()) for tv in tv_symbols
    }
    bars_by_tv, strategy_lookback = _prepare_strategy_bars(
        cfg,
        bars_by_tv,
        price_panel,
        tv_symbols,
        fetch_start,
        fetch_end,
        fetcher,
        warnings,
    )
    lookback = max(lookback, strategy_lookback)

    if cfg.fundamentals_provider:
        fundamental_fetcher = build_fundamental_fetcher(
            cfg.fundamentals_provider,
            fields=cfg.fundamental_fields or None,
            lag_days=cfg.fundamental_lag_days,
        )
        if fundamental_fetcher is not None:
            fundamentals = fundamental_fetcher.fetch(
                yf_by_tv.values(), fetch_start, fetch_end, cfg.market
            )
            bars_by_tv = merge_fundamentals_into_bars(
                bars_by_tv, fundamentals, yf_by_tv
            )

    entry_signals_by_tv = _precompute_entry_signals(bars_by_tv, entry_ast, warnings)
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
            portfolio=None,
            slot_states={},
            slot_bars={},
            selection_rows=[],
            fill_model=None,
            day_loop=None,
        )

    master_dates = list(pd.DatetimeIndex(np.unique(np.concatenate(day_arrays))))
    candidate_matrices = _build_rolling_candidate_matrices(
        bars_by_tv,
        entry_signals_by_tv,
        filter_signals_by_tv,
        master_dates,
        lookback,
        membership_added=dict(cfg.membership_added) or None,
        regime_allowed=regime_allowed,
        warnings=warnings,
    )
    portfolio = Portfolio(cfg.initial_capital, max(cfg.top, 1))
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    selection_rows: list[dict] = []

    fill_model = FillModel(cfg)
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
        # once per ticker instead of once per slot open.
        self.caches = _RunCaches()

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
                bars = self.bars_by_tv.get(ticker, pd.DataFrame())
                if (
                    bars is None or bars.empty
                ):  # pragma: no cover - only valid tickers ranked
                    continue
                state, warn = _make_slot_state(
                    ticker,
                    bars,
                    int(row["signal_idx"]),
                    cfg,
                    self.exit_ast,
                    int(row["rank"]),
                    self.fill_model,
                    caches=self.caches,
                    entry_budget=portfolio.entry_budget(),
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
                    commission_bps=cfg.commission_bps,
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

    date_set: set[pd.Timestamp] = set(master_dates)
    for trade in trades:
        frame = bars_by_tv.get(trade.ticker)
        if frame is None or frame.empty:  # pragma: no cover - traded ticker has bars
            continue
        # frame.index is a sorted DatetimeIndex; slice by position instead of
        # building two boolean masks per trade.
        lo = frame.index.searchsorted(pd.Timestamp(trade.entry_date), side="left")
        hi = frame.index.searchsorted(pd.Timestamp(trade.exit_date), side="right")
        date_set.update(frame.index[lo:hi].tolist())
    calendar = pd.DatetimeIndex(sorted(date_set))
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


def run_rolling_backtest(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_date: date,
    end_date: date,
) -> BacktestResult:
    """Run a daily rolling simulation over ``[start_date, end_date]``."""
    warnings: list[str] = []
    start_ts = pd.Timestamp(start_date).normalize()
    # For daily bars the window upper bound is midnight of end_date (bar
    # timestamps are midnight-normalized). Intraday bars carry a time-of-day, so
    # midnight would exclude every bar on the final session — extend the bound to
    # the last instant of end_date so the closing session is included.
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

    setup = _prepare_simulation(
        cfg,
        fetcher,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
    )
    if setup.early_result is not None:
        return setup.early_result

    assert setup.candidate_matrices is not None
    assert setup.portfolio is not None
    assert setup.fill_model is not None
    assert setup.day_loop is not None

    source = _DailyRankingSource(
        candidate_matrices=setup.candidate_matrices,
        bars_by_tv=setup.bars_by_tv,
        cfg=cfg,
        exit_ast=setup.exit_ast,
        fill_model=setup.fill_model,
        portfolio=setup.portfolio,
        slot_states=setup.slot_states,
        slot_bars=setup.slot_bars,
        end_ts=end_ts,
        selection_rows=setup.selection_rows,
        warnings=warnings,
    )
    run_day_loop(setup.master_dates, setup.day_loop, source)

    _force_close_open_slots(
        slot_states=setup.slot_states,
        slot_bars=setup.slot_bars,
        cfg=cfg,
        portfolio=setup.portfolio,
        end_ts=end_ts,
        fill_model=setup.fill_model,
    )

    return _assemble_results(
        portfolio=setup.portfolio,
        master_dates=setup.master_dates,
        bars_by_tv=setup.bars_by_tv,
        cfg=cfg,
        benchmark=setup.benchmark,
        selection_rows=setup.selection_rows,
        warnings=warnings,
    )
