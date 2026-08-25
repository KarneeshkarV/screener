"""Rolling backtest simulation and execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from screener.backtester.book import BOOK_CONFIG_FIELDS, open_book
from screener.backtester.core import (
    _active_or_pending_tickers,
    _bar_label,
    _build_frame_cache,
    _FrameCache,
    _make_slot_state,
    _RunCaches,
    _SlotState,
)
from screener.backtester.data import PriceFetcher
from screener.backtester.day_loop import (
    FreedSlot,
    _force_close_open_slots,
    run_day_loop,
)
from screener.backtester.fills import FillModel
from screener.backtester.fundamentals import FundamentalFetcher
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
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.price_panel import (
    PRICE_PANEL_CONFIG_FIELDS,
    PricePanelInputs,
    build_price_panel,
)
from screener.backtester.rolling_candidates import (
    _candidate_rows_for_day,
    _RollingCandidateMatrices,
)
from screener.backtester.signal_panel import (
    SIGNAL_PANEL_CONFIG_FIELDS,
    SignalPanelInputs,
    build_signal_panel,
    parse_signal_program,
)
from screener.backtester.sizing import entry_budget_for

# The reuse key is derived, not curated: every field either module declares as
# an input is in it, plus every field no module claims at all. The old
# hand-maintained allowlist had the opposite default - a new config field that
# changed bars was silently absent from the key, so a sweep happily reused
# stale panels. Here the cost of forgetting is a rebuild, not a wrong number.
_PREPARED_CONFIG_FIELDS = PRICE_PANEL_CONFIG_FIELDS | SIGNAL_PANEL_CONFIG_FIELDS


def _preparation_fingerprint(cfg: BacktestConfig) -> dict[str, Any]:
    """Return the reuse key: prepared-panel inputs plus anything unclassified."""
    dumped = cfg.model_dump(mode="json")
    return {
        name: value
        for name, value in sorted(dumped.items())
        if name in _PREPARED_CONFIG_FIELDS or name not in BOOK_CONFIG_FIELDS
    }


@dataclass(frozen=True)
class PreparedRollingBacktest:
    """Reusable data, signals and memoized frame primitives for simulations.

    Book settings (hold period, stops, sizing, costs, top-N, initial capital)
    may change between simulations; everything else is guarded by
    ``config_fingerprint``, which is derived from the panel modules' declared
    inputs rather than curated by hand. The internal frame-cache dictionary is
    populated lazily by sequential runs; callers should not mutate it directly.
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


def _mark_rank_exits(
    *,
    prev_day: pd.Timestamp,
    cfg: BacktestConfig,
    candidate_matrices: _RollingCandidateMatrices,
    slot_states: dict[int, _SlotState | None],
    warnings: list[str],
) -> None:
    """Flag held slots whose ticker left the prior completed bar's top-N.

    The membership decision reads the ranking of ``prev_day`` (with held
    tickers included, ``exclude=set()``) so no same-bar information feeds the
    decision; the shared exit sweep then closes flagged slots later today at
    that bar's close, through the normal dividend -> partial -> full-exit
    sequence with stop/trail/target still taking precedence. Strict rule:
    anything absent from the (possibly short) list is flagged.
    """
    rows, day_warnings = _candidate_rows_for_day(
        prev_day,
        candidate_matrices,
        exclude=set(),
        limit=int(cfg.rank_universe_size),
    )
    warnings.extend(day_warnings)
    top = {str(row["ticker"]) for row in rows}
    for state in slot_states.values():
        if state is not None:
            state.rank_exit = state.ticker not in top


class _DailyRankingSource:
    """Rolling :class:`~screener.backtester.day_loop.CandidateSource` adapter.

    Owns the rolling fill half: after the shared exit sweep, every slot that is
    now empty (whether idle since setup or freed today) is refilled from that
    day's freshly ranked candidate scan. ``before_exits`` only marks rank-exit
    flags on scheduled bars when the feature is enabled. All state (slot maps,
    portfolio, selection rows, warnings) is shared by reference with the driver.
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
        # Trading-bar counter and last-seen bar for the rank-exit schedule;
        # membership is judged on the prior completed bar, fires every Nth bar.
        self._day_count = 0
        self._prev_day: pd.Timestamp | None = None
        # Per-run memo: exit-AST evaluations and frame primitives are computed
        # once per ticker instead of once per slot open. Exit signals are filled
        # up front in one panel pass rather than one interpreted AST walk per
        # first-traded ticker. Bool ndarrays are derived once here so the day
        # loop never re-boxes Series for the per-bar exit check.
        exit_signal_values = {
            ticker: np.asarray(signal.to_numpy(), dtype=bool)
            for ticker, signal in exit_signals.items()
            if not isinstance(signal, str)
        }
        self.caches = _RunCaches(
            exit_signals=dict(exit_signals),
            exit_signal_values=exit_signal_values,
            frames=frame_caches,
        )

    def before_exits(self, day: pd.Timestamp) -> None:
        if self.cfg.rank_exit_every is not None:
            self._day_count += 1
            if (
                self._day_count % int(self.cfg.rank_exit_every) == 0
                and self._prev_day is not None
            ):
                _mark_rank_exits(
                    prev_day=self._prev_day,
                    cfg=self.cfg,
                    candidate_matrices=self.candidate_matrices,
                    slot_states=self.slot_states,
                    warnings=self.warnings,
                )
        self._prev_day = day

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

        # One active-ticker set for the day: exclude at rank time and keep it
        # updated as slots open so the inner loop is an O(1) membership check
        # instead of rescanning slot_states per candidate.
        active_tickers = _active_or_pending_tickers(slot_states)

        # Rank the full eligible set but only materialise top-N plus a small
        # overfetch for open failures (session-last, quote gaps). Without the
        # cap every free day built list[dict] for the whole universe.
        overfetch = max(8, int(cfg.top))
        materialise_limit = max(len(free_slots), int(cfg.top)) + overfetch
        candidates, day_warnings = _candidate_rows_for_day(
            day,
            self.candidate_matrices,
            exclude=active_tickers,
            limit=materialise_limit,
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
                    ticker in active_tickers
                ):  # pragma: no cover - candidates pre-excluded
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
                active_tickers.add(ticker)
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

    Composes the two knowledge modules - the price panel (what bars exist) and
    the signal panel (which ticker is eligible on which bar). The book (capital,
    slots, fills) is deliberately absent: it belongs to a single simulation, so
    it is opened per run by :func:`run_prepared_rolling_backtest`.

    The returned object can drive many simulations whose changes are confined
    to execution/portfolio settings (for example hold, stops, sizing, costs,
    top-N or initial capital).
    """
    if cfg.fundamentals_provider and fundamental_fetcher is None:
        raise ValueError("fundamentals_provider requires a resolved FundamentalFetcher")
    warnings: list[str] = []
    start_ts, end_ts = _window_bounds(cfg, start_date, end_date)

    signal_inputs = SignalPanelInputs.from_config(cfg)
    program = parse_signal_program(signal_inputs)
    panel = build_price_panel(
        PricePanelInputs.from_config(cfg),
        fetcher,
        entry_ast=program.entry_ast,
        exit_ast=program.exit_ast,
        lookback=program.lookback,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        fundamental_fetcher=fundamental_fetcher,
    )
    signals = build_signal_panel(
        signal_inputs,
        panel,
        program=program,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        earnings_blackout=earnings_blackout,
    )

    early_result: BacktestResult | None = None
    if signals.candidate_matrices is None:
        early_result = no_trades_result(
            cfg,
            calendar=pd.bdate_range(start_ts, end_ts),
            benchmark=panel.benchmark,
            warnings=warnings + ["no trading days with price data in rolling window"],
        )
    prepared_warnings = (
        tuple(early_result.warnings) if early_result is not None else tuple(warnings)
    )
    # Prebuild frame primitives once for the whole universe. Sequential runs that
    # reuse this prepared object (parameter sweeps) then skip the lazy first-open
    # construction, and the first simulation no longer pays for it on the hot day
    # path either.
    frame_caches = {
        tv: _build_frame_cache(bars)
        for tv, bars in panel.bars_by_tv.items()
        if bars is not None and not bars.empty
    }
    return PreparedRollingBacktest(
        config_fingerprint=_preparation_fingerprint(cfg),
        start_ts=start_ts,
        end_ts=end_ts,
        master_dates=tuple(panel.master_dates),
        candidate_matrices=signals.candidate_matrices,
        bars_by_tv=panel.bars_by_tv,
        benchmark=panel.benchmark,
        exit_ast=program.exit_ast,
        exit_signals=signals.exit_signals,
        frame_caches=frame_caches,
        warnings=prepared_warnings,
        early_result=early_result,
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
        return no_trades_result(
            cfg,
            calendar=prepared.early_result.equity_curve.index,
            benchmark=prepared.benchmark,
            warnings=warnings,
        )

    assert prepared.candidate_matrices is not None
    book = open_book(cfg)

    source = _DailyRankingSource(
        candidate_matrices=prepared.candidate_matrices,
        bars_by_tv=prepared.bars_by_tv,
        cfg=cfg,
        exit_ast=prepared.exit_ast,
        fill_model=book.fill_model,
        portfolio=book.portfolio,
        slot_states=book.slot_states,
        slot_bars=book.slot_bars,
        end_ts=prepared.end_ts,
        selection_rows=book.selection_rows,
        warnings=warnings,
        exit_signals=prepared.exit_signals,
        frame_caches=prepared.frame_caches,
    )
    run_day_loop(prepared.master_dates, book.day_loop, source)

    _force_close_open_slots(
        slot_states=book.slot_states,
        slot_bars=book.slot_bars,
        cfg=cfg,
        portfolio=book.portfolio,
        end_ts=prepared.end_ts,
        fill_model=book.fill_model,
    )

    return _assemble_results(
        portfolio=book.portfolio,
        master_dates=list(prepared.master_dates),
        bars_by_tv=prepared.bars_by_tv,
        cfg=cfg,
        benchmark=prepared.benchmark,
        selection_rows=book.selection_rows,
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
