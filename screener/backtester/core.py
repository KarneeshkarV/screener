"""Shared backtest primitives used by multiple backtest flows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional, Protocol, Union, cast

import numpy as np
import pandas as pd

from screener.backtester.costs import corwin_schultz_half_spread
from screener.backtester.data import PriceFetcher
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.ledger import ExitReason
from screener.backtester.pine import (
    PineError,
    evaluate,
    evaluate_panel,
    panel_index_key,
)
from screener.backtester.portfolio import Portfolio
from screener.backtester.sessions import is_session_last, market_timezone

if TYPE_CHECKING:
    from screener.strategies.spec import StrategySpec


class UniverseSpec(Protocol):
    """The config values universe resolution reads, and nothing else.

    Structural so both ``BacktestConfig`` and the panel-scoped input records in
    :mod:`screener.backtester.price_panel` satisfy it without either module
    having to know about the other.
    """

    @property
    def tickers(self) -> Optional[tuple[str, ...]]: ...

    @property
    def universe_file(self) -> Optional[str]: ...

    @property
    def membership_windows(self) -> tuple[tuple[str, date, date | None], ...]: ...

    @property
    def dynamic_universe_size(self) -> int | None: ...

    @property
    def max_universe(self) -> int: ...


class LiquidityFilterSpec(Protocol):
    """The config values the min-price/ADV entry filters read, and nothing else."""

    @property
    def min_price(self) -> Optional[float]: ...

    @property
    def min_avg_dollar_volume(self) -> Optional[float]: ...

    @property
    def avg_dollar_volume_window(self) -> int: ...


def _bar_label(ts, cfg: BacktestConfig) -> Union[date, datetime]:
    """Return the trade/position stamp for a bar timestamp.

    Daily bars are midnight-normalized, so a plain ``date`` is returned — this
    keeps the ledger byte-for-byte identical to the pre-intraday engine and
    comparable to the ``date``-typed test fixtures. Intraday bars return a full
    ``datetime`` so time-of-day survives into ``Trade``/``Position``.
    """
    ts = pd.Timestamp(ts)
    if cfg.interval == "1d":
        return cast(date, ts.date())
    return cast(datetime, ts.to_pydatetime())


def _trailing_liquidity(
    bars: pd.DataFrame, signal_idx: int, window: int = 20
) -> tuple[float, float]:
    """Return ``(adv_shares, sigma_daily)`` over trailing bars ending at ``signal_idx``."""
    if signal_idx < 0 or window <= 0:
        return 0.0, 0.0
    start = max(0, signal_idx - window + 1)
    window_bars = bars.iloc[start : signal_idx + 1]
    if window_bars.empty:
        return 0.0, 0.0
    vol = window_bars["volume"].astype(float)
    adv = float(vol.mean()) if vol.size else 0.0
    close = window_bars["close"].astype(float)
    if close.size < 2:
        sigma = 0.0
    else:
        rets = close.pct_change(fill_method=None).dropna()
        sigma = float(rets.std()) if rets.size else 0.0
    if not np.isfinite(adv):
        adv = 0.0
    if not np.isfinite(
        sigma
    ):  # pragma: no cover - defensive guard for pathological data
        sigma = 0.0
    return adv, sigma


@dataclass
class _FrameCache:
    """Once-per-frame primitives shared by every slot opened on one ticker.

    Raw numpy copies of the OHLC(+dividend) columns plus a date->position map
    let the per-bar hot loops (``_check_exit_at_bar``, ``_close_slot_at_day``,
    ``_fire_partial_exits_at_bar``) run on integer positions instead of the
    pandas scalar-access family (``bars.iloc[i]`` / ``Index.get_loc``). The
    float volume Series and precomputed close-to-close returns back
    :func:`_cached_trailing_liquidity` so per-slot liquidity stats reuse the
    exact original arithmetic without re-slicing the frame. Built lazily by
    :meth:`_RunCaches.frame`.
    """

    open_arr: np.ndarray
    high_arr: np.ndarray
    low_arr: np.ndarray
    close_arr: np.ndarray
    dividend_arr: Optional[np.ndarray]
    # int64 epoch-nanosecond view of a unique, naive datetime64[ns] index for
    # O(log n) day->position lookups; None otherwise, and callers then keep
    # the original ``Index.get_loc`` semantics (duplicates, exotic dtypes).
    index_i8: Optional[np.ndarray]
    volume_f: pd.Series
    # close[i] / close[i-1] - 1: the exact ``pct_change`` arithmetic for
    # NaN-free windows (NaN-containing windows fall back to the original).
    rets_f: pd.Series
    liquidity_by_idx: dict[tuple[int, int], tuple[float, float]] = field(
        default_factory=dict
    )
    # Lazily filled Corwin-Schultz half-spread series (fraction), aligned to
    # ``bars.index``. None until first ``spread_proxy`` lookup for this frame.
    half_spread_s: Optional[pd.Series] = None
    # Lazily filled session-last mask (see ``sessions.is_session_last``); None
    # until the first ``intraday_only`` slot open on this frame.
    session_last: Optional[np.ndarray] = None


def _build_frame_cache(bars: pd.DataFrame) -> _FrameCache:
    close_arr = bars["close"].to_numpy(dtype=float)
    index = bars.index
    index_i8: Optional[np.ndarray] = None
    if (
        isinstance(index, pd.DatetimeIndex)
        and index.dtype == np.dtype("datetime64[ns]")
        and index.is_unique
    ):
        index_i8 = index.to_numpy().view("i8")
    rets = np.empty(close_arr.shape[0], dtype=float)
    if rets.shape[0]:
        rets[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            rets[1:] = close_arr[1:] / close_arr[:-1] - 1.0
    return _FrameCache(
        open_arr=bars["open"].to_numpy(dtype=float),
        high_arr=bars["high"].to_numpy(dtype=float),
        low_arr=bars["low"].to_numpy(dtype=float),
        close_arr=close_arr,
        dividend_arr=(
            bars["dividend"].to_numpy() if "dividend" in bars.columns else None
        ),
        index_i8=index_i8,
        volume_f=bars["volume"].astype(float),
        rets_f=pd.Series(rets, index=bars.index),
    )


@dataclass
class _RunCaches:
    """Per-run memoisation shared across slot opens and candidate checks.

    Keyed by ticker: each engine holds one immutable frame per ticker for the
    whole run, so full-frame signal evaluations and frame primitives are
    computed once and reused. ``entry_signals`` stores the raw entry-AST
    evaluation (or the :class:`PineError` it raised, so repeat callers keep
    the original per-call failure semantics); ``exit_signals`` stores the
    normalised boolean exit Series (or the warning string its failure
    produced).
    """

    entry_signals: dict[str, pd.Series | PineError] = field(default_factory=dict)
    exit_signals: dict[str, pd.Series | str] = field(default_factory=dict)
    frames: dict[str, _FrameCache] = field(default_factory=dict)

    def frame(self, ticker: str, bars: pd.DataFrame) -> _FrameCache:
        cache = self.frames.get(ticker)
        if cache is None:
            cache = _build_frame_cache(bars)
            self.frames[ticker] = cache
        return cache

    def entry_signal(
        self, ticker: str, bars: pd.DataFrame, entry_ast
    ) -> pd.Series | PineError:
        cached = self.entry_signals.get(ticker)
        if cached is None:
            try:
                cached = evaluate(entry_ast, bars)
            except PineError as exc:
                cached = exc
            self.entry_signals[ticker] = cached
        return cached

    def prewarm_exit_signals(
        self,
        bars_by_ticker: dict[str, pd.DataFrame],
        exit_ast,
        *,
        evaluated: dict[str, pd.Series | PineError] | None = None,
    ) -> None:
        """Fill ``exit_signals`` for every ticker in one panel pass.

        ``_make_slot_state`` otherwise evaluates the exit AST on the first slot
        open per ticker — correct, but one interpreted walk each. Batching them
        up front costs one walk per index-group for the whole universe, which is
        cheaper even though not every ticker ends up being traded.

        Entries are stored in exactly the form the lazy path stores: the
        normalised boolean Series, or the warning string a failure produced, so
        slot-open behaviour and warning text are unchanged.
        """
        if exit_ast is None:
            return
        results = (
            evaluated
            if evaluated is not None
            else evaluate_panel(exit_ast, bars_by_ticker)
        )
        for ticker, result in results.items():
            if ticker in self.exit_signals:
                continue
            if isinstance(result, PineError):
                self.exit_signals[ticker] = f"exit eval failed: {result}"
            else:
                self.exit_signals[ticker] = result.fillna(False).astype(bool)


def _cached_trailing_liquidity(
    frame_cache: _FrameCache,
    bars: pd.DataFrame,
    signal_idx: int,
    window: int = 20,
) -> tuple[float, float]:
    """:func:`_trailing_liquidity` backed by a :class:`_FrameCache`.

    Identical arithmetic: the volume mean and return std run over the same
    float values in the same order (pandas ``Series.mean``/``std``), just
    without re-slicing the frame and re-converting dtypes per slot open, and
    memoised per ``(signal_idx, window)``.
    """
    if signal_idx < 0 or window <= 0:
        return 0.0, 0.0
    memo = frame_cache.liquidity_by_idx
    cached = memo.get((signal_idx, window))
    if cached is not None:
        return cached
    start = max(0, signal_idx - window + 1)
    close_win = frame_cache.close_arr[start : signal_idx + 1]
    if close_win.size == 0:
        result = (0.0, 0.0)
    elif np.isnan(close_win).any():
        # A NaN close inside the window changes ``pct_change``'s pad-fill
        # across the window boundary; defer to the original slice-based
        # computation for exact parity.
        result = _trailing_liquidity(bars, signal_idx, window)
    else:
        vol = frame_cache.volume_f.iloc[start : signal_idx + 1]
        adv = float(vol.mean()) if vol.size else 0.0
        if close_win.size < 2:
            sigma = 0.0
        else:
            rets = frame_cache.rets_f.iloc[start + 1 : signal_idx + 1].dropna()
            sigma = float(rets.std()) if rets.size else 0.0
        if not np.isfinite(adv):
            adv = 0.0
        if not np.isfinite(
            sigma
        ):  # pragma: no cover - defensive guard for pathological data
            sigma = 0.0
        result = (adv, sigma)
    memo[(signal_idx, window)] = result
    return result


def _passes_entry_filters(
    bars: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    cfg: BacktestConfig,
) -> tuple[bool, Optional[str]]:
    """Check min-price and liquidity filters against history up to ``as_of_ts``."""
    if cfg.min_price is None and cfg.min_avg_dollar_volume is None:
        return True, None
    # bars.index is a sorted DatetimeIndex; use searchsorted for O(log n) lookup.
    pos = int(bars.index.searchsorted(as_of_ts, side="right"))
    if pos <= 0:
        return False, "no history"
    last = bars.iloc[pos - 1]
    close = float(last["close"])
    # A non-finite close never passes the price filter, and cannot be rescued
    # by an ADV mean that would otherwise skip its missing value.
    if not np.isfinite(close):
        return False, f"price filter requires a finite close (close={close:.4f})"
    if cfg.min_price is not None:
        min_price = float(cfg.min_price)
        if not np.isfinite(min_price) or close < min_price:
            return False, (
                "price filter requires finite close and min_price "
                f"(close={close:.4f}, min_price={min_price})"
            )
    if cfg.min_avg_dollar_volume is not None:
        window = max(int(cfg.avg_dollar_volume_window), 1)
        start = max(0, pos - window)
        tail = bars.iloc[start:pos]
        if tail.empty:  # pragma: no cover - pos>0 guarantees a non-empty tail
            return False, "no volume history"
        adv = float((tail["close"] * tail["volume"]).mean())
        if not np.isfinite(adv) or adv < cfg.min_avg_dollar_volume:
            return False, f"adv {adv:.0f} < {cfg.min_avg_dollar_volume}"
    return True, None


@dataclass
class _SlotState:
    """Mutable state for a single slot during the event-driven simulation."""

    ticker: str
    entry_idx: int
    entry_date: Union[date, datetime]
    entry_fill: float
    signal_date: Union[date, datetime]
    rank: int
    stop_ref: Optional[float]
    target_ref: Optional[float]
    hold_limit_idx: int
    peak: float
    exit_signal: Optional[pd.Series]
    adv_shares: float = 0.0
    sigma_daily: float = 0.0
    half_spread: float = 0.0
    partial_targets: tuple[float, ...] = ()
    partial_fractions: tuple[float, ...] = ()
    partial_fired: list[bool] = field(default_factory=list)
    # Optional hot-loop accelerators set by _make_slot_state when a _RunCaches
    # is threaded through; direct constructors (tests) leave them None and the
    # per-bar helpers fall back to the original pandas access paths.
    frame_cache: Optional[_FrameCache] = None
    exit_signal_values: Optional[np.ndarray] = None
    # Session-last mask aligned to ``bars.index`` when ``cfg.intraday_only``.
    session_last: Optional[np.ndarray] = None
    # Size used by a liquidity-sensitive FillModel and passed unchanged to
    # Portfolio.open. Fixed-price models retain legacy portfolio sizing.
    # See docs/adr/0001-pre-impact-entry-sizing.md.
    entry_shares: float | None = None


def _half_spread_at_signal(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    frame_cache: Optional[_FrameCache] = None,
) -> float:
    """Corwin-Schultz half-spread at ``signal_idx`` when ``spread_proxy`` is on."""
    if not cfg.spread_proxy:
        return 0.0
    if signal_idx < 0 or signal_idx >= len(bars):
        return 0.0
    if "high" not in bars.columns or "low" not in bars.columns:
        return 0.0
    series: Optional[pd.Series] = None
    if frame_cache is not None and frame_cache.half_spread_s is not None:
        series = frame_cache.half_spread_s
    else:
        try:
            series = corwin_schultz_half_spread(bars["high"], bars["low"])
        except (TypeError, ValueError):
            return 0.0
        if frame_cache is not None:
            frame_cache.half_spread_s = series
    try:
        value = float(series.iloc[signal_idx])
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _make_slot_state(
    ticker: str,
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast,
    rank: int,
    fill_model: Optional[FillModel] = None,
    caches: Optional[_RunCaches] = None,
    entry_budget: Optional[float] = None,
) -> tuple[Optional[_SlotState], Optional[str]]:
    """Build the per-slot state used by both historical and rolling flows."""
    fills = fill_model if fill_model is not None else FillModel(cfg)
    frame_cache = caches.frame(ticker, bars) if caches is not None else None
    if fills.needs_liquidity_inputs:
        adv_shares, sigma_daily = (
            _cached_trailing_liquidity(frame_cache, bars, signal_idx)
            if frame_cache is not None
            else _trailing_liquidity(bars, signal_idx)
        )
    else:
        adv_shares, sigma_daily = 0.0, 0.0
    half_spread = _half_spread_at_signal(bars, signal_idx, cfg, frame_cache)
    entry_idx, entry_fill, entry_shares, entry_warn = fills.entry_quote(
        bars,
        signal_idx,
        budget=entry_budget,
        adv_shares=adv_shares,
        sigma_daily=sigma_daily,
        half_spread=half_spread,
        arrays=frame_cache,
    )
    if entry_idx is None or entry_fill is None:
        return None, entry_warn
    session_last = None
    if cfg.intraday_only:
        if frame_cache is not None and frame_cache.session_last is not None:
            session_last = frame_cache.session_last
        else:
            session_last = is_session_last(
                cast(pd.DatetimeIndex, bars.index), market_timezone(cfg.market)
            )
            if frame_cache is not None:
                frame_cache.session_last = session_last
        if session_last[entry_idx]:
            return None, "entry skipped: session-last bar (intraday-only)"
    exit_signal = None
    if exit_ast is not None:
        cached_exit = caches.exit_signals.get(ticker) if caches is not None else None
        if isinstance(cached_exit, str):
            return None, cached_exit
        if cached_exit is not None:
            exit_signal = cached_exit
        else:
            try:
                exit_signal = evaluate(exit_ast, bars).fillna(False).astype(bool)
            except PineError as exc:
                warn = f"exit eval failed: {exc}"
                if caches is not None:
                    caches.exit_signals[ticker] = warn
                return None, warn
            if caches is not None:
                caches.exit_signals[ticker] = exit_signal
    stop_ref = entry_fill * (1.0 - cfg.stop_loss) if cfg.stop_loss else None
    target_ref = entry_fill * (1.0 + cfg.take_profit) if cfg.take_profit else None
    partial_targets = tuple(
        entry_fill * (1.0 + pct) for pct, _frac in cfg.partial_exits
    )
    partial_fractions = tuple(frac for _pct, frac in cfg.partial_exits)
    return (
        _SlotState(
            ticker=ticker,
            entry_idx=entry_idx,
            entry_date=_bar_label(bars.index[entry_idx], cfg),
            entry_fill=entry_fill,
            signal_date=_bar_label(bars.index[signal_idx], cfg),
            rank=rank,
            stop_ref=stop_ref,
            target_ref=target_ref,
            hold_limit_idx=entry_idx + cfg.hold,
            peak=entry_fill,
            exit_signal=exit_signal,
            adv_shares=adv_shares,
            sigma_daily=sigma_daily,
            half_spread=half_spread,
            partial_targets=partial_targets,
            partial_fractions=partial_fractions,
            partial_fired=[False] * len(partial_targets),
            frame_cache=frame_cache,
            exit_signal_values=(
                exit_signal.to_numpy() if exit_signal is not None else None
            ),
            session_last=session_last,
            entry_shares=(entry_shares if fills.needs_liquidity_inputs else None),
        ),
        None,
    )


def _maybe_credit_dividends(
    portfolio: Portfolio,
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
) -> None:
    """Credit a cash dividend on an ex-date bar if the frame carries one."""
    if cfg.price_adjustment == "full":
        return
    frame_cache = state.frame_cache
    if frame_cache is not None:
        if frame_cache.dividend_arr is None:
            return
        try:
            div = float(frame_cache.dividend_arr[i])
        except (TypeError, ValueError):
            return
    else:
        if "dividend" not in bars.columns:
            return
        try:
            div = float(bars.iloc[i]["dividend"])
        except (KeyError, TypeError, ValueError):
            return
    if not math.isfinite(div) or div <= 0:
        return
    portfolio.credit_dividends(state.ticker, div)


def _fire_partial_exits_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
    portfolio: Portfolio,
    fill_model: FillModel,
) -> None:
    """Close configured partial-exit tranches if their target prices trade."""
    if not state.partial_targets:
        return
    pos = portfolio.get_position(state.ticker)
    if pos is None or pos.shares <= 0:
        return
    frame_cache = state.frame_cache
    if frame_cache is not None:
        bar_open = float(frame_cache.open_arr[i])
        high = float(frame_cache.high_arr[i])
    else:
        bar = bars.iloc[i]
        bar_open = float(bar["open"])
        high = float(bar["high"])
    bar_date = _bar_label(bars.index[i], cfg)
    for tier_idx, target_price in enumerate(state.partial_targets):
        if state.partial_fired[tier_idx] or high < target_price:
            continue
        fill = fill_model.exit_price(
            reason="target",
            bar_open=bar_open,
            level=target_price,
            shares=pos.shares * state.partial_fractions[tier_idx],
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
            half_spread=state.half_spread,
        )
        portfolio.partial_close(
            ticker=state.ticker,
            exit_date=bar_date,
            exit_price=fill,
            reason="target",
            fraction=state.partial_fractions[tier_idx],
        )
        state.partial_fired[tier_idx] = True
        if state.stop_ref is None or state.stop_ref < state.entry_fill:
            state.stop_ref = state.entry_fill


def _check_exit_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
    fill_model: FillModel,
    shares: float = 0.0,
) -> Optional[tuple[float, ExitReason]]:
    """Evaluate exit rules for ``state`` at ``bars[i]``."""
    frame_cache = state.frame_cache
    if frame_cache is not None:
        bar_open = float(frame_cache.open_arr[i])
        high = float(frame_cache.high_arr[i])
        low = float(frame_cache.low_arr[i])
        close = float(frame_cache.close_arr[i])
    else:
        bar = bars.iloc[i]
        bar_open = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

    trail_ref = state.peak * (1.0 - cfg.trailing_stop) if cfg.trailing_stop else None
    stop_hit = state.stop_ref is not None and low <= state.stop_ref
    target_hit = state.target_ref is not None and high >= state.target_ref
    trail_hit = trail_ref is not None and low <= trail_ref

    def _sell(reason: ExitReason, level: Optional[float] = None) -> float:
        return fill_model.exit_price(
            reason=reason,
            bar_open=bar_open,
            level=level,
            close=close,
            shares=shares,
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
            half_spread=state.half_spread,
        )

    if stop_hit and target_hit:
        return _sell("stop", state.stop_ref), "stop"
    if stop_hit:
        return _sell("stop", state.stop_ref), "stop"
    if trail_hit:
        return _sell("trail", trail_ref), "trail"
    if target_hit:
        return _sell("target", state.target_ref), "target"

    if high > state.peak:
        state.peak = high

    if state.exit_signal is not None:
        fired = (
            bool(state.exit_signal_values[i])
            if state.exit_signal_values is not None
            else bool(state.exit_signal.iloc[i])
        )
        if fired:
            return _sell("exit_expr"), "exit_expr"
    if state.session_last is not None and state.session_last[i]:
        return _sell("session"), "session"
    if i >= state.hold_limit_idx:
        return _sell("time"), "time"
    return None


_NO_UNIVERSE_MSG = (
    "No universe provided: pass --tickers or --universe-file. The TradingView "
    "current-screener fallback was removed because it injects survivorship bias "
    "(delisted or deleveraged tickers as of the signal date would be silently "
    "excluded)."
)


def _resolve_universe(cfg: UniverseSpec) -> tuple[list[str], list[str]]:
    """Return ``(tv_symbols, warnings)`` for the configured universe."""
    warnings: list[str] = []
    # Dynamic ADV ranking and snapshot membership windows do the real
    # selection downstream; pre-capping here would silently shrink the
    # candidate pool to the first ``max_universe`` names in loader order.
    skip_cap = cfg.dynamic_universe_size is not None or bool(cfg.membership_windows)

    def _cap(tickers: list[str]) -> list[str]:
        if skip_cap:
            return tickers
        max_universe = int(cfg.max_universe)
        if max_universe <= 0 or len(tickers) <= max_universe:
            return tickers
        warnings.append(
            f"capped universe from {len(tickers)} to {max_universe} tickers"
        )
        return tickers[:max_universe]

    if cfg.tickers:
        return _cap(list(cfg.tickers)), warnings
    if cfg.universe_file:
        from screener.universes import read_universe_file

        tickers = read_universe_file(cfg.universe_file, comment_prefixes=("#",))
        return _cap(tickers), warnings
    raise ValueError(_NO_UNIVERSE_MSG)


def prepare_strategy_bars(
    strategy: str | StrategySpec | None,
    bars_by_tv: dict[str, pd.DataFrame],
    price_panel: dict[str, pd.DataFrame],
    tv_symbols: list[str],
    start: date,
    end: date,
    fetcher: PriceFetcher,
    warnings: list[str],
    *,
    market: str,
    benchmark: str,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Resolve a strategy and run its preparation/lookback hooks."""
    from screener.strategies.spec import (
        ExpressionStrategySpec,
        PrepareCtx,
        resolve_strategy_spec,
    )

    try:
        spec = (
            resolve_strategy_spec(strategy) if isinstance(strategy, str) else strategy
        )
    except ValueError as exc:
        warnings.append(f"strategy error: {exc}")
        return bars_by_tv, 0
    if not isinstance(spec, ExpressionStrategySpec):
        return bars_by_tv, 0
    lookback_floor = spec.required_lookback() if spec.required_lookback else 0
    if spec.prepare_bars is None:
        return bars_by_tv, lookback_floor
    ctx = PrepareCtx(
        market=market,
        benchmark=benchmark,
        bars_by_tv=bars_by_tv,
        price_panel=price_panel,
        tv_symbols=tv_symbols,
        start=start,
        end=end,
        fetcher=fetcher,
        warnings=warnings,
    )
    return spec.prepare_bars(ctx), lookback_floor


def _benchmark_series_from_panel(
    price_panel: dict[str, pd.DataFrame], symbol: str
) -> pd.Series:
    """Return the benchmark close series from the already-fetched panel.

    The benchmark (``cfg.benchmark``) is fetched into ``price_panel`` alongside
    the portfolio symbols and, in the ``splits_only`` regime, split-adjusted by
    ``apply_splits_only_adjustment``. Reusing it here keeps the benchmark a single
    source of truth, consistent with the portfolio bars, instead of re-fetching it
    raw (which would inject a phantom split jump into alpha/beta/regime metrics).

    Shared by both the single-pass (``historical``) and rolling backtest flows so
    the fix applies identically on whichever path a strategy runs.
    """
    frame = price_panel.get(symbol)
    if frame is None or frame.empty:
        return pd.Series(
            index=pd.DatetimeIndex([], name="date"), dtype=float, name=symbol
        )
    series = frame["close"].astype(float).copy()
    series.name = symbol
    return series


def _eligible_reserve_signal_idx(
    bars: pd.DataFrame,
    exit_day: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    lookback: int,
    *,
    ticker: Optional[str] = None,
    caches: Optional[_RunCaches] = None,
) -> Optional[int]:
    """Return signal index if a reserve passes filters and entry AST on ``exit_day``.

    When ``ticker`` and ``caches`` are given, the entry AST is evaluated once
    over the full frame and reused across calls (positional indexing at the
    signal bar; every Pine op is causal, so the value matches the
    truncated-history evaluation — the same discipline the rolling engine's
    ``_precompute_entry_signals`` relies on).
    """
    # bars.index is a sorted DatetimeIndex; searchsorted is O(log n).
    pos = int(bars.index.searchsorted(exit_day, side="right")) - 1
    if pos < 0:
        return None
    if pos + 1 < lookback + 1:
        return None
    passes, _ = _passes_entry_filters(bars, exit_day, cfg)
    if not passes:
        return None
    if caches is not None and ticker is not None:
        signal = caches.entry_signal(ticker, bars, entry_ast)
        if isinstance(signal, PineError):
            return None
        if signal.empty:  # pragma: no cover - bars are non-empty here
            return None
        last = signal.iloc[pos]
    else:
        try:
            signal = evaluate(entry_ast, bars.iloc[: pos + 1])
        except PineError:
            return None
        if signal.empty:  # pragma: no cover - history is non-empty here
            return None
        last = signal.iloc[-1]
    if pd.isna(last) or not bool(last):
        return None
    return pos


def _bar_index_on_or_before(bars: pd.DataFrame, day: pd.Timestamp) -> Optional[int]:
    # ``bars.index`` is a sorted DatetimeIndex; searchsorted is O(log n).
    pos = bars.index.searchsorted(day, side="right")
    if pos <= 0:
        return None
    return int(pos - 1)


def _active_or_pending_tickers(
    slot_states: dict[int, Optional[_SlotState]],
) -> set[str]:
    return {state.ticker for state in slot_states.values() if state is not None}


def _precompute_entry_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    warnings: list[str],
    *,
    evaluated: dict[str, pd.Series | PineError] | None = None,
) -> dict[str, pd.Series]:
    results = (
        evaluated
        if evaluated is not None
        else evaluate_panel(entry_ast, bars_by_ticker)
    )
    signals: dict[str, pd.Series] = {}
    # Iterate the input order, not the panel's grouping order, so signal and
    # warning ordering stays byte-identical to the per-ticker loop.
    for ticker in bars_by_ticker:
        result = results.get(ticker)
        if result is None:
            continue
        if isinstance(result, PineError):
            warnings.append(f"entry eval failed: {ticker}: {result}")
            continue
        signals[ticker] = result.fillna(False).astype(bool)
    return signals


def _precompute_filter_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg: LiquidityFilterSpec,
) -> dict[str, pd.Series]:
    """Per-ticker boolean Series: True when min-price + ADV filters pass on that bar.

    Returns an empty dict (treated as a sentinel meaning "no filters configured")
    when both ``cfg.min_price`` and ``cfg.min_avg_dollar_volume`` are ``None`` —
    callers should fall back to the per-call behaviour in that case.
    """
    if cfg.min_price is None and cfg.min_avg_dollar_volume is None:
        return {}
    window = max(int(cfg.avg_dollar_volume_window), 1)

    # Group tickers whose bar indexes are interchangeable, then run the filter
    # once per group over a (bars x tickers) block instead of once per ticker.
    # The arithmetic is identical either way — what this removes is paying
    # pandas' fixed per-call cost (astype, Series construction, operator
    # dispatch, rolling setup) once per name. Same trick, and the same grouping
    # rule, as pine.evaluate_panel.
    groups: dict[object, list[str]] = {}
    solo: list[str] = []
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            continue
        key = panel_index_key(bars.index)
        if key is None:
            solo.append(ticker)
            continue
        groups.setdefault(key, []).append(ticker)

    out: dict[str, pd.Series] = {}
    for tickers in groups.values():
        out.update(_filter_signals_for_group(tickers, bars_by_ticker, cfg, window))
    for ticker in solo:
        out[ticker] = _filter_signals_for_group([ticker], bars_by_ticker, cfg, window)[
            ticker
        ]
    # Restore insertion order: callers build DataFrames straight from this dict,
    # and grouping visits tickers out of order.
    return {t: out[t] for t in bars_by_ticker if t in out}


def _filter_signals_for_group(
    tickers: list[str],
    bars_by_ticker: dict[str, pd.DataFrame],
    cfg: LiquidityFilterSpec,
    window: int,
) -> dict[str, pd.Series]:
    """Evaluate the price/ADV filters for one index-compatible group of tickers."""
    frames = [bars_by_ticker[t] for t in tickers]
    index = frames[0].index
    n_bars, n_tickers = len(index), len(tickers)

    # ``astype(float)`` rather than ``to_numpy(dtype=float)``: it is the exact
    # expression the per-ticker path used, and it yields a private copy, so the
    # block can never alias a caller's frame.
    close_block = np.empty((n_bars, n_tickers), dtype=float)
    for position, frame in enumerate(frames):
        close_block[:, position] = frame["close"].astype(float).to_numpy()

    # A non-finite close never passes either entry filter, including ADV-only.
    passes = np.isfinite(close_block)
    if cfg.min_price is not None:
        min_price = float(cfg.min_price)
        # A non-finite close never passes the price filter.
        passes &= (
            np.isfinite(close_block)
            & np.isfinite(min_price)
            & (close_block >= min_price)
        )
    if cfg.min_avg_dollar_volume is not None:
        dollar_vol = np.empty((n_bars, n_tickers), dtype=float)
        for position, frame in enumerate(frames):
            dollar_vol[:, position] = frame["volume"].astype(float).to_numpy()
        dollar_vol *= close_block
        # One columnwise rolling pass for the whole group; pandas applies it
        # per column, so each ticker sees exactly its own Series' result.
        adv = (
            pd.DataFrame(dollar_vol, index=index, columns=tickers, copy=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .to_numpy()
        )
        # NaN (or +-inf) ADV must fail the filter.
        passes &= np.isfinite(adv) & (adv >= float(cfg.min_avg_dollar_volume))

    return {
        ticker: pd.Series(passes[:, position], index=index)
        for position, ticker in enumerate(tickers)
    }
