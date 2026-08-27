"""Shared backtest primitives used by multiple backtest flows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

from screener.backtester.costs import corwin_schultz_half_spread
from screener.backtester.data import PriceFetcher
from screener.backtester.fills import FillModel
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import (
    PineError,
    evaluate,
    evaluate_panel,
    panel_index_key,
)
from screener.backtester.portfolio import Portfolio
from screener.backtester.sessions import is_session_last, market_timezone
from screener.ledger import ExitReason

if TYPE_CHECKING:
    from screener.strategies.spec import StrategySpec


class UniverseSpec(Protocol):
    """The config values universe resolution reads, and nothing else.

    Structural so both ``BacktestConfig`` and the panel-scoped input records in
    :mod:`screener.backtester.price_panel` satisfy it without either module
    having to know about the other.
    """

    @property
    def tickers(self) -> tuple[str, ...] | None: ...

    @property
    def universe_file(self) -> str | None: ...

    @property
    def membership_windows(self) -> tuple[tuple[str, date, date | None], ...]: ...

    @property
    def dynamic_universe_size(self) -> int | None: ...

    @property
    def max_universe(self) -> int: ...


class LiquidityFilterSpec(Protocol):
    """The config values the min-price/ADV entry filters read, and nothing else."""

    @property
    def min_price(self) -> float | None: ...

    @property
    def min_avg_dollar_volume(self) -> float | None: ...

    @property
    def avg_dollar_volume_window(self) -> int: ...


def _bar_label(ts, cfg: BacktestConfig) -> date | datetime:
    """Return the trade/position stamp for a bar timestamp.

    Daily bars are midnight-normalized, so a plain ``date`` is returned - this
    keeps the ledger byte-for-byte identical to the pre-intraday engine and
    comparable to the ``date``-typed test fixtures. Intraday bars return a full
    ``datetime`` so time-of-day survives into ``Trade``/``Position``.

    The daily path avoids ``pd.Timestamp(...)`` when the input is already a
    Timestamp/datetime/date: the day loop stamps every fill, so the constructor
    was pure overhead on the hot path.
    """
    if cfg.interval == "1d":
        if isinstance(ts, pd.Timestamp):
            return ts.date()
        # ``datetime`` is a subclass of ``date``; check it first so time-bearing
        # values still drop to a calendar date rather than passing through.
        if isinstance(ts, datetime):
            return ts.date()
        if isinstance(ts, date):
            return ts
        return pd.Timestamp(ts).date()
    return pd.Timestamp(ts).to_pydatetime()


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
    dividend_arr: np.ndarray | None
    # int64 epoch-nanosecond view of a unique, naive datetime64 index for
    # O(log n) day->position lookups against ``Timestamp.value`` (always ns).
    # None otherwise; callers keep ``Index.get_loc`` (duplicates, tz-aware,
    # exotic dtypes). Non-ns resolutions are converted via ``as_unit("ns")``.
    index_i8: np.ndarray | None
    volume_f: pd.Series
    # close[i] / close[i-1] - 1: the exact ``pct_change`` arithmetic for
    # NaN-free windows (NaN-containing windows fall back to the original).
    rets_f: pd.Series
    liquidity_by_idx: dict[tuple[int, int], tuple[float, float]] = field(
        default_factory=dict
    )
    # Lazily filled Corwin-Schultz half-spread series (fraction), aligned to
    # ``bars.index``. None until first ``spread_proxy`` lookup for this frame.
    half_spread_s: pd.Series | None = None
    # Lazily filled session-last mask (see ``sessions.is_session_last``); None
    # until the first ``intraday_only`` slot open on this frame.
    session_last: np.ndarray | None = None


def _build_frame_cache(bars: pd.DataFrame) -> _FrameCache:
    close_arr = bars["close"].to_numpy(dtype=float)
    index = bars.index
    index_i8: np.ndarray | None = None
    if (
        isinstance(index, pd.DatetimeIndex)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.datetime64)
        and index.is_unique
    ):
        # day_loop compares against Timestamp.value, which is always nanoseconds.
        # Pandas 3 defaults many calendars to datetime64[us]; viewing those as i8
        # without converting would make every searchsorted miss.
        index_i8 = index.as_unit("ns").to_numpy().view("i8")
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
    produced). ``exit_signal_values`` mirrors successful exit Series as a bool
    ndarray so the per-bar exit check never re-boxes the Series.
    """

    entry_signals: dict[str, pd.Series | PineError] = field(default_factory=dict)
    exit_signals: dict[str, pd.Series | str] = field(default_factory=dict)
    exit_signal_values: dict[str, np.ndarray] = field(default_factory=dict)
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

    def store_exit_signal(self, ticker: str, signal: pd.Series) -> pd.Series:
        """Normalise, cache, and return a boolean exit Series for ``ticker``."""
        normalised = signal.fillna(False).astype(bool)
        self.exit_signals[ticker] = normalised
        # Contiguous bool buffer for the hot exit path; copy so a later Series
        # mutation cannot silently alias into a live slot state.
        self.exit_signal_values[ticker] = np.asarray(normalised.to_numpy(), dtype=bool)
        return normalised

    def prewarm_exit_signals(
        self,
        bars_by_ticker: dict[str, pd.DataFrame],
        exit_ast,
        *,
        evaluated: dict[str, pd.Series | np.ndarray | PineError] | None = None,
    ) -> None:
        """Fill ``exit_signals`` for every ticker in one panel pass.

        ``_make_slot_state`` otherwise evaluates the exit AST on the first slot
        open per ticker - correct, but one interpreted walk each. Batching them
        up front costs one walk per index-group for the whole universe, which is
        cheaper even though not every ticker ends up being traded.

        Entries are stored in exactly the form the lazy path stores: the
        normalised boolean Series, or the warning string a failure produced, so
        slot-open behaviour and warning text are unchanged. Bool ndarrays are
        cached alongside so the first open does not pay ``Series.to_numpy``.
        When ``evaluated`` already holds bool ndarrays (panel path with
        ``as_bool_array=True``), rebuild only the thin Series wrapper needed by
        the public cache type - the ndarray is stored directly.
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
                continue
            if isinstance(result, np.ndarray):
                bars = bars_by_ticker[ticker]
                values = (
                    result if result.dtype == bool else np.asarray(result, dtype=bool)
                )
                series = pd.Series(values, index=bars.index, dtype=bool, copy=False)
                self.exit_signals[ticker] = series
                self.exit_signal_values[ticker] = values
                continue
            self.store_exit_signal(ticker, result)


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
) -> tuple[bool, str | None]:
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
    entry_date: date | datetime
    entry_fill: float
    signal_date: date | datetime
    rank: int
    stop_ref: float | None
    target_ref: float | None
    hold_limit_idx: int
    peak: float
    exit_signal: pd.Series | None
    adv_shares: float = 0.0
    sigma_daily: float = 0.0
    half_spread: float = 0.0
    partial_targets: tuple[float, ...] = ()
    partial_fractions: tuple[float, ...] = ()
    partial_fired: list[bool] = field(default_factory=list)
    # Optional hot-loop accelerators set by _make_slot_state when a _RunCaches
    # is threaded through; direct constructors (tests) leave them None and the
    # per-bar helpers fall back to the original pandas access paths.
    frame_cache: _FrameCache | None = None
    exit_signal_values: np.ndarray | None = None
    # Session-last mask aligned to ``bars.index`` when ``cfg.intraday_only``.
    session_last: np.ndarray | None = None
    # Size used by a liquidity-sensitive FillModel and passed unchanged to
    # Portfolio.open. Fixed-price models retain legacy portfolio sizing.
    # See docs/adr/0001-pre-impact-entry-sizing.md.
    entry_shares: float | None = None
    # Rank-exit rebalance flag (rolling engine): set between sweeps when the
    # ticker left the prior bar's top-N ranking; the shared exit sweep then
    # closes the slot through the normal dividend -> partial -> exit sequence.
    rank_exit: bool = False


def _half_spread_at_signal(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    frame_cache: _FrameCache | None = None,
) -> float:
    """Corwin-Schultz half-spread at ``signal_idx`` when ``spread_proxy`` is on."""
    if not cfg.spread_proxy:
        return 0.0
    if signal_idx < 0 or signal_idx >= len(bars):
        return 0.0
    if "high" not in bars.columns or "low" not in bars.columns:
        return 0.0
    series: pd.Series | None = None
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
    fill_model: FillModel | None = None,
    caches: _RunCaches | None = None,
    entry_budget: float | None = None,
) -> tuple[_SlotState | None, str | None]:
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
    exit_signal_values: np.ndarray | None = None
    if exit_ast is not None:
        cached_exit = caches.exit_signals.get(ticker) if caches is not None else None
        if isinstance(cached_exit, str):
            return None, cached_exit
        if cached_exit is not None:
            exit_signal = cached_exit
            if caches is not None:
                exit_signal_values = caches.exit_signal_values.get(ticker)
        else:
            try:
                raw_exit = evaluate(exit_ast, bars)
            except PineError as exc:
                warn = f"exit eval failed: {exc}"
                if caches is not None:
                    caches.exit_signals[ticker] = warn
                return None, warn
            if caches is not None:
                exit_signal = caches.store_exit_signal(ticker, raw_exit)
                exit_signal_values = caches.exit_signal_values[ticker]
            else:
                exit_signal = raw_exit.fillna(False).astype(bool)
        if exit_signal_values is None and exit_signal is not None:
            exit_signal_values = np.asarray(exit_signal.to_numpy(), dtype=bool)
            if caches is not None:
                caches.exit_signal_values[ticker] = exit_signal_values
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
            exit_signal_values=exit_signal_values,
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
) -> tuple[float, ExitReason] | None:
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

    def _sell(reason: ExitReason, level: float | None = None) -> float:
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

    # Rank-exit rebalance membership rule: protective price exits above win,
    # discretionary/session/time exits below lose to it.
    if state.rank_exit:
        return _sell("rank"), "rank"

    state.peak = max(state.peak, high)

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


def strategy_required_lookback(
    strategy: str | StrategySpec | None,
) -> int:
    """Return the bars a named strategy needs before its prepared columns exist.

    Fetch windows sized from the entry and exit expressions miss this floor.
    Strategies such as a Bollinger Band breakout write ``bb_upper`` in
    ``prepare_bars`` (about 350 prior bars) while the expression only names
    the column. Too little history leaves the indicator as NaN and silently
    drops valid trades.
    """
    from screener.strategies.spec import (
        ExpressionStrategySpec,
        resolve_strategy_spec,
    )

    if strategy is None:
        return 0
    try:
        spec = (
            resolve_strategy_spec(strategy) if isinstance(strategy, str) else strategy
        )
    except ValueError:
        return 0
    if not isinstance(spec, ExpressionStrategySpec) or spec.required_lookback is None:
        return 0
    return int(spec.required_lookback())


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
    lookback_floor = strategy_required_lookback(spec)
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
    ticker: str | None = None,
    caches: _RunCaches | None = None,
) -> int | None:
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


def _bar_index_on_or_before(bars: pd.DataFrame, day: pd.Timestamp) -> int | None:
    # ``bars.index`` is a sorted DatetimeIndex; searchsorted is O(log n).
    pos = bars.index.searchsorted(day, side="right")
    if pos <= 0:
        return None
    return int(pos - 1)


def _active_or_pending_tickers(
    slot_states: dict[int, _SlotState | None],
) -> set[str]:
    return {state.ticker for state in slot_states.values() if state is not None}


def _precompute_entry_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    warnings: list[str],
    *,
    evaluated: dict[str, pd.Series | np.ndarray | PineError] | None = None,
) -> dict[str, pd.Series | np.ndarray]:
    """Return per-ticker entry masks as Series or bool ndarrays.

    When the panel evaluator is asked for ``as_bool_array=True``, values are
    already bool ndarrays aligned to each ticker's bar index — keep them that
    way so :func:`_build_rolling_candidate_matrices` can assemble signal_mat
    without a Series detour. Series results (solo path, legacy callers) are
    normalised with ``fillna(False).astype(bool)`` as before.
    """
    results = (
        evaluated
        if evaluated is not None
        else evaluate_panel(entry_ast, bars_by_ticker)
    )
    signals: dict[str, pd.Series | np.ndarray] = {}
    # Iterate the input order, not the panel's grouping order, so signal and
    # warning ordering stays byte-identical to the per-ticker loop.
    for ticker in bars_by_ticker:
        result = results.get(ticker)
        if result is None:
            continue
        if isinstance(result, PineError):
            warnings.append(f"entry eval failed: {ticker}: {result}")
            continue
        if isinstance(result, np.ndarray):
            if result.dtype == bool:
                signals[ticker] = result
            else:
                signals[ticker] = np.asarray(result, dtype=bool)
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


def _rolling_mean_min_periods_1(values: np.ndarray, window: int) -> np.ndarray:
    """Column-wise rolling mean with ``min_periods=1``, matching pandas skipna.

    Used by :func:`_filter_signals_for_group` so the ADV filter does not pay
    DataFrame construction + ``rolling().mean()`` dispatch for every panel
    group. Window edges shorter than ``window`` average the available prefix;
    non-finite inputs are excluded from the mean (pandas ``Series.mean`` /
    rolling default), and a window with zero finite observations yields NaN.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    finite = np.isfinite(values)
    # Zero out non-finite so cumsum stays well-defined; count them separately.
    filled = np.where(finite, values, 0.0)
    count = finite.astype(np.float64, copy=False)
    sum_cs = np.cumsum(filled, axis=0)
    count_cs = np.cumsum(count, axis=0)
    sum_win = sum_cs.copy()
    count_win = count_cs.copy()
    if window < values.shape[0]:
        sum_win[window:] = sum_cs[window:] - sum_cs[:-window]
        count_win[window:] = count_cs[window:] - count_cs[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(sum_win / count_win, dtype=float)
    out[count_win < 1.0] = np.nan
    return out


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

    # Write into a private block via ``to_numpy(dtype=float)``. The destination
    # array is ours, so a view into an already-float64 caller column is fine
    # (nothing mutates through the block). Avoids a redundant ``astype`` copy.
    close_block = np.empty((n_bars, n_tickers), dtype=float)
    for position, frame in enumerate(frames):
        close_block[:, position] = frame["close"].to_numpy(dtype=float, copy=False)

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
            dollar_vol[:, position] = frame["volume"].to_numpy(dtype=float, copy=False)
        dollar_vol *= close_block
        # One columnwise rolling pass for the whole group; arithmetic matches
        # pandas ``rolling(window, min_periods=1).mean()`` (skipna).
        adv = _rolling_mean_min_periods_1(dollar_vol, window)
        # NaN (or +-inf) ADV must fail the filter.
        passes &= np.isfinite(adv) & (adv >= float(cfg.min_avg_dollar_volume))

    return {
        ticker: pd.Series(passes[:, position], index=index)
        for position, ticker in enumerate(tickers)
    }
