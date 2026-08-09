"""Price-panel knowledge: what bars does a run actually see?

This module owns one question and every input to it: universe resolution, the
warmup window, fetching, split adjustment, strategy bar preparation, and the
fundamental/option columns merged onto those bars. It never sees a
:class:`~screener.backtester.models.BacktestConfig`; it is handed a
:class:`PricePanelInputs`, which is the complete, explicit list of config
values a panel's contents depend on.

That is what makes the reuse fingerprint in
``screener.backtester.rolling_simulation`` derivable rather than
hand-maintained: the fingerprint reads :data:`PRICE_PANEL_CONFIG_FIELDS`,
which is computed from :class:`PricePanelInputs` itself, so a value this
module consumes cannot be missing from the key.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from screener.backtester.core import (
    _benchmark_series_from_panel,
    _resolve_universe,
    prepare_strategy_bars,
    strategy_lookback_floor,
)
from screener.backtester.data import PriceFetcher
from screener.backtester.fundamentals import (
    FundamentalFetcher,
    merge_fundamentals_into_bars,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import Node
from screener.backtester.warmup import _warmup_days_for_interval
from screener.options.backtest import merge_referenced_options


@dataclass(frozen=True)
class PricePanelInputs:
    """Every config value the fetched/prepared bars depend on.

    Field names match :class:`~screener.backtester.models.BacktestConfig`
    field names so :data:`PRICE_PANEL_CONFIG_FIELDS` can be derived from this
    class instead of restated by hand.
    """

    market: str
    benchmark: str
    tickers: tuple[str, ...] | None
    universe_file: str | None
    membership_windows: tuple[tuple[str, date, date | None], ...]
    dynamic_universe_size: int | None
    max_universe: int
    interval: str
    price_adjustment: Literal["full", "splits_only", "none"]
    strategy_name: str | None
    fundamentals_provider: str | None

    @classmethod
    def from_config(cls, cfg: BacktestConfig) -> PricePanelInputs:
        return cls(
            market=cfg.market,
            benchmark=cfg.benchmark,
            tickers=cfg.tickers,
            universe_file=cfg.universe_file,
            membership_windows=cfg.membership_windows,
            dynamic_universe_size=cfg.dynamic_universe_size,
            max_universe=cfg.max_universe,
            interval=cfg.interval,
            price_adjustment=cfg.price_adjustment,
            strategy_name=cfg.strategy_name,
            fundamentals_provider=cfg.fundamentals_provider,
        )


# Values the panel does not read itself but whose bars still depend on them:
# the caller derives the simulation window from ``as_of``, and the two
# ``fundamental_*`` knobs shape the injected ``FundamentalFetcher`` whose
# values are merged into the bars below.
_INDIRECT_PRICE_PANEL_FIELDS = frozenset(
    {"as_of", "fundamental_fields", "fundamental_lag_days"}
)

PRICE_PANEL_CONFIG_FIELDS = (
    frozenset(f.name for f in fields(PricePanelInputs)) | _INDIRECT_PRICE_PANEL_FIELDS
)


@dataclass(frozen=True)
class PricePanel:
    """Bars, benchmark and trading calendar for one simulation window."""

    tv_symbols: list[str]
    yf_by_tv: dict[str, str]
    bars_by_tv: dict[str, pd.DataFrame]
    benchmark: pd.Series
    lookback: int
    master_dates: list[pd.Timestamp]


def _master_dates(
    bars_by_tv: dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Sorted union of every ticker's bar stamps inside the window."""
    day_arrays: list[np.ndarray] = []
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        idx = bars.index
        mask = (idx >= start_ts) & (idx <= end_ts)
        if mask.any():
            day_arrays.append(idx[mask].to_numpy())
    if not day_arrays:
        return []
    return list(pd.DatetimeIndex(np.unique(np.concatenate(day_arrays))))


def build_price_panel(
    inputs: PricePanelInputs,
    fetcher: PriceFetcher,
    *,
    entry_ast: Node,
    exit_ast: Node | None,
    lookback: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warnings: list[str],
    fundamental_fetcher: FundamentalFetcher | None = None,
) -> PricePanel:
    """Resolve the universe, fetch its bars and merge every bar-shaping source.

    ``entry_ast``/``exit_ast``/``lookback`` come from the signal side: the panel
    does not parse expressions, it is told how much warmup history to buy and
    which option legs the expressions reference.
    """
    from screener.backtester.data import tv_to_yf

    tv_symbols, univ_warnings = _resolve_universe(inputs)
    warnings.extend(univ_warnings)
    yf_by_tv = {tv: tv_to_yf(tv, inputs.market) for tv in tv_symbols}
    yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [inputs.benchmark]))

    # Warmup is measured in BARS (enough history for the longest indicator).
    # For daily bars one bar ~ one calendar day, so the legacy day-based padding
    # stands. For intraday, convert the required warmup bars into calendar days
    # via bars-per-session (with slack for weekends/holidays) so we don't request
    # ~365 days of minute data - which both blows past yfinance's intraday cap
    # and is unnecessary. Chunking longer intraday windows is Phase 2.
    #
    # The expression AST is not the whole story. A strategy that builds its own
    # columns in ``prepare_bars`` reads them as bare names ("mom_12_1 > 0"), so
    # the parser measures a lookback of zero and this would buy the 365-day
    # floor. The eligibility gate downstream still demands the strategy's
    # declared lookback, so the shortfall is not an error - it silently eats the
    # front of the backtest window, which is worse. Ask the spec first.
    lookback = max(lookback, strategy_lookback_floor(inputs.strategy_name))
    warmup_days = _warmup_days_for_interval(lookback, inputs.interval)
    fetch_start = (start_ts - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_ts.date()
    price_panel = fetcher.fetch(yf_symbols, fetch_start, fetch_end)

    if inputs.price_adjustment == "splits_only":
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
        inputs.strategy_name,
        bars_by_tv,
        price_panel,
        tv_symbols,
        fetch_start,
        fetch_end,
        fetcher,
        warnings,
        market=inputs.market,
        benchmark=inputs.benchmark,
    )
    effective_lookback = max(lookback, strategy_lookback)

    if fundamental_fetcher is not None:
        fundamentals = fundamental_fetcher.fetch(
            yf_by_tv.values(), fetch_start, fetch_end
        )
        bars_by_tv = merge_fundamentals_into_bars(bars_by_tv, fundamentals, yf_by_tv)

    bars_by_tv = merge_referenced_options(
        bars_by_tv,
        market=inputs.market,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        warnings=warnings,
    )

    # Reuse the benchmark already fetched into ``price_panel`` (it is included in
    # ``yf_symbols`` above and split-adjusted alongside the portfolio symbols in
    # ``splits_only`` mode). Fetching it raw here would reintroduce the phantom
    # split jump into the regime gate, the aligned curve, and regime metrics.
    benchmark = _benchmark_series_from_panel(price_panel, inputs.benchmark)

    return PricePanel(
        tv_symbols=tv_symbols,
        yf_by_tv=yf_by_tv,
        bars_by_tv=bars_by_tv,
        benchmark=benchmark,
        lookback=effective_lookback,
        master_dates=_master_dates(bars_by_tv, start_ts, end_ts),
    )
