"""The screen's half of the unified candidate layer.

A criterion name used to mean two things at once: a set of TradingView filters
and a ranking recipe over the row those filters returned. For the three names
that also exist as a strategy - ``breakout``, ``momentum_12_1`` and
``mark_minervini`` - that meant two implementations of one rule, which is the
defect this whole line of work removes.

Those names are now *aliases* onto ``screener.strategies.spec.registry``. The
rule is the strategy's entry expression, evaluated over local bars through
:func:`screener.backtester.signal_panel.build_day_candidates` - the same
function, over the same matrices, that the rolling backtester ranks with. The
TradingView filters survive only as the strategy's declared
``profile.tv_prefilter``: a field cut that runs before bars are downloaded and
may only ever remove names the bar rule would have removed anyway.

Two universe modes, and the mode is recorded in the run label:

* **default** - TradingView answers "which names are worth downloading", then
  the bar rule decides which of them are candidates. Fast, and the field cut
  is the vendor's.
* **``--universe``** - the exact path. The universe comes from
  ``screener.universes`` and no prefilter runs, so the answer depends on no
  vendor field at all. Slower, because every name in the universe is fetched.

A criterion with no strategy behind it keeps its old snapshot behaviour
untouched; this module returns ``None`` for it and the workflow takes the path
it always took.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from screener.criteria import registry as criteria_registry
from screener.markets import get_market
from screener.scoring.bar_scores import (
    DEFAULT_PRICE_ADJUSTMENT,
    PERCENTILE_SCALE,
    PriceAdjustment,
)
from screener.scoring.components import percentile
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    ExpressionStrategySpec,
    StrategySpec,
)
from screener.strategies.spec import registry as strategy_registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.fundamentals import FundamentalFetcher
    from screener.backtester.signal_panel import Candidate

#: Calendar days of window handed to the candidate layer. The screen asks for
#: one date, but that date can be a weekend or a holiday, so the window has to
#: hold at least one bar for the as-of snap-back to land on.
_WINDOW_SLACK_DAYS = 15

#: Window used instead when the strategy reads fundamentals. A provider frame
#: is clipped to the fetch window before it is merged, so a window shorter than
#: a reporting cycle can contain no filing at all and forward-fill nothing -
#: which reads downstream as "the fundamental gate failed", not as "no data".
#: Five quarters always contains at least one filing.
_FUNDAMENTAL_WINDOW_DAYS = 460

#: Names a bar-path screen must produce for history and display. ``save_run``
#: reads ``name`` as the ticker, so it is not optional.
_DISPLAY_COLUMNS = (
    "name",
    "description",
    "close",
    "change",
    "volume",
    "market_cap_basic",
)

OUTPUT_SCORE_COLUMN = "setup_score"


class UnscreenableStrategyError(ValueError):
    """Raised when a strategy has no expression the candidate layer can run.

    Four strategies generate trades with a backward search that carries state
    between bars rather than with a per-bar boolean (``heikin_ashi``,
    ``shooting_star``, ``bb_pattern``, ``rsi_pattern``). They have no entry
    expression, so there is nothing for the shared candidate layer to evaluate.
    Refusing them by *kind* rather than by name means a future callable-only
    strategy is refused too, instead of silently screening as something else.
    """


@dataclass(frozen=True)
class ScreenStrategy:
    """A criterion name resolved onto the strategy it aliases."""

    criterion: str
    spec: ExpressionStrategySpec

    @property
    def tv_prefilter(self) -> str | None:
        """The criterion whose filters cut the field, or ``None``."""
        profile = self.spec.profile or DEFAULT_STRATEGY_PROFILE
        return profile.tv_prefilter


def aliased_strategy(criterion: str) -> StrategySpec | None:
    """The strategy a criterion name aliases, or ``None`` for a filters-only name."""
    return strategy_registry.get_optional(criterion)


def ensure_screenable(spec: StrategySpec) -> ExpressionStrategySpec:
    """Narrow ``spec`` to the expression shape the candidate layer needs."""
    if isinstance(spec, ExpressionStrategySpec):
        return spec
    raise UnscreenableStrategyError(
        f"strategy {spec.name!r} is callable-only: its trades come from a "
        "stateful backward search over bars, not from a per-bar entry "
        "expression, so the shared candidate layer cannot evaluate it. Screen "
        "it through the backtester's own callable path instead."
    )


def resolve_screen_strategy(names: Sequence[str]) -> ScreenStrategy | None:
    """Resolve criterion names onto one strategy, or ``None`` for the old path.

    Combining names with ``-c a -c b`` means "intersect these filter sets",
    which has no meaning once a name carries a whole entry expression: two
    expressions do not compose into one rule. So a strategy alias may only be
    screened on its own.
    """
    selected = tuple(names)
    aliased = [name for name in selected if aliased_strategy(name) is not None]
    if not aliased:
        return None
    if len(selected) > 1:
        others = sorted(set(selected) - set(aliased))
        # With two strategy aliases the difference is empty, and "cannot be
        # combined with []" reads as a bug rather than as the refusal it is.
        combined = f"with {others}" if others else "with each other"
        raise UnscreenableStrategyError(
            f"criteria {sorted(aliased)} name strategies, which carry a whole "
            "entry rule rather than a filter set, so they cannot be combined "
            f"{combined}. Screen one at a time."
        )
    name = aliased[0]
    spec = aliased_strategy(name)
    assert spec is not None  # aliased was built from a non-None lookup
    return ScreenStrategy(criterion=name, spec=ensure_screenable(spec))


def screen_label(
    names: Sequence[str],
    *,
    strategy: ScreenStrategy | None,
    universe: str | None,
) -> str:
    """The ``runs.criteria`` label, which records the semantics of the run.

    A filters-only run keeps the label it always had, so its history is
    continuous. A bar-rule run gets a new one, because the same name now
    answers a different question and a diff must never cross that change (D17
    in docs/plans/unify-screen-backtest.md). The universe mode is part of the
    label for the same reason: the two modes see different fields, so their
    added/removed diffs are not comparable (D9).
    """
    joined = "+".join(names)
    if strategy is None:
        return joined
    return f"{joined}@universe:{universe}" if universe else f"{joined}@tv"


def prefilter_filters(strategy: ScreenStrategy) -> list:
    """The TradingView filters that cut the field for ``strategy``.

    Empty when the strategy declares no prefilter, which means the scan is
    unfiltered and the bar rule does all the work.
    """
    name = strategy.tv_prefilter
    if name is None:
        return []
    return list(criteria_registry.get(name)())


def resolve_universe_tickers(universe: str, market: str) -> list[str]:
    """Resolve a named universe or a universe file into TradingView symbols."""
    from screener.universes import (
        UniverseRequest,
        UniverseSource,
        available_universes,
        resolve_universe,
    )

    is_index = universe in available_universes()
    source = UniverseSource.INDEX_CURRENT if is_index else UniverseSource.FILE
    request = UniverseRequest(
        source=source,
        market=market,
        index_name=universe if is_index else None,
        file=None if is_index else universe,
        comment_prefixes=("#",),
    )
    return resolve_universe(request)


def _bar_display_row(bars: pd.DataFrame, ticker: str) -> dict[str, object]:
    """Display columns for a ticker the TradingView scan never returned.

    ``--universe`` mode runs no scan, so there is no snapshot row to show. The
    values come from the same bars the rule was evaluated on, which is the
    honest source: ``market_cap_basic`` has no bar equivalent and stays NaN
    rather than being invented.
    """
    close = float(bars["close"].iloc[-1])
    previous = float(bars["close"].iloc[-2]) if len(bars) > 1 else close
    change = (close / previous - 1.0) * 100.0 if previous else float("nan")
    return {
        "ticker": ticker,
        # ``name`` is the plain symbol everywhere else - history keys on it and
        # both enrichment paths look it up by it - so the exchange prefix that
        # only the bar path carries has to come off here, or ``--earnings``
        # answers None for every row and saved runs key differently from every
        # other screen.
        "name": ticker.split(":", 1)[-1],
        "description": "",
        "close": close,
        "change": change,
        "volume": float(bars["volume"].iloc[-1]),
        "market_cap_basic": float("nan"),
    }


def _scores(candidates: Sequence[Candidate]) -> pd.Series:
    """``setup_score`` for one day's candidates, on the usual 0-100 scale.

    The percentile is of whatever the candidate layer actually ranked on,
    which :attr:`Candidate.rank_basis` names, so the column keeps agreeing
    with the ordering instead of asserting a scale of its own.
    """
    values = pd.Series(
        [
            c.rank_score if c.rank_score is not None else c.as_of_dollar_vol
            for c in candidates
        ],
        index=[c.ticker for c in candidates],
        dtype=float,
    )
    return percentile(values) * PERCENTILE_SCALE


def screen_candidates(
    strategy: ScreenStrategy,
    *,
    market: str,
    tickers: Sequence[str],
    as_of: date,
    scanned: pd.DataFrame | None = None,
    limit: int | None = None,
    order_by: str | None = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    strict: bool = False,
    warnings: list[str],
) -> pd.DataFrame:
    """Rank ``tickers`` by ``strategy``'s entry rule as of ``as_of``.

    Builds the price panel and the signal panel exactly the way
    ``prepare_rolling_backtest`` does and reads the ``as_of`` row out of it, so
    the rows returned here are the rows the rolling engine would have entered
    on that date. That identity is the point of the whole exercise and is
    pinned by ``tests/correctness`` rather than argued for here.

    ``scanned`` carries the TradingView snapshot rows in default mode, so the
    display columns stay the ones the screen has always shown. In
    ``--universe`` mode it is ``None`` and the display columns come from bars.

    ``strict`` is forwarded with ``refresh`` to the price fetcher: this whole
    path ranks off bars, so serving a stale panel here would rank a run on
    bars that were not actually refreshed. ``strict`` without ``refresh`` is a
    no-op, exactly as on the bar-score path.

    ``limit`` and ``order_by`` are applied to the finished frame rather than
    inside the candidate layer, so ``setup_score`` is a percentile of every
    candidate the rule found instead of a percentile of the top ``limit`` -
    which would make the same name score differently at ``-n 10`` and
    ``-n 100``.
    """
    from screener.backtester.data import build_price_fetcher
    from screener.backtester.price_panel import PricePanelInputs, build_price_panel
    from screener.backtester.signal_panel import (
        SignalPanelInputs,
        build_day_candidates,
        parse_signal_program,
    )

    if not tickers:
        return pd.DataFrame(columns=[*_DISPLAY_COLUMNS, OUTPUT_SCORE_COLUMN])

    profile = strategy.spec.profile or DEFAULT_STRATEGY_PROFILE
    venue = get_market(market)
    entry_expr = profile.entry_expr or strategy.spec.entry
    exit_expr = profile.exit_expr or strategy.spec.exit
    fundamental_fetcher, fundamentals_provider = _fundamentals_for(
        entry_expr, exit_expr, market=market, refresh=refresh
    )
    end_ts = pd.Timestamp(as_of)
    start_ts = end_ts - pd.Timedelta(days=_window_days(fundamental_fetcher))

    signal_inputs = SignalPanelInputs(
        market=market,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        regime_filter=profile.regime_filter,
        earnings_blackout_days=profile.earnings_blackout_days,
        sector_neutral=profile.sector_neutral,
        min_price=profile.min_price,
        min_avg_dollar_volume=profile.min_avg_dollar_volume,
        avg_dollar_volume_window=profile.avg_dollar_volume_window,
        membership_added=(),
        membership_windows=(),
        dynamic_universe_size=None,
        dynamic_universe_lookback=0,
        dynamic_universe_rebalance="never",
    )
    program = parse_signal_program(signal_inputs)

    panel_inputs = PricePanelInputs(
        market=market,
        benchmark=venue.benchmark,
        tickers=tuple(tickers),
        universe_file=None,
        membership_windows=(),
        dynamic_universe_size=None,
        max_universe=len(tickers),
        interval="1d",
        price_adjustment=price_adjustment,
        strategy_name=strategy.spec.name,
        fundamentals_provider=fundamentals_provider,
    )
    fetcher = build_price_fetcher(
        auto_adjust=(price_adjustment == "full"), refresh=refresh, strict=strict
    )
    panel = build_price_panel(
        panel_inputs,
        fetcher,
        entry_ast=program.entry_ast,
        exit_ast=program.exit_ast,
        lookback=program.lookback,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        fundamental_fetcher=fundamental_fetcher,
    )
    day = build_day_candidates(
        signal_inputs,
        panel,
        program=program,
        as_of=as_of,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
        limit=None,
    )
    return _candidate_frame(
        day.candidates,
        panel.bars_by_tv,
        scanned,
        limit=limit,
        order_by=order_by,
        warnings=warnings,
    )


def _candidate_frame(
    candidates: Sequence[Candidate],
    bars_by_tv: dict[str, pd.DataFrame],
    scanned: pd.DataFrame | None,
    *,
    limit: int | None = None,
    order_by: str | None = None,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Render candidates as the screen's result frame, in rank order.

    ``order_by`` re-sorts the finished rows the way the snapshot path lets
    ``--sort`` re-sort a scan; the rule still decides membership, so only the
    presentation order changes. ``limit`` is applied last, after the score, so
    the score does not depend on it.
    """
    if not candidates:
        return pd.DataFrame(columns=[*_DISPLAY_COLUMNS, OUTPUT_SCORE_COLUMN])
    scores = _scores(candidates)
    order = [c.ticker for c in candidates]

    if scanned is not None and not scanned.empty and "ticker" in scanned.columns:
        rows = scanned.set_index("ticker").reindex(order).reset_index()
    else:
        rows = pd.DataFrame(
            [
                _bar_display_row(bars_by_tv[t], t)
                for t in order
                if t in bars_by_tv and not bars_by_tv[t].empty
            ]
        )
        if rows.empty:
            return pd.DataFrame(columns=[*_DISPLAY_COLUMNS, OUTPUT_SCORE_COLUMN])
        order = list(rows["ticker"])
    rows[OUTPUT_SCORE_COLUMN] = [float(scores[t]) for t in order]
    rows = _sorted_rows(rows, order_by, warnings)
    if limit is not None:
        rows = rows.head(limit)
    return rows.reset_index(drop=True)


def _sorted_rows(
    rows: pd.DataFrame, order_by: str | None, warnings: list[str] | None
) -> pd.DataFrame:
    """Order the finished rows by ``order_by``, descending.

    ``setup_score`` and ``None`` both mean "keep rank order": the rows arrive
    ranked by exactly what the score is a percentile of, so re-sorting on it
    would only reorder ties. Any other name is a column of the display frame -
    a TradingView column in default mode - and is refused loudly rather than
    silently ignored when the frame does not carry it, which is the whole
    failure this replaces.
    """
    if order_by is None or order_by == OUTPUT_SCORE_COLUMN:
        return rows
    if order_by not in rows.columns:
        if warnings is not None:
            warnings.append(
                f"--sort {order_by} is not a column of this screen's results, "
                "so the rows keep the rule's own rank order."
            )
        return rows
    return rows.sort_values(order_by, ascending=False, kind="stable")


def _window_days(fundamental_fetcher: FundamentalFetcher | None) -> int:
    """Calendar days of window to build the panel over."""
    return (
        _WINDOW_SLACK_DAYS
        if fundamental_fetcher is None
        else max(_WINDOW_SLACK_DAYS, _FUNDAMENTAL_WINDOW_DAYS)
    )


def _fundamentals_for(
    entry_expr: str | None,
    exit_expr: str | None,
    *,
    market: str,
    refresh: bool,
) -> tuple[FundamentalFetcher | None, str | None]:
    """The fundamentals fetcher this strategy's expressions need, if any.

    A strategy naming ``revenue_up_3q`` or ``eps_growth_yoy`` has no such
    column on bars until a provider merges it in. Without this the name simply
    failed to resolve per ticker, and the per-ticker guard turned that into a
    warning and an empty screen - a refusal dressed as a result. The provider
    default matches the backtester's, so the two paths fetch the same values.
    """
    from screener.backtester.cli_common import referenced_fundamental_fields
    from screener.backtester.fundamentals import (
        build_fundamental_fetcher,
        fundamental_filing_lag_days,
    )

    needed = referenced_fundamental_fields(entry_expr, exit_expr)
    if not needed:
        return None, None
    provider = "fmp" if market == "us" else "openscreener"
    try:
        fetcher = build_fundamental_fetcher(
            provider,
            market=market,
            fields=tuple(sorted(needed)),
            lag_days=max(fundamental_filing_lag_days(provider), 0),
            refresh=refresh,
        )
    except ValueError as exc:
        raise UnscreenableStrategyError(
            f"this strategy reads {sorted(needed)}, which needs a fundamentals "
            f"provider, and none supports -m {market}: {exc}"
        ) from exc
    return fetcher, provider


__all__ = [
    "OUTPUT_SCORE_COLUMN",
    "ScreenStrategy",
    "UnscreenableStrategyError",
    "aliased_strategy",
    "ensure_screenable",
    "prefilter_filters",
    "resolve_screen_strategy",
    "resolve_universe_tickers",
    "screen_candidates",
    "screen_label",
]
