"""The screen's half of the unified candidate layer.

A criterion name used to mean two things at once: a set of TradingView filters
and a ranking recipe over the row those filters returned. For criterion names
that also exist as a strategy, such as ``breakout``, ``momentum_12_1`` and
``mark_minervini``, that meant two implementations of one rule. This module
removes that duplication.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from screener.criteria import registry as criteria_registry
from screener.markets import get_market
from screener.scoring.bar_scores import (
    DEFAULT_PRICE_ADJUSTMENT,
    PriceAdjustment,
)
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    ExpressionStrategySpec,
    StrategyProfile,
    StrategySpec,
    resolve_strategy_profile,
)
from screener.strategies.spec import registry as strategy_registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.fundamentals import FundamentalFetcher
    from screener.backtester.signal_panel import Candidate

#: The interval a screen runs at unless asked otherwise. Daily bars are the
#: only ones the dated inputs (fundamentals, earnings) have a meaning for.
DEFAULT_INTERVAL = "1d"

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


#: Criteria whose strategy is registered under a different name. Most aliases
#: are implicit - ``breakout`` the criterion and ``breakout`` the strategy share
#: a name, so the registry lookup finds them. This table covers the ones where
#: the strategy earned a more descriptive name than the criterion it implements,
#: and without it the criterion silently keeps the vendor-snapshot path while a
#: bar rule for the very same question sits unused in the registry.
_STRATEGY_ALIASES = {
    # EMA5 > EMA20 > EMA100 > EMA200, the default screen. `ema` is that stack
    # as TradingView columns; `ema_stack` is the same stack as an entry
    # expression, and is what backtest-rolling measures.
    "ema": "ema_stack",
}


def aliased_strategy(criterion: str) -> StrategySpec | None:
    """The strategy a criterion name aliases, or ``None`` for a filters-only name."""
    return strategy_registry.get_optional(_STRATEGY_ALIASES.get(criterion, criterion))


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


#: Hex digits of the settings fingerprint carried in a bar-rule label. Eight
#: is short enough to read in a terminal and wide enough that two settings a
#: user would actually type will not collide.
_FINGERPRINT_WIDTH = 8

#: Profile fields the fingerprint ignores. ``tv_prefilter`` narrows the field
#: before bars are fetched and may only ever remove names the bar rule would
#: have removed anyway, so it cannot change who is a candidate - and a run with
#: a prefilter must stay diffable against one without.
_UNFINGERPRINTED_GATES = frozenset({"tv_prefilter"})


def settings_fingerprint(
    gates: StrategyProfile,
    *,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """A short stable digest of everything that decides who is a candidate.

    Two runs sharing a fingerprint asked the same question, so their
    added/removed diff is meaningful. Two runs that do not share one did not,
    and history must not diff across them - which is the whole reason a label
    carries it.

    Derived from the field list rather than from a hand-written tuple, so a
    gate added to :class:`~screener.strategies.spec.StrategyProfile` is
    fingerprinted without anyone remembering to add it here.
    """
    import hashlib
    import json

    payload = {
        name: value
        for name, value in sorted(gates.model_dump().items())
        if name not in _UNFINGERPRINTED_GATES
    }
    payload["price_adjustment"] = price_adjustment
    payload["interval"] = interval
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return digest[:_FINGERPRINT_WIDTH]


def screen_label(
    names: Sequence[str],
    *,
    strategy: ScreenStrategy | None,
    universe: str | None,
    fingerprint: str | None = None,
) -> str:
    """The ``runs.criteria`` label, which records the semantics of the run.

    A filters-only run keeps the label it always had, so its history is
    continuous. A bar-rule run gets a new one, because the same name now
    answers a different question and a diff must never cross that change (D17
    in docs/plans/unify-screen-backtest.md). The universe mode is part of the
    label for the same reason: the two modes see different fields, so their
    added/removed diffs are not comparable (D9).

    ``fingerprint`` extends that rule to the gates. A screen run with
    ``--min-price 50`` is not the same question as one without, so it must not
    diff against it; the digest from :func:`settings_fingerprint` says so in
    the label. The universe stays spelled out rather than folded into the
    digest: it is the part a reader needs to see.
    """
    joined = "+".join(names)
    if strategy is None:
        return joined
    label = f"{joined}@universe:{universe}" if universe else f"{joined}@tv"
    return label if fingerprint is None else f"{label}#{fingerprint}"


def prefilter_filters(strategy: ScreenStrategy) -> list:
    """The TradingView filters that cut the field for ``strategy``.

    Empty when the strategy declares no prefilter, which means the scan is
    unfiltered and the bar rule does all the work.
    """
    name = strategy.tv_prefilter
    if name is None:
        return []
    return list(criteria_registry.get(name)())


@dataclass(frozen=True)
class UniverseField:
    """The names ``--universe`` will screen, and how they were chosen.

    ``note`` is empty for a plain index or file, where the name says everything
    there is to say. A membership-window universe fills it in, because there
    the field is a dated answer and the reader has to be told which date.
    """

    tickers: list[str]
    note: str = ""


def _members_open_on(
    windows: Sequence[tuple[str, date, date | None]], as_of: date
) -> list[str]:
    """Symbols whose half-open ``[start, end)`` window contains ``as_of``.

    A screen is a statement about today, so today's open windows are the whole
    of the correct field: a name added last month belongs in it, and one
    dropped last month does not, even though both appear in the snapshot file.
    Taking the union of every snapshot instead - which is what
    ``UniverseSelection.symbols`` is - would screen names that have left the
    index.
    """
    return list(
        dict.fromkeys(
            symbol
            for symbol, start, end in windows
            if start <= as_of and (end is None or as_of < end)
        )
    )


def resolve_universe_field(
    universe: str,
    market: str,
    *,
    config_path: str | Path | None = None,
    as_of: date | None = None,
) -> UniverseField:
    """Resolve a named universe, a config-defined universe, or a file.

    Built-ins win, then the ``--universe-config`` definitions, then the name is
    read as a path. That is the same order :func:`load_universe_selection` uses
    for ``backtest-rolling``, so one name cannot mean two different books
    depending on which command asked.
    """
    from screener.universes import (
        UniverseRequest,
        UniverseSource,
        available_universes,
        load_universe_selection,
        resolve_universe,
    )

    is_index = universe in available_universes()
    if not is_index and config_path is not None:
        today = as_of or date.today()
        selection = load_universe_selection(
            universe, market=market, as_of=today, config_path=config_path
        )
        if selection.membership_windows:
            tickers = _members_open_on(selection.membership_windows, today)
            return UniverseField(
                tickers,
                f"{selection.name}: {len(tickers)} names in the membership "
                f"window open on {today.isoformat()} ({selection.source})",
            )
        return UniverseField(
            list(selection.symbols),
            f"{selection.name}: {len(selection.symbols)} names ({selection.source})",
        )

    source = UniverseSource.INDEX_CURRENT if is_index else UniverseSource.FILE
    request = UniverseRequest(
        source=source,
        market=market,
        index_name=universe if is_index else None,
        file=None if is_index else universe,
        comment_prefixes=("#",),
    )
    return UniverseField(resolve_universe(request))


def resolve_universe_tickers(
    universe: str, market: str, *, config_path: str | Path | None = None
) -> list[str]:
    """Resolve a named universe or a universe file into TradingView symbols."""
    return resolve_universe_field(universe, market, config_path=config_path).tickers


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


#: Share of the requested names that must arrive with bars before the run is
#: reported without comment. The measure is "any bars at all in the window",
#: not "a bar on the as-of date": a micro cap that simply did not trade that
#: session is ordinary on a field this wide, while a name the vendor served
#: nothing for is a hole in the field. ``setup_score`` is a percentile of
#: whichever names loaded, so a field with holes scores differently from the
#: same field without them - which is what makes this worth saying.
_FIELD_COVERAGE_FLOOR = 0.9


def _warn_thin_field(
    bars_by_tv: dict[str, pd.DataFrame],
    *,
    requested: int,
    as_of: pd.Timestamp | None,
    warnings: list[str],
) -> None:
    """Report a field the vendor served nothing for a real share of."""
    if as_of is None or requested <= 0:
        return
    loaded = sum(
        1 for bars in bars_by_tv.values() if bars is not None and not bars.empty
    )
    if loaded >= requested * _FIELD_COVERAGE_FLOOR:
        return
    warnings.append(
        f"only {loaded} of {requested} requested names arrived with bars, so "
        f"the rule judged that many as of {pd.Timestamp(as_of).date()} and "
        "setup_score is a percentile of them; the rest served no history at "
        "all this run (vendor gaps, or downloads that failed). Re-run with "
        "--refresh to fill them in."
    )


def resolve_screen_gates(
    strategy: ScreenStrategy,
    *,
    market: str,
    overrides: Mapping[str, Any] | None = None,
) -> StrategyProfile:
    """The candidate gates the screen applies for ``strategy`` on ``market``.

    The mirror of
    :func:`screener.backtester.workflow.resolve_rolling_gates`: for one
    strategy and one market the two must return the same gates, or a screen
    names candidates no backtest would have entered. ``tests/correctness``
    asserts that equality across the whole strategy registry.

    ``overrides`` are the gate flags the user actually typed, in the same
    ``resolve_strategy_profile`` form the rolling engine builds - so a typed
    flag wins here exactly as it wins there.
    """
    return resolve_strategy_profile(strategy.spec, overrides, market=market)


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
    gates: StrategyProfile | None = None,
    interval: str = DEFAULT_INTERVAL,
    max_universe: int = 0,
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
    inside the candidate layer. The candidate layer takes the percentile over
    the whole eligible field anyway, so the same name scores the same at
    ``-n 10`` and at ``-n 100``.

    ``gates`` is the resolved candidate gates. ``None`` means "whatever this
    strategy declares on this market", which is what
    :func:`resolve_screen_gates` answers and what the rolling backtest would
    have used; a caller passing a profile is stating the gates outright.

    ``max_universe`` caps the field before bars are fetched, ``0`` meaning no
    cap. It is run-scoped, like the universe itself, so it is not a gate.
    """
    from screener.backtester.data import build_price_fetcher
    from screener.backtester.price_panel import PricePanelInputs, build_price_panel
    from screener.backtester.signal_panel import (
        DEFAULT_MIN_AS_OF_COVERAGE,
        SignalPanelInputs,
        build_day_candidates,
        parse_signal_program,
    )

    if not tickers:
        return pd.DataFrame(columns=[*_DISPLAY_COLUMNS, OUTPUT_SCORE_COLUMN])

    profile = (
        gates if gates is not None else resolve_screen_gates(strategy, market=market)
    )
    venue = get_market(market)
    entry_expr = profile.entry_expr or strategy.spec.entry
    exit_expr = profile.exit_expr or strategy.spec.exit
    fundamental_fetcher, fundamentals_provider = _fundamentals_for(
        entry_expr, exit_expr, market=market, refresh=refresh
    )
    _check_interval(
        interval,
        fundamental_fetcher=fundamental_fetcher,
        earnings_blackout_days=profile.earnings_blackout_days,
    )
    # An intraday screen asks about the last completed bar *of* the as-of date,
    # so the window has to run to the end of that session rather than to its
    # midnight - which is where a bare date lands and which would exclude every
    # bar of the day being asked about.
    end_ts = _end_of_window(as_of, interval)
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
        min_score=profile.min_score,
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
        max_universe=max_universe,
        interval=interval,
        price_adjustment=price_adjustment,
        strategy_name=strategy.spec.name,
        fundamentals_provider=fundamentals_provider,
    )
    fetcher = build_price_fetcher(
        auto_adjust=(price_adjustment == "full"),
        refresh=refresh,
        strict=strict,
        interval=interval,
    )
    panel = build_price_panel(
        panel_inputs,
        fetcher,
        entry_ast=program.entry_ast,
        exit_ast=program.exit_ast,
        # The liquidity gate is a rolling mean over ``avg_dollar_volume_window``
        # bars, so the panel has to carry that much history even when the entry
        # rule itself needs less. A backtest's window is long enough that this
        # never bites; a screen's window is a fortnight, so without it the gate
        # would be judged on a handful of bars.
        lookback=max(program.lookback, profile.avg_dollar_volume_window),
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
        # A screen runs against a live market, where the vendor serves a
        # partial bar for the open session to whichever names it happened to
        # refresh. That bar is on the master calendar (a union over tickers),
        # so without this the as-of snaps onto a session almost nobody has and
        # the run ranks a handful of names against each other - a different
        # handful every run, as the cache fills. The rolling engine never sees
        # such a row, which is why the floor is the screen's to set.
        min_coverage=DEFAULT_MIN_AS_OF_COVERAGE,
        # The screen's as-of bar is the newest bar there is, so the
        # backtester's "a later bar must exist to fill the entry on" rule would
        # reject every name. A screen names today's triggers; the fill is
        # tomorrow's problem, and tomorrow's bar does not exist yet.
        require_next_bar=False,
    )
    _warn_thin_field(
        panel.bars_by_tv,
        requested=len(tickers),
        as_of=day.as_of,
        warnings=warnings,
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
    # The candidate layer scored these names when it ranked them, over the
    # whole eligible field and before ``limit``, so there is nothing to
    # recompute here - and nothing that could disagree with the ranking.
    scores = {c.ticker: c.setup_score for c in candidates}
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


class IntervalNotScreenableError(ValueError):
    """Raised when an interval cannot answer the question being asked.

    A domain error, like :class:`UnscreenableStrategyError`: the Click adapter
    turns it into a ``UsageError`` so this module stays free of the CLI.
    """


def _check_interval(
    interval: str,
    *,
    fundamental_fetcher: FundamentalFetcher | None,
    earnings_blackout_days: int | None,
) -> None:
    """Refuse an intraday interval the rest of the screen cannot honour.

    Both refusals are about dated data meeting undated bars. A fundamental
    column is stamped with a filing date and forward-filled onto daily bars;
    an earnings blackout suppresses whole calendar days. Neither has an
    intraday spelling, so an intraday run would silently apply them to the
    wrong bars rather than fail. Saying so is the honest answer.
    """
    if interval == DEFAULT_INTERVAL:
        return
    if fundamental_fetcher is not None:
        raise IntervalNotScreenableError(
            f"--interval {interval} cannot be used with a strategy that reads "
            "fundamental fields: filings are dated to a day and are merged onto "
            f"daily bars. Screen this strategy at --interval {DEFAULT_INTERVAL}."
        )
    if earnings_blackout_days is not None:
        raise IntervalNotScreenableError(
            f"--interval {interval} cannot be used with --earnings-blackout: "
            "the blackout suppresses whole calendar days, which has no intraday "
            f"meaning. Screen at --interval {DEFAULT_INTERVAL}, or drop the "
            "blackout."
        )


def _end_of_window(as_of: date, interval: str) -> pd.Timestamp:
    """The last instant of ``as_of`` the panel should be built through.

    Daily bars are stamped at midnight, so a bare date is already the right
    end. Intraday bars are stamped through the session, so the window has to
    run to the end of the day or it would hold none of them.
    """
    end = pd.Timestamp(as_of)
    if interval == DEFAULT_INTERVAL:
        return end
    return end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


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
    "DEFAULT_INTERVAL",
    "OUTPUT_SCORE_COLUMN",
    "IntervalNotScreenableError",
    "ScreenStrategy",
    "UniverseField",
    "UnscreenableStrategyError",
    "aliased_strategy",
    "ensure_screenable",
    "prefilter_filters",
    "resolve_screen_gates",
    "resolve_screen_strategy",
    "settings_fingerprint",
    "resolve_universe_field",
    "resolve_universe_tickers",
    "screen_candidates",
    "screen_label",
]
