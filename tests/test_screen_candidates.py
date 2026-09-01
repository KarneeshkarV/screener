"""Stage 6 guards: criterion names as aliases onto the strategy registry.

The flip (docs/plans/unify-screen-backtest.md) makes selected criterion names
resolve to a strategy whose entry expression is evaluated over local bars,
instead of to TradingView filters plus a snapshot ranking recipe. These tests
pin the resolution, the refusals, the run label and the two universe modes.
Whether the bar path agrees with the rolling engine is pinned separately in
``tests/correctness``.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from screener import screen_workflow
from screener.backtester.signal_panel import Candidate
from screener.criteria import CRITERIA
from screener.criteria import registry as criteria_registry
from screener.screen_candidates import (
    OUTPUT_SCORE_COLUMN,
    _candidate_frame,
    _warn_thin_field,
    _fundamentals_for,
    ScreenStrategy,
    UniverseField,
    UnscreenableStrategyError,
    aliased_strategy,
    ensure_screenable,
    prefilter_filters,
    resolve_screen_strategy,
    resolve_universe_field,
    resolve_universe_tickers,
    screen_candidates,
    screen_label,
)
from screener.screen_workflow import (
    _PREFILTER_CANDIDATE_CAP,
    ScreenRequest,
    run_screen_workflow,
)
from screener.strategies.spec import (
    CallableStrategySpec,
    ExpressionStrategySpec,
    discover_plugins,
    registry as strategy_registry,
)

#: The criterion names that also exist as a strategy. Stage 6 turns exactly
#: these into aliases; every other criterion keeps its old snapshot path.
ALIASED = frozenset(
    {"breakout", "mark_minervini", "momentum_12_1", "momentum_12_1_ema10"}
)


@pytest.fixture(autouse=True)
def _plugins() -> None:
    discover_plugins()


def _request(**overrides: object) -> ScreenRequest:
    base: dict[str, object] = {
        "market": "india",
        "criteria_names": ("breakout",),
        "limit": 5,
        "order_by": OUTPUT_SCORE_COLUMN,
        "output_csv": True,
        "detail": False,
        "refresh": False,
        "cache_ttl": "15m",
        "report_path": None,
    }
    base.update(overrides)
    return ScreenRequest(**base)  # type: ignore[arg-type]


class TestAliasing:
    def test_the_overlap_between_criteria_and_strategies_is_exactly_the_aliases(
        self,
    ) -> None:
        # If a fourth name ever exists in both registries it silently changes
        # meaning, so the set is asserted rather than derived at screen time.
        overlap = set(CRITERIA) & set(strategy_registry.names())
        assert overlap == set(ALIASED)

    @pytest.mark.parametrize("name", sorted(ALIASED))
    def test_an_aliased_criterion_resolves_to_its_strategy(self, name: str) -> None:
        resolved = resolve_screen_strategy((name,))

        assert resolved is not None
        assert resolved.criterion == name
        assert resolved.spec.name == name
        assert isinstance(resolved.spec, ExpressionStrategySpec)

    @pytest.mark.parametrize("name", ["value", "near_52_high"])
    def test_a_filters_only_criterion_keeps_the_old_path(self, name: str) -> None:
        assert aliased_strategy(name) is None
        assert resolve_screen_strategy((name,)) is None

    def test_the_default_criterion_aliases_the_rolling_ema_stack(self) -> None:
        """`-c ema` must run the same rule backtest-rolling measures.

        The criterion is named `ema` and the strategy `ema_stack`, so the
        registry lookup alone never connected them and the default screen
        silently ran the TradingView filter set instead of the bar rule.
        """
        resolved = resolve_screen_strategy(("ema",))

        assert resolved is not None
        assert resolved.criterion == "ema"
        assert resolved.spec.name == "ema_stack"

    def test_combining_filters_only_criteria_stays_legal(self) -> None:
        assert resolve_screen_strategy(("value", "near_52_high")) is None

    def test_a_strategy_alias_refuses_to_be_combined(self) -> None:
        with pytest.raises(UnscreenableStrategyError, match="Screen one at a time"):
            resolve_screen_strategy(("breakout", "value"))

    def test_two_strategy_aliases_refuse_each_other_too(self) -> None:
        with pytest.raises(UnscreenableStrategyError, match="Screen one at a time"):
            resolve_screen_strategy(("breakout", "momentum_12_1"))

    def test_the_two_alias_refusal_names_the_conflict_rather_than_an_empty_set(
        self,
    ) -> None:
        # The difference between the selected names and the aliased ones is
        # empty here, so the message must not render it as "combined with []".
        with pytest.raises(UnscreenableStrategyError) as excinfo:
            resolve_screen_strategy(("breakout", "momentum_12_1"))

        assert "[]" not in str(excinfo.value)
        assert "with each other" in str(excinfo.value)


class TestUnscreenable:
    def test_a_callable_only_strategy_is_refused_by_kind(self) -> None:
        # Refusal keys off the spec's shape, not off a name list, so a future
        # callable-only strategy is refused without anyone updating a constant.
        spec = CallableStrategySpec(name="stateful", callable_fn=lambda *a, **k: [])

        with pytest.raises(UnscreenableStrategyError, match="callable-only"):
            ensure_screenable(spec)

    def test_an_expression_strategy_passes_through_unchanged(self) -> None:
        spec = ExpressionStrategySpec(name="expr", entry="close > 0")

        assert ensure_screenable(spec) is spec

    def test_every_registered_callable_strategy_is_refused(self) -> None:
        callable_specs = [
            spec
            for _, spec in strategy_registry.items()
            if isinstance(spec, CallableStrategySpec)
        ]
        assert callable_specs, "fixture is vacuous: no callable strategy registered"
        for spec in callable_specs:
            with pytest.raises(UnscreenableStrategyError):
                ensure_screenable(spec)


class TestLabel:
    def test_a_filters_only_run_keeps_its_historical_label(self) -> None:
        assert screen_label(("ema", "value"), strategy=None, universe=None) == (
            "ema+value"
        )

    def test_a_bar_rule_run_is_labelled_so_a_diff_cannot_cross_the_flip(self) -> None:
        # D17: the same name now answers a different question, so its run must
        # not diff against the pre-flip history stored under the bare name.
        strategy = resolve_screen_strategy(("breakout",))
        assert screen_label(("breakout",), strategy=strategy, universe=None) == (
            "breakout@tv"
        )

    def test_the_universe_mode_is_part_of_the_label(self) -> None:
        # D9: the two modes see different fields, so their added/removed diffs
        # are not comparable and must not share a history row.
        strategy = resolve_screen_strategy(("breakout",))
        assert (
            screen_label(("breakout",), strategy=strategy, universe="nifty50")
            == "breakout@universe:nifty50"
        )


# An alias usually fronts the criterion of the same name. Two do not, both
# because the criterion drops names the bar rule keeps: ``breakout``'s
# ``price_52_week_high`` leg reads the 52-week high of highs while its rule
# reads the high of closes, so it fronts the volume leg alone; and
# ``momentum_12_1``'s ``Perf.Y > Perf.1M`` is a calendar-anchored reading of a
# rule written on 21/252 session offsets, which leaves no sound leg at all.
_PREFILTER_OF: dict[str, str | None] = {
    "breakout": "above_avg_volume",
    "mark_minervini": "mark_minervini",
    "momentum_12_1": None,
    "momentum_12_1_ema10": None,
}


class TestPrefilter:
    @pytest.mark.parametrize("name", sorted(ALIASED))
    def test_each_alias_declares_the_criterion_whose_filters_cut_the_field(
        self, name: str
    ) -> None:
        strategy = resolve_screen_strategy((name,))
        assert strategy is not None
        assert strategy.tv_prefilter == _PREFILTER_OF[name]

    @pytest.mark.parametrize("name", sorted(ALIASED))
    def test_the_prefilter_yields_that_criterion_s_own_filters(self, name: str) -> None:
        strategy = resolve_screen_strategy((name,))
        assert strategy is not None
        criterion = _PREFILTER_OF[name]
        expected = [] if criterion is None else list(criteria_registry.get(criterion)())
        assert prefilter_filters(strategy) == expected

    def test_a_strategy_without_a_prefilter_scans_unfiltered(self) -> None:
        # No declared prefilter means no field cut, which is always sound: the
        # bar rule then judges every name the scan returned.
        spec = ExpressionStrategySpec(name="no_prefilter", entry="close > 0")
        strategy = ScreenStrategy(criterion="no_prefilter", spec=spec)

        assert strategy.tv_prefilter is None
        assert prefilter_filters(strategy) == []


class TestUniverseMode:
    def test_a_named_index_resolves_to_tickers(self) -> None:
        tickers = resolve_universe_tickers("nifty50", "india")

        assert len(tickers) > 10
        assert all(isinstance(t, str) and t for t in tickers)

    def test_a_universe_file_resolves_and_ignores_comments(self, tmp_path) -> None:
        path = tmp_path / "u.txt"
        path.write_text("# a comment\nNSE:RELIANCE\nNSE:TCS\n", encoding="utf-8")

        assert resolve_universe_tickers(str(path), "india") == [
            "NSE:RELIANCE",
            "NSE:TCS",
        ]

    def test_a_snapshot_universe_resolves_to_the_window_open_today(
        self, tmp_path
    ) -> None:
        # The union of every snapshot would screen names that have left the
        # index. Only the window open today is the live field, so LEFT - which
        # was dropped in 2021 - must not come back.
        snapshots = tmp_path / "snaps.csv"
        snapshots.write_text(
            "effective_date,symbol\n"
            "2020-01-01,STAY.NS\n"
            "2020-01-01,LEFT.NS\n"
            "2021-01-01,STAY.NS\n"
            "2021-01-01,JOINED.NS\n",
            encoding="utf-8",
        )
        config = tmp_path / "universes.yaml"
        config.write_text(
            "universes:\n"
            "  book_pit:\n"
            "    type: snapshots\n"
            "    market: india\n"
            '    benchmark: "^NSEI"\n'
            "    path: snaps.csv\n",
            encoding="utf-8",
        )

        field = resolve_universe_field("book_pit", "india", config_path=config)

        assert field.tickers == ["STAY.NS", "JOINED.NS"]
        assert "book_pit" in field.note

    def test_a_config_universe_needs_its_config(self) -> None:
        # Without --universe-config the name falls through to the file reader,
        # which is the honest failure: there is nowhere else it could be.
        with pytest.raises(FileNotFoundError):
            resolve_universe_tickers("nifty500_pit", "india")

    def test_universe_without_a_strategy_alias_is_refused(self, monkeypatch) -> None:
        # A filters-only criterion has no bar rule, so there is nothing to run
        # against a local universe; failing loudly beats screening on nothing.
        def unexpected_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
            raise AssertionError("scan must not run once the request is refused")

        monkeypatch.setattr(screen_workflow, "scan", unexpected_scan)

        with pytest.raises(UnscreenableStrategyError, match="--universe needs"):
            run_screen_workflow(_request(criteria_names=("value",), universe="nifty50"))


class TestWorkflowWiring:
    def test_the_bar_path_scans_with_the_prefilter_and_never_with_a_scorer(
        self, monkeypatch
    ) -> None:
        # The scan's only job in default mode is the field cut: ranking comes
        # from the bar rule, so asking TradingView to score would be dead work
        # on a scale the run does not use.
        captured: dict[str, object] = {}

        def fake_scan(**kwargs: object) -> tuple[int, pd.DataFrame, datetime]:
            captured.update(kwargs)
            return (
                2,
                pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB"]}),
                datetime(2024, 6, 3, 12, 0),
            )

        def fake_candidates(strategy, **kwargs: object) -> pd.DataFrame:
            captured["tickers"] = list(kwargs["tickers"])
            captured["strategy"] = strategy.spec.name
            return pd.DataFrame({"name": ["AAA"], OUTPUT_SCORE_COLUMN: [100.0]})

        monkeypatch.setattr(screen_workflow, "scan", fake_scan)
        monkeypatch.setattr(screen_workflow, "screen_candidates", fake_candidates)

        outcome = run_screen_workflow(_request(criteria_names=("breakout",)))

        assert captured["scorer"] is None
        assert captured["filters"] == list(
            criteria_registry.get(_PREFILTER_OF["breakout"])()
        )
        assert captured["tickers"] == ["NSE:AAA", "NSE:BBB"]
        assert captured["strategy"] == "breakout"
        # The digest is the settings fingerprint; the readable part is what
        # says which mode ran. Both must be there, so the mode is asserted
        # exactly and the digest only for its shape.
        assert outcome.label.startswith("breakout@tv#")
        assert outcome.total == 2

    def test_the_prefilter_scan_is_not_cut_to_the_result_limit(
        self, monkeypatch
    ) -> None:
        # The scan is a field cut, so capping it at ``-n`` would hand the rule
        # only the top ``-n`` names by raw volume and drop names the rule
        # keeps - the one thing a prefilter may never do (D21). ``--limit``
        # belongs to the candidates the rule returns.
        captured: dict[str, object] = {}

        def fake_scan(**kwargs: object) -> tuple[int, pd.DataFrame, datetime]:
            captured.update(kwargs)
            return (
                2,
                pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB"]}),
                datetime(2024, 6, 3, 12, 0),
            )

        def fake_candidates(strategy, **kwargs: object) -> pd.DataFrame:
            captured["candidate_limit"] = kwargs["limit"]
            captured["candidate_order_by"] = kwargs["order_by"]
            return pd.DataFrame({"name": ["AAA"], OUTPUT_SCORE_COLUMN: [100.0]})

        monkeypatch.setattr(screen_workflow, "scan", fake_scan)
        monkeypatch.setattr(screen_workflow, "screen_candidates", fake_candidates)

        run_screen_workflow(_request(criteria_names=("breakout",), limit=5))

        assert captured["limit"] == _PREFILTER_CANDIDATE_CAP
        assert captured["limit"] > 5
        assert captured["candidate_limit"] == 5

    def test_the_bar_path_forwards_sort_to_the_candidate_layer(
        self, monkeypatch
    ) -> None:
        # --sort was silently dropped on this path: the run answered in rank
        # order with no error, so a user asking for volume order got rank
        # order and no way to tell.
        captured: dict[str, object] = {}

        def fake_scan(**kwargs: object) -> tuple[int, pd.DataFrame, datetime]:
            return (
                1,
                pd.DataFrame({"ticker": ["NSE:AAA"]}),
                datetime(2024, 6, 3, 12, 0),
            )

        def fake_candidates(strategy, **kwargs: object) -> pd.DataFrame:
            captured["order_by"] = kwargs["order_by"]
            return pd.DataFrame({"name": ["AAA"], OUTPUT_SCORE_COLUMN: [100.0]})

        monkeypatch.setattr(screen_workflow, "scan", fake_scan)
        monkeypatch.setattr(screen_workflow, "screen_candidates", fake_candidates)

        run_screen_workflow(_request(criteria_names=("breakout",), order_by="volume"))

        assert captured["order_by"] == "volume"

    def test_universe_mode_runs_no_scan_at_all(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def unexpected_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
            raise AssertionError("--universe must consult no vendor field")

        def fake_candidates(strategy, **kwargs: object) -> pd.DataFrame:
            captured["tickers"] = list(kwargs["tickers"])
            captured["scanned"] = kwargs["scanned"]
            return pd.DataFrame({"name": ["AAA"], OUTPUT_SCORE_COLUMN: [100.0]})

        monkeypatch.setattr(screen_workflow, "scan", unexpected_scan)
        monkeypatch.setattr(screen_workflow, "screen_candidates", fake_candidates)
        monkeypatch.setattr(
            screen_workflow,
            "resolve_universe_field",
            lambda universe, market, config_path=None: UniverseField(
                ["NSE:AAA", "NSE:BBB", "NSE:CCC"]
            ),
        )

        outcome = run_screen_workflow(
            _request(criteria_names=("breakout",), universe="nifty50")
        )

        assert captured["scanned"] is None
        assert captured["tickers"] == ["NSE:AAA", "NSE:BBB", "NSE:CCC"]
        assert outcome.label.startswith("breakout@universe:nifty50#")
        # `total` is the field the rule judged, which in this mode is the whole
        # universe rather than a vendor-narrowed slice.
        assert outcome.total == 3


class TestCandidateFrame:
    def test_no_tickers_yields_the_empty_result_shape(self) -> None:
        # An empty field must not raise: it is an ordinary "nothing qualified"
        # answer, and the caller still needs the display columns to render it.
        strategy = resolve_screen_strategy(("breakout",))
        assert strategy is not None

        df = screen_candidates(
            strategy,
            market="india",
            tickers=[],
            as_of=date(2024, 6, 3),
            warnings=[],
        )

        assert df.empty
        assert "name" in df.columns
        assert OUTPUT_SCORE_COLUMN in df.columns

    def test_an_intraday_screen_builds_an_intraday_price_fetcher(
        self, monkeypatch
    ) -> None:
        from screener.backtester import data as backtest_data

        captured: dict[str, object] = {}

        class FetcherBuilt(Exception):
            pass

        def capture_fetcher(**kwargs: object) -> None:
            captured.update(kwargs)
            raise FetcherBuilt

        monkeypatch.setattr(backtest_data, "build_price_fetcher", capture_fetcher)
        strategy = resolve_screen_strategy(("breakout",))
        assert strategy is not None

        with pytest.raises(FetcherBuilt):
            screen_candidates(
                strategy,
                market="india",
                tickers=["NSE:RELIANCE"],
                as_of=date(2024, 6, 3),
                interval="1h",
                warnings=[],
            )

        assert captured["interval"] == "1h"


def _candidate(
    ticker: str, rank: int, score: float | None, setup_score: float | None = None
) -> Candidate:
    """One candidate as the candidate layer hands it over.

    ``setup_score`` defaults to a value that falls with ``rank``, which is the
    only relationship the frame builder relies on: the layer takes the
    percentile over the whole eligible field, so a lower rank always carries a
    lower score.
    """
    return Candidate(
        ticker=ticker,
        rank=rank,
        rank_basis="rank_score" if score is not None else "as_of_dollar_vol",
        rank_score=score,
        as_of_close=10.0,
        as_of_volume=1_000.0,
        as_of_dollar_vol=10_000.0 * rank,
        setup_score=100.0 - (rank - 1) * 10.0 if setup_score is None else setup_score,
        signal_idx=5,
        role="active",
    )


def _bars(close: float, previous: float, volume: float) -> pd.DataFrame:
    index = pd.date_range("2024-06-01", periods=2, freq="D")
    return pd.DataFrame(
        {"close": [previous, close], "volume": [volume, volume]}, index=index
    )


class TestResultFrame:
    def test_rows_keep_the_candidate_layer_s_rank_order(self) -> None:
        # The screen's ordering must be the candidate layer's ordering, not a
        # re-sort by the score column: they can only agree if the score is
        # derived from the same basis, which is what the next test pins.
        scanned = pd.DataFrame(
            {
                "ticker": ["NSE:AAA", "NSE:BBB", "NSE:CCC"],
                "description": ["A", "B", "C"],
            }
        )
        candidates = [
            _candidate("NSE:CCC", 1, 9.0),
            _candidate("NSE:AAA", 2, 5.0),
            _candidate("NSE:BBB", 3, 1.0),
        ]

        df = _candidate_frame(candidates, {}, scanned)

        assert df["ticker"].tolist() == ["NSE:CCC", "NSE:AAA", "NSE:BBB"]
        assert df["description"].tolist() == ["C", "A", "B"]

    def test_the_score_column_is_the_one_the_candidate_layer_assigned(
        self,
    ) -> None:
        # The score is computed with the candidate, over the whole eligible
        # field, so the frame must carry it through untouched rather than
        # re-deriving a percentile of whoever survived to this point.
        candidates = [
            _candidate("NSE:CCC", 1, 900.0, setup_score=100.0),
            _candidate("NSE:AAA", 2, 5.0, setup_score=62.5),
            _candidate("NSE:BBB", 3, 1.0, setup_score=12.5),
        ]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB", "NSE:CCC"]})

        scores = _candidate_frame(candidates, {}, scanned)[OUTPUT_SCORE_COLUMN]

        assert scores.tolist() == [100.0, 62.5, 12.5]

    def test_a_dollar_volume_ranked_day_still_scores(self) -> None:
        # A strategy that writes no factor score ranks on dollar volume, and the
        # score column must follow that basis instead of coming out all-NaN.
        candidates = [_candidate("NSE:BBB", 1, None), _candidate("NSE:AAA", 2, None)]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB"]})

        scores = _candidate_frame(candidates, {}, scanned)[OUTPUT_SCORE_COLUMN]

        assert scores.notna().all()

    def test_universe_mode_takes_its_display_columns_from_bars(self) -> None:
        # There is no snapshot row in --universe mode, so close/change/volume
        # come from the bars the rule was evaluated on. market_cap_basic has no
        # bar equivalent and stays NaN rather than being invented.
        candidates = [_candidate("NSE:AAA", 1, 5.0)]
        bars = {"NSE:AAA": _bars(close=110.0, previous=100.0, volume=2_000.0)}

        df = _candidate_frame(candidates, bars, None)

        row = df.iloc[0]
        # ``name`` is the plain symbol, matching what the TradingView snapshot
        # puts there: history keys on it and both enrichment paths look the
        # symbol up by it, so the exchange prefix must not survive.
        assert row["ticker"] == "NSE:AAA"
        assert row["name"] == "AAA"
        assert row["close"] == pytest.approx(110.0)
        assert row["change"] == pytest.approx(10.0)
        assert row["volume"] == pytest.approx(2_000.0)
        assert pd.isna(row["market_cap_basic"])

    def test_sort_reorders_the_finished_rows_by_a_display_column(self) -> None:
        # Membership still comes from the rule; only the presentation order
        # changes, exactly as --sort re-sorts a snapshot scan.
        candidates = [_candidate("NSE:AAA", 1, 9.0), _candidate("NSE:BBB", 2, 8.0)]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB"], "volume": [1.0, 2.0]})

        df = _candidate_frame(candidates, {}, scanned, order_by="volume")

        assert df["ticker"].tolist() == ["NSE:BBB", "NSE:AAA"]

    def test_sort_on_a_column_the_results_lack_warns_instead_of_going_quiet(
        self,
    ) -> None:
        candidates = [_candidate("NSE:AAA", 1, 9.0)]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA"]})
        warnings: list[str] = []

        df = _candidate_frame(
            candidates, {}, scanned, order_by="market_cap_basic", warnings=warnings
        )

        assert df["ticker"].tolist() == ["NSE:AAA"]
        assert any("market_cap_basic" in w for w in warnings)

    def test_the_score_does_not_depend_on_the_result_limit(self) -> None:
        # setup_score used to be a percentile of the truncated top-N, so the
        # same name scored differently at -n 1 and -n 3 and the column meant
        # something different from the snapshot path's absolute composite.
        candidates = [
            _candidate("NSE:AAA", 1, 9.0),
            _candidate("NSE:BBB", 2, 5.0),
            _candidate("NSE:CCC", 3, 1.0),
        ]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB", "NSE:CCC"]})

        full = _candidate_frame(candidates, {}, scanned)
        capped = _candidate_frame(candidates, {}, scanned, limit=1)

        assert capped["ticker"].tolist() == ["NSE:AAA"]
        assert capped[OUTPUT_SCORE_COLUMN].iloc[0] == full[OUTPUT_SCORE_COLUMN].iloc[0]

    def test_a_candidate_without_bars_is_dropped_rather_than_shown_blank(self) -> None:
        candidates = [_candidate("NSE:AAA", 1, 5.0), _candidate("NSE:GONE", 2, 4.0)]
        bars = {"NSE:AAA": _bars(close=110.0, previous=100.0, volume=2_000.0)}

        df = _candidate_frame(candidates, bars, None)

        assert df["name"].tolist() == ["AAA"]
        assert len(df) == 1

    def test_no_candidates_yields_the_empty_result_shape(self) -> None:
        df = _candidate_frame([], {}, None)

        assert df.empty
        assert OUTPUT_SCORE_COLUMN in df.columns


class TestFundamentalWiring:
    """A strategy that reads a fundamental must get a provider, not a warning.

    Nothing merged those dated columns into the bars on this path, so the name
    failed to resolve per ticker and the per-ticker guard turned the refusal
    into a log line plus an empty table.
    """

    def test_an_expression_reading_fundamentals_gets_a_fetcher(self) -> None:
        fetcher, provider = _fundamentals_for(
            "close > 0 and revenue_up_3q > 0",
            None,
            market="india",
            refresh=False,
        )

        assert provider == "openscreener"
        assert fetcher is not None
        assert "revenue_up_3q" in fetcher.fields

    def test_a_price_only_expression_still_fetches_nothing(self) -> None:
        assert _fundamentals_for("close > 0", None, market="india", refresh=False) == (
            None,
            None,
        )

    def test_the_us_default_provider_matches_the_backtester_s(self) -> None:
        _fetcher, provider = _fundamentals_for(
            "eps_growth_yoy > 0", None, market="us", refresh=False
        )

        assert provider == "fmp"

    @pytest.mark.parametrize(
        "name",
        [
            "ema150_200_revenue_up_3q",
            "minervini_growth_in",
            "minervini_pro_in",
        ],
    )
    def test_every_fundamental_reading_strategy_resolves_a_provider(
        self, name: str
    ) -> None:
        spec = strategy_registry.get_optional(name)
        assert isinstance(spec, ExpressionStrategySpec)
        profile = spec.profile
        entry = (profile.entry_expr if profile else None) or spec.entry
        exit_expr = (profile.exit_expr if profile else None) or spec.exit

        fetcher, provider = _fundamentals_for(
            entry, exit_expr, market="india", refresh=False
        )

        assert provider is not None and fetcher is not None


class TestThinFieldWarning:
    """``setup_score`` is a percentile of the names that loaded, so say so.

    A name missing from the panel is not ranked and not shown; nothing else in
    the output reveals that the field the percentile is over shrank.
    """

    @staticmethod
    def _bars(*days: str) -> pd.DataFrame:
        index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
        return pd.DataFrame({"close": [1.0] * len(index)}, index=index)

    def test_a_field_the_vendor_served_nothing_for_is_reported(self) -> None:
        as_of = pd.Timestamp("2026-08-28")
        bars_by_tv = {
            "NSE:AAA": self._bars("2026-08-28"),
            "NSE:BBB": pd.DataFrame(),
            "NSE:CCC": pd.DataFrame(),
        }
        warnings: list[str] = []

        _warn_thin_field(bars_by_tv, requested=3, as_of=as_of, warnings=warnings)

        assert len(warnings) == 1
        assert "only 1 of 3" in warnings[0]
        assert "2026-08-28" in warnings[0]

    def test_a_name_that_simply_did_not_trade_is_not_a_hole(self) -> None:
        """Illiquid names skip sessions; that is the field, not a failure."""
        as_of = pd.Timestamp("2026-08-28")
        bars_by_tv = {f"NSE:{i:03d}": self._bars("2026-08-27") for i in range(19)}
        bars_by_tv["NSE:AAA"] = self._bars("2026-08-28")
        warnings: list[str] = []

        _warn_thin_field(bars_by_tv, requested=20, as_of=as_of, warnings=warnings)

        assert warnings == []

    def test_a_field_that_mostly_loaded_says_nothing(self) -> None:
        as_of = pd.Timestamp("2026-08-28")
        bars_by_tv = {f"NSE:{i:03d}": self._bars("2026-08-28") for i in range(19)}
        bars_by_tv["NSE:GAP"] = pd.DataFrame()
        warnings: list[str] = []

        _warn_thin_field(bars_by_tv, requested=20, as_of=as_of, warnings=warnings)

        assert warnings == []

    def test_no_as_of_bar_is_not_a_coverage_claim(self) -> None:
        """The empty-window case is already reported; do not report it twice."""
        warnings: list[str] = []
        _warn_thin_field({}, requested=10, as_of=None, warnings=warnings)
        assert warnings == []
