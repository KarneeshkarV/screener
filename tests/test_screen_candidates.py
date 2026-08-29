"""Stage 6 guards: criterion names as aliases onto the strategy registry.

The flip (docs/plans/unify-screen-backtest.md) makes three criterion names -
``breakout``, ``mark_minervini`` and ``momentum_12_1`` - resolve to a strategy
whose entry expression is evaluated over local bars, instead of to a set of
TradingView filters plus a snapshot ranking recipe. These tests pin the
resolution, the refusals, the run label and the two universe modes. Whether the
bar path *agrees* with the rolling engine is pinned separately, in
``tests/correctness``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener import screen_workflow
from screener.backtester.signal_panel import Candidate
from screener.criteria import CRITERIA
from screener.criteria import registry as criteria_registry
from screener.screen_candidates import (
    OUTPUT_SCORE_COLUMN,
    _candidate_frame,
    ScreenStrategy,
    UnscreenableStrategyError,
    aliased_strategy,
    ensure_screenable,
    prefilter_filters,
    resolve_screen_strategy,
    resolve_universe_tickers,
    screen_candidates,
    screen_label,
)
from screener.screen_workflow import ScreenRequest, run_screen_workflow
from screener.strategies.spec import (
    CallableStrategySpec,
    ExpressionStrategySpec,
    discover_plugins,
    registry as strategy_registry,
)

#: The criterion names that also exist as a strategy. Stage 6 turns exactly
#: these into aliases; every other criterion keeps its old snapshot path.
ALIASED = frozenset({"breakout", "mark_minervini", "momentum_12_1"})


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

    @pytest.mark.parametrize("name", ["ema", "value", "near_52_high"])
    def test_a_filters_only_criterion_keeps_the_old_path(self, name: str) -> None:
        assert aliased_strategy(name) is None
        assert resolve_screen_strategy((name,)) is None

    def test_combining_filters_only_criteria_stays_legal(self) -> None:
        assert resolve_screen_strategy(("ema", "value")) is None

    def test_a_strategy_alias_refuses_to_be_combined(self) -> None:
        with pytest.raises(UnscreenableStrategyError, match="Screen one at a time"):
            resolve_screen_strategy(("breakout", "ema"))

    def test_two_strategy_aliases_refuse_each_other_too(self) -> None:
        with pytest.raises(UnscreenableStrategyError, match="Screen one at a time"):
            resolve_screen_strategy(("breakout", "momentum_12_1"))


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


class TestPrefilter:
    @pytest.mark.parametrize("name", sorted(ALIASED))
    def test_each_alias_declares_the_criterion_whose_filters_cut_the_field(
        self, name: str
    ) -> None:
        strategy = resolve_screen_strategy((name,))
        assert strategy is not None
        assert strategy.tv_prefilter == name

    @pytest.mark.parametrize("name", sorted(ALIASED))
    def test_the_prefilter_yields_that_criterion_s_own_filters(self, name: str) -> None:
        strategy = resolve_screen_strategy((name,))
        assert strategy is not None
        assert prefilter_filters(strategy) == list(criteria_registry.get(name)())

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

    def test_universe_without_a_strategy_alias_is_refused(self, monkeypatch) -> None:
        # A filters-only criterion has no bar rule, so there is nothing to run
        # against a local universe; failing loudly beats screening on nothing.
        def unexpected_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
            raise AssertionError("scan must not run once the request is refused")

        monkeypatch.setattr(screen_workflow, "scan", unexpected_scan)

        with pytest.raises(UnscreenableStrategyError, match="--universe needs"):
            run_screen_workflow(_request(criteria_names=("ema",), universe="nifty50"))


class TestWorkflowWiring:
    def test_the_bar_path_scans_with_the_prefilter_and_never_with_a_scorer(
        self, monkeypatch
    ) -> None:
        # The scan's only job in default mode is the field cut: ranking comes
        # from the bar rule, so asking TradingView to score would be dead work
        # on a scale the run does not use.
        captured: dict[str, object] = {}

        def fake_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
            captured.update(kwargs)
            return 2, pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB"]})

        def fake_candidates(strategy, **kwargs: object) -> pd.DataFrame:
            captured["tickers"] = list(kwargs["tickers"])
            captured["strategy"] = strategy.spec.name
            return pd.DataFrame({"name": ["AAA"], OUTPUT_SCORE_COLUMN: [100.0]})

        monkeypatch.setattr(screen_workflow, "scan", fake_scan)
        monkeypatch.setattr(screen_workflow, "screen_candidates", fake_candidates)

        outcome = run_screen_workflow(_request(criteria_names=("breakout",)))

        assert captured["scorer"] is None
        assert captured["filters"] == list(criteria_registry.get("breakout")())
        assert captured["tickers"] == ["NSE:AAA", "NSE:BBB"]
        assert captured["strategy"] == "breakout"
        assert outcome.label == "breakout@tv"
        assert outcome.total == 2

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
            "resolve_universe_tickers",
            lambda universe, market: ["NSE:AAA", "NSE:BBB", "NSE:CCC"],
        )

        outcome = run_screen_workflow(
            _request(criteria_names=("breakout",), universe="nifty50")
        )

        assert captured["scanned"] is None
        assert captured["tickers"] == ["NSE:AAA", "NSE:BBB", "NSE:CCC"]
        assert outcome.label == "breakout@universe:nifty50"
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


def _candidate(ticker: str, rank: int, score: float | None) -> Candidate:
    return Candidate(
        ticker=ticker,
        rank=rank,
        rank_basis="rank_score" if score is not None else "as_of_dollar_vol",
        rank_score=score,
        as_of_close=10.0,
        as_of_volume=1_000.0,
        as_of_dollar_vol=10_000.0 * rank,
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

    def test_the_score_column_is_a_percentile_of_what_was_actually_ranked(
        self,
    ) -> None:
        # setup_score describes the ranking rather than asserting a scale of its
        # own, so it must decrease monotonically down the rank order and stay
        # inside 0-100 regardless of the raw basis values.
        candidates = [
            _candidate("NSE:CCC", 1, 900.0),
            _candidate("NSE:AAA", 2, 5.0),
            _candidate("NSE:BBB", 3, 1.0),
        ]
        scanned = pd.DataFrame({"ticker": ["NSE:AAA", "NSE:BBB", "NSE:CCC"]})

        scores = _candidate_frame(candidates, {}, scanned)[OUTPUT_SCORE_COLUMN]

        assert scores.is_monotonic_decreasing
        assert scores.between(0.0, 100.0).all()

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
        assert row["name"] == "NSE:AAA"
        assert row["close"] == pytest.approx(110.0)
        assert row["change"] == pytest.approx(10.0)
        assert row["volume"] == pytest.approx(2_000.0)
        assert pd.isna(row["market_cap_basic"])

    def test_a_candidate_without_bars_is_dropped_rather_than_shown_blank(self) -> None:
        candidates = [_candidate("NSE:AAA", 1, 5.0), _candidate("NSE:GONE", 2, 4.0)]
        bars = {"NSE:AAA": _bars(close=110.0, previous=100.0, volume=2_000.0)}

        df = _candidate_frame(candidates, bars, None)

        assert df["name"].tolist() == ["NSE:AAA"]
        assert len(df) == 1

    def test_no_candidates_yields_the_empty_result_shape(self) -> None:
        df = _candidate_frame([], {}, None)

        assert df.empty
        assert OUTPUT_SCORE_COLUMN in df.columns
