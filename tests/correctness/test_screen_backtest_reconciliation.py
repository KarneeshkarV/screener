"""Stage 5 of ``docs/plans/unify-screen-backtest.md``: the two paths reconcile.

Two guarantees, both driven off the live strategy registry so a new plugin
cannot forget them.

Criterion 1: on one golden fixture panel and one fixed as-of date, the screen's
one-day entry point (:func:`build_day_candidates`) returns exactly the rolling
engine's candidate rows for that date. The rolling engine is the reference
(D7), so the expected side is read out of the matrices
:func:`prepare_rolling_backtest` built, never re-derived here.

Criterion 2: a strategy's declared ``tv_prefilter`` never drops a name the bar
rules kept. The prefilter is an optimisation, never a rule, so containment is
one-sided on purpose: a wider prefilter is fine, a narrower one is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.price_panel import (
    PricePanel,
    PricePanelInputs,
    build_price_panel,
)
from screener.backtester.rolling_candidates import _candidate_rows_for_day
from screener.backtester.rolling_simulation import prepare_rolling_backtest
from screener.backtester.signal_panel import (
    RUN_SCOPED_SIGNAL_PANEL_FIELDS,
    SIGNAL_PANEL_INPUT_FIELDS,
    DayCandidates,
    SignalPanelInputs,
    SignalProgram,
    build_day_candidates,
    build_signal_panel,
    day_candidates_from_panel,
    parse_signal_program,
)
from screener.criteria import registry as criteria_registry
from screener.strategies.spec import (
    ExpressionStrategySpec,
    StrategySpec,
    discover_plugins,
    registry as strategy_registry,
    resolve_strategy_profile,
)
from tests.conftest import StubPriceFetcher

# ---------------------------------------------------------------------------
# The golden fixture panel
# ---------------------------------------------------------------------------

_MARKET = "us"
_BENCHMARK = "SPY"
_TICKERS = (
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
    "GGG",
    "HHH",
    "III",
    "JJJ",
    "KKK",
    "LLL",
)
# The longest declared warm-up in the registry is 350 bars (``bb_breakout``),
# and ``build_price_panel`` buys ``3 * lookback + 30`` calendar days of history
# before the window start. 900 bars covers that with room to spare, so no
# strategy silently evaluates over an all-NaN indicator column.
_BAR_COUNT = 900
_WINDOW_BARS = 60
_AS_OF_OFFSET = 30


def _fixture_bars(seed: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One ticker's OHLCV walk: seeded, so the panel is a fixed golden input.

    The per-ticker drift and volatility come off the seed so the cross-section
    holds both long uptrends and chop. Without that spread the trend and
    breakout strategies would find nobody and the equality below would hold
    vacuously.
    """
    rng = np.random.default_rng(1000 + seed)
    n = len(dates)
    drift = 0.0012 - 0.0004 * (seed % 5)
    vol = 0.010 + 0.004 * (seed % 3)
    returns = rng.normal(drift, vol, n)
    close = 50.0 * (1.0 + seed * 0.1) * np.exp(np.cumsum(returns))
    openp = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(openp, close) * (1.0 + rng.uniform(0.001, 0.008, n))
    low = np.minimum(openp, close) * (1.0 - rng.uniform(0.001, 0.008, n))
    volume = rng.uniform(500_000.0, 3_000_000.0, n) * (1.0 + 0.1 * seed)
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def fixture_panel() -> dict[str, pd.DataFrame]:
    """The golden panel, built once: every strategy runs over the same bars."""
    dates = pd.bdate_range("2019-01-01", periods=_BAR_COUNT)
    bars = {tv: _fixture_bars(i, dates) for i, tv in enumerate(_TICKERS)}
    bars[_BENCHMARK] = _fixture_bars(len(_TICKERS), dates)
    return bars


@pytest.fixture(scope="module")
def fetcher(fixture_panel: dict[str, pd.DataFrame]) -> StubPriceFetcher:
    return StubPriceFetcher(fixture_panel)


@pytest.fixture(scope="module")
def window(fixture_panel: dict[str, pd.DataFrame]) -> tuple[date, date, pd.Timestamp]:
    """``(start_date, end_date, as_of)``: a short window late in the fixture.

    The window is short because candidate equality is a per-day question; the
    history in front of it is what the long warm-ups need.
    """
    index = fixture_panel[_TICKERS[0]].index
    return (
        index[-_WINDOW_BARS].date(),
        index[-1].date(),
        pd.Timestamp(index[-_AS_OF_OFFSET]),
    )


# ---------------------------------------------------------------------------
# Registry-driven parametrisation
# ---------------------------------------------------------------------------


def _registered_specs() -> list[tuple[str, StrategySpec]]:
    discover_plugins()
    return sorted(strategy_registry.items())


_SPECS = _registered_specs()
_EXPRESSION_SPECS = [
    pytest.param(spec, id=name) for name, spec in _SPECS if spec.kind == "expression"
]
_CALLABLE_NAMES = [name for name, spec in _SPECS if spec.kind == "callable"]

# A registry that lost most of its strategies would make the sweep look green
# while covering almost nothing.
assert len(_EXPRESSION_SPECS) >= 30, "expression-strategy sweep shrank unexpectedly"


def _gate_values(spec: ExpressionStrategySpec) -> dict[str, object]:
    """The candidate gates both paths must be handed, keyed by panel field name.

    Derived from :data:`SIGNAL_PANEL_INPUT_FIELDS` rather than listed by hand,
    so a gate added to ``SignalPanelInputs`` lands in both the screen inputs
    and the backtest config here instead of quietly defaulting on one side.
    ``BacktestConfig`` shares those field names, which is what lets one mapping
    feed both.
    """
    profile = resolve_strategy_profile(spec)
    run_scoped: dict[str, object] = {
        "market": _MARKET,
        "membership_added": (),
        "membership_windows": (),
        "dynamic_universe_size": None,
        "dynamic_universe_lookback": 60,
        "dynamic_universe_rebalance": "monthly",
    }
    values: dict[str, object] = {}
    for field in sorted(SIGNAL_PANEL_INPUT_FIELDS):
        if field in RUN_SCOPED_SIGNAL_PANEL_FIELDS:
            values[field] = run_scoped[field]
        else:
            values[field] = getattr(profile, field)
    # The profile leaves these ``None`` to mean "the spec's own rules stand".
    values["entry_expr"] = profile.entry_expr or spec.entry
    values["exit_expr"] = profile.exit_expr or spec.exit
    return values


@dataclass(frozen=True)
class _ScreenSide:
    """The screen's resolved gates and bars, reusable across as-of dates."""

    inputs: SignalPanelInputs
    program: SignalProgram
    panel: PricePanel
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    warnings: list[str]


def _backtest_config(spec: ExpressionStrategySpec, as_of: date) -> BacktestConfig:
    """A rolling-engine config carrying exactly the gates the screen gets."""
    return BacktestConfig(
        **_gate_values(spec),
        as_of=as_of,
        benchmark=_BENCHMARK,
        tickers=_TICKERS,
        strategy_name=spec.name,
        hold=5,
        top=5,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
    )


def _screen_panel(
    spec: ExpressionStrategySpec,
    fetcher: StubPriceFetcher,
    start_date: date,
    end_date: date,
    as_of: pd.Timestamp,
) -> _ScreenSide:
    """Everything the screen resolves before it asks for a day: gates and bars."""
    cfg = _backtest_config(spec, as_of.date())
    inputs = SignalPanelInputs(**_gate_values(spec))  # type: ignore[arg-type]
    program = parse_signal_program(inputs)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    warnings: list[str] = []
    panel = build_price_panel(
        PricePanelInputs.from_config(cfg),
        fetcher,
        entry_ast=program.entry_ast,
        exit_ast=program.exit_ast,
        lookback=program.lookback,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
    )
    return _ScreenSide(
        inputs=inputs,
        program=program,
        panel=panel,
        start_ts=start_ts,
        end_ts=end_ts,
        warnings=warnings,
    )


def _screen_candidates(
    spec: ExpressionStrategySpec,
    fetcher: StubPriceFetcher,
    start_date: date,
    end_date: date,
    as_of: pd.Timestamp,
) -> DayCandidates:
    """The screen's answer: the unified one-day entry point, own panel and all."""
    side = _screen_panel(spec, fetcher, start_date, end_date, as_of)
    return build_day_candidates(
        side.inputs,
        side.panel,
        program=side.program,
        as_of=as_of,
        start_ts=side.start_ts,
        end_ts=side.end_ts,
        warnings=side.warnings,
    )


def _rolling_candidate_rows(
    spec: ExpressionStrategySpec,
    fetcher: StubPriceFetcher,
    start_date: date,
    end_date: date,
    as_of: pd.Timestamp,
) -> list[dict]:
    """The reference answer: the rolling engine's own rows for ``as_of``."""
    prepared = prepare_rolling_backtest(
        _backtest_config(spec, as_of.date()),
        fetcher,
        start_date=start_date,
        end_date=end_date,
    )
    matrices = prepared.candidate_matrices
    if matrices is None:
        return []
    rows, _ = _candidate_rows_for_day(as_of, matrices, exclude=set(), limit=None)
    return list(rows)


# ---------------------------------------------------------------------------
# Criterion 1 - the screen's candidate set equals the rolling engine's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _EXPRESSION_SPECS)
def test_screen_candidates_equal_the_rolling_engine(
    spec: ExpressionStrategySpec,
    fetcher: StubPriceFetcher,
    window: tuple[date, date, pd.Timestamp],
    sweep_record: dict[str, set[str]],
) -> None:
    """Every registered expression strategy agrees, field for field, on one day."""
    start_date, end_date, as_of = window
    expected = _rolling_candidate_rows(spec, fetcher, start_date, end_date, as_of)
    day = _screen_candidates(spec, fetcher, start_date, end_date, as_of)

    sweep_record["ran"].add(spec.name)
    assert day.as_of == as_of
    assert [c.ticker for c in day.candidates] == [str(r["ticker"]) for r in expected]
    for got, want in zip(day.candidates, expected, strict=True):
        assert got.rank == want["rank"]
        assert got.role == want["role"]
        assert got.signal_idx == want["signal_idx"]
        assert got.as_of_close == pytest.approx(want["as_of_close"])
        assert got.as_of_volume == pytest.approx(want["as_of_volume"])
        assert got.as_of_dollar_vol == pytest.approx(want["as_of_dollar_vol"])
    if day.candidates:
        sweep_record["non_empty"].add(spec.name)


@pytest.fixture(scope="module")
def sweep_record() -> dict[str, set[str]]:
    """Which strategies the sweep ran, and which of them fired."""
    return {"ran": set(), "non_empty": set()}


def test_the_fixture_makes_the_equality_sweep_non_vacuous(
    sweep_record: dict[str, set[str]],
) -> None:
    """A panel where nothing ever fires would prove nothing above.

    Declared after the sweep so it reads the finished record. The floor guards
    against a change that silently empties every candidate set; it does not pin
    which strategies fire. A partial run (``-k``) skips instead of failing,
    because the record is then incomplete rather than alarming.
    """
    if len(sweep_record["ran"]) < len(_EXPRESSION_SPECS):
        pytest.skip("equality sweep was filtered; the record is incomplete")
    assert len(sweep_record["non_empty"]) >= 12, (
        "the fixture panel produced candidates for only "
        f"{sorted(sweep_record['non_empty'])}; the equality sweep is near-vacuous"
    )


def test_callable_strategies_are_skipped_by_kind_not_by_name() -> None:
    """The four non-convertible callables have no entry expression to reconcile.

    Stage 3 left them callable because their trade generation is not a per-bar
    boolean, and stage 6 rejects them at screen time. They are excluded by
    ``kind`` above, so converting one later pulls it into the sweep with no
    edit here.
    """
    assert _CALLABLE_NAMES, "the callable escape hatch is untested if none exist"
    for name in _CALLABLE_NAMES:
        assert not isinstance(strategy_registry.get(name), ExpressionStrategySpec)


# ---------------------------------------------------------------------------
# Criterion 2 - the TradingView prefilter never drops a bar-rule candidate
# ---------------------------------------------------------------------------


def _snapshot_row(bars: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """The vendor snapshot columns the declared prefilters read, from fixture bars.

    Computed from the same bars the bar rules see so the comparison is about
    the two rule spellings, not about two data sources. ``price_52_week_*`` are
    high/low extremes, and ``Perf.*`` are trailing-window returns on 252/21
    trading days, which is the closest bar-side reading of TradingView's
    calendar anchors.
    """
    history = bars.loc[bars.index <= as_of]
    close = history["close"].astype(float)
    return {
        "close": float(close.iloc[-1]),
        "volume": float(history["volume"].iloc[-1]),
        "average_volume_10d_calc": float(history["volume"].iloc[-10:].mean()),
        "price_52_week_high": float(history["high"].iloc[-252:].max()),
        "price_52_week_low": float(history["low"].iloc[-252:].min()),
        "SMA50": float(close.iloc[-50:].mean()),
        "SMA150": float(close.iloc[-150:].mean()),
        "SMA200": float(close.iloc[-200:].mean()),
        # ``iloc[-1 - k]`` is bar ``t - k``, matching how the 12-1 recipe reads
        # its two legs. Reading ``iloc[-252]`` instead is bar ``t - 251`` and
        # shifts the whole comparison by one session.
        "Perf.Y": float(close.iloc[-1] / close.iloc[-1 - 252] - 1.0),
        "Perf.1M": float(close.iloc[-1] / close.iloc[-1 - 21] - 1.0),
    }


def _resolve_operand(row: dict[str, float], operand: object) -> float:
    """A filter's ``right``/``left`` is either a snapshot column or a literal."""
    if isinstance(operand, str):
        if operand not in row:
            raise KeyError(f"prefilter reads unmodelled snapshot column {operand!r}")
        return row[operand]
    return float(operand)  # type: ignore[arg-type]


def _passes_filter(row: dict[str, float], filt: dict) -> bool:
    """Evaluate one TradingView filter dict against a snapshot row.

    Unknown operations raise rather than pass. A criterion that grows an
    operation this interpreter does not model must fail loudly here, otherwise
    the containment guarantee below silently stops covering it.
    """
    operation = filt["operation"]
    left = _resolve_operand(row, filt["left"])
    right = filt["right"]
    if operation == "greater":
        return left > _resolve_operand(row, right)
    if operation == "above%":
        column, pct = right
        return left >= _resolve_operand(row, column) * float(pct)
    raise NotImplementedError(
        f"prefilter operation {operation!r} is not modelled by this test; "
        "add it here rather than letting the filter go un-evaluated"
    )


def _prefilter_survivors(
    criterion_name: str, snapshots: dict[str, dict[str, float]]
) -> set[str]:
    filters = criteria_registry.get(criterion_name)()
    return {
        ticker
        for ticker, row in snapshots.items()
        if all(_passes_filter(row, dict(f)) for f in filters)
    }


# Known-narrow prefilters: a defect this test found, recorded rather than
# hidden. ``strict=True`` means fixing one turns this file red, so the entry
# has to be removed with the fix.
#
# ``breakout`` fronts ``close >= highest(close, 252) * 0.9`` with the vendor
# column ``price_52_week_high``, which is the extreme of *highs*. That extreme
# is never below the extreme of closes, so the vendor threshold sits above the
# rule's and drops every name inside the band between them. Fixing it means
# picking one 52-week high for both spellings, which moves backtest numbers and
# so belongs to stage 6, not to this test file.
_NARROW_PREFILTERS = {
    "breakout": "prefilter reads the 52-week high of highs, the rule uses closes",
}

_PREFILTERED_SPECS = [
    pytest.param(
        spec,
        id=name,
        marks=(
            [pytest.mark.xfail(reason=_NARROW_PREFILTERS[name], strict=True)]
            if name in _NARROW_PREFILTERS
            else []
        ),
    )
    for name, spec in _SPECS
    if spec.kind == "expression" and resolve_strategy_profile(spec).tv_prefilter
]

# Three strategies declare one today. A drop to zero would make the sweep
# silently empty, which is exactly the failure this file exists to prevent.
assert _PREFILTERED_SPECS, "no strategy declares a tv_prefilter"


@pytest.mark.parametrize("spec", _PREFILTERED_SPECS)
def test_prefilter_keeps_every_bar_rule_candidate(
    spec: ExpressionStrategySpec,
    fetcher: StubPriceFetcher,
    fixture_panel: dict[str, pd.DataFrame],
    window: tuple[date, date, pd.Timestamp],
) -> None:
    """The vendor field cut may be wider than the bar rules, never narrower.

    Every bar in the window is checked, not just the as-of date. The gap this
    hunts for is a threshold one, for example a 90% band read off the 52-week
    high of *highs* rather than of closes: it only shows up on the names sitting
    inside the band, so one day of one panel is too small a sample.
    """
    start_date, end_date, as_of = window
    criterion_name = resolve_strategy_profile(spec).tv_prefilter
    assert criterion_name is not None

    side = _screen_panel(spec, fetcher, start_date, end_date, as_of)
    signals = build_signal_panel(
        side.inputs,
        side.panel,
        program=side.program,
        start_ts=side.start_ts,
        end_ts=side.end_ts,
        warnings=side.warnings,
    )
    assert signals.candidate_matrices is not None

    violations: list[str] = []
    kept_total = 0
    for day in side.panel.master_dates:
        kept = {c.ticker for c in day_candidates_from_panel(signals, day).candidates}
        if not kept:
            continue
        kept_total += len(kept)
        snapshots = {tv: _snapshot_row(fixture_panel[tv], day) for tv in kept}
        survivors = _prefilter_survivors(criterion_name, snapshots)
        for ticker in sorted(kept - survivors):
            violations.append(f"{day.date()} {ticker}")

    assert kept_total, f"{spec.name} kept nobody in the window; the sweep is vacuous"
    assert not violations, (
        f"criterion {criterion_name!r} drops {len(violations)} name-days the "
        f"{spec.name} bar rules keep ({violations[:10]}): the prefilter is "
        "narrower than the rule it fronts"
    )
