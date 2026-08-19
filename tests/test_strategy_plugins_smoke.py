"""Generic smoke coverage for every registered expression strategy.

Each plugin file registers an expression strategy whose ``prepare_bars`` hook
attaches the derived columns its entry/exit rules reference. Those hooks are
plain pandas over the OHLCV panel, so one synthetic multi-symbol panel
exercises all of them. The per-plugin test modules pin the *semantics* of the
signals worth pinning; this module pins the contract every plugin shares:

- ``required_lookback()`` returns a sane positive bar count,
- ``prepare_bars`` returns one frame per input symbol, preserving the index,
- the frames survive an empty / missing-benchmark panel without raising,
- the entry and exit expressions parse and evaluate against the prepared
  frames, yielding a boolean signal.

Registering a new plugin therefore gets this contract checked for free, and a
hook that raises on a degenerate panel fails here rather than mid-backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import evaluate, parse
from screener.strategies.spec import (
    ExpressionStrategySpec,
    PrepareCtx,
    discover_plugins,
    registry,
)

_BARS = 900
_SYMBOLS = ("AAA", "BBB", "CCC")
_BENCHMARK = "SPY"

# Fundamentals are merged onto the bars by the backtester *after* prepare_bars
# runs, so the hooks never see them -- but the entry/exit expressions do. Held
# constant per symbol: this module checks the expressions evaluate, not what
# they decide.
_FUNDAMENTAL_COLUMNS = {
    "pe_ttm": 18.0,
    "pb_ttm": 2.5,
    "roe_ttm": 0.18,
    "debt_to_equity": 0.4,
    "eps_growth_yoy": 0.15,
    "revenue_growth_yoy": 0.12,
    "revenue_up_3q": 1.0,
    "gross_margin_ttm": 0.35,
    "gross_profit_to_assets": 0.28,
    "fcf_yield": 0.05,
    "operating_cash_flow": 1.0e9,
    "accruals": -0.02,
    "asset_growth": 0.06,
    "piotroski_fscore": 7.0,
}


def _bars(seed: int, n: int = _BARS) -> pd.DataFrame:
    """A trending series with cyclical pullbacks, indexed by date."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 24, n)
    drift = np.linspace(0, 60 + seed * 10, n)
    close = 100.0 + drift + np.sin(x) * 9 + rng.normal(0, 0.6, n)
    frame = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.3, n),
            "high": close + np.abs(rng.normal(1.5, 0.4, n)),
            "low": close - np.abs(rng.normal(1.5, 0.4, n)),
            "close": close,
            "adj_close": close,
            "volume": rng.uniform(5e5, 5e6, n),
        },
        index=pd.date_range("2021-01-01", periods=n, freq="B", name="date"),
    )
    for column, value in _FUNDAMENTAL_COLUMNS.items():
        frame[column] = value
    return frame


@pytest.fixture(scope="module")
def panel() -> dict[str, pd.DataFrame]:
    return {symbol: _bars(i + 1) for i, symbol in enumerate(_SYMBOLS)}


@pytest.fixture(scope="module")
def benchmark_bars() -> pd.DataFrame:
    return _bars(0)


def _ctx(
    bars_by_tv: dict[str, pd.DataFrame],
    price_panel: dict[str, pd.DataFrame],
    *,
    market: str = "us",
) -> PrepareCtx:
    index = next(
        (b.index for b in bars_by_tv.values() if b is not None and not b.empty),
        pd.DatetimeIndex([pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")]),
    )
    return PrepareCtx(
        market=market,
        benchmark=_BENCHMARK,
        bars_by_tv=bars_by_tv,
        price_panel=price_panel,
        tv_symbols=list(bars_by_tv),
        start=index[0].date(),
        end=index[-1].date(),
        fetcher=lambda *_args, **_kwargs: {},
        warnings=[],
    )


def _expression_specs() -> list[ExpressionStrategySpec]:
    discover_plugins()
    return sorted(
        (
            spec
            for _name, spec in registry.items()
            if isinstance(spec, ExpressionStrategySpec)
        ),
        key=lambda spec: spec.name,
    )


def _prepared_specs() -> list[ExpressionStrategySpec]:
    return [spec for spec in _expression_specs() if spec.prepare_bars is not None]


def _ids(specs: list[ExpressionStrategySpec]) -> list[str]:
    return [spec.name for spec in specs]


_ALL = _expression_specs()
_WITH_PREPARE = _prepared_specs()


def test_plugin_discovery_registers_every_expression_strategy() -> None:
    # A plugin file that fails to import would silently drop its strategies
    # from the registry, so the parametrized cases below would just vanish.
    assert len(_ALL) >= 90
    assert len(_WITH_PREPARE) >= 50


@pytest.mark.parametrize("spec", _ALL, ids=_ids(_ALL))
def test_expression_strategy_rules_parse(spec: ExpressionStrategySpec) -> None:
    assert parse(spec.entry) is not None
    if spec.exit:
        assert parse(spec.exit) is not None


@pytest.mark.parametrize("spec", _WITH_PREPARE, ids=_ids(_WITH_PREPARE))
def test_required_lookback_is_a_sane_bar_count(spec: ExpressionStrategySpec) -> None:
    if spec.required_lookback is None:
        pytest.skip(f"{spec.name} declares no lookback")
    lookback = spec.required_lookback()
    assert isinstance(lookback, int)
    # Under one bar cannot warm any rolling window; over the synthetic panel
    # length would make the evaluation case below vacuous.
    assert 1 <= lookback <= _BARS


@pytest.mark.parametrize("spec", _WITH_PREPARE, ids=_ids(_WITH_PREPARE))
def test_prepare_bars_returns_aligned_frames(
    spec: ExpressionStrategySpec,
    panel: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
) -> None:
    assert spec.prepare_bars is not None
    ctx = _ctx(dict(panel), {_BENCHMARK: benchmark_bars})

    prepared = spec.prepare_bars(ctx)

    assert set(prepared) == set(panel)
    for symbol, frame in prepared.items():
        assert isinstance(frame, pd.DataFrame), symbol
        # Hooks decorate the bars in place; dropping or reordering rows would
        # desynchronize the signal from the price panel the backtester fills on.
        assert frame.index.equals(panel[symbol].index), symbol
        assert set(panel[symbol].columns) <= set(frame.columns), symbol


@pytest.mark.parametrize("spec", _WITH_PREPARE, ids=_ids(_WITH_PREPARE))
def test_prepared_frames_evaluate_entry_and_exit(
    spec: ExpressionStrategySpec,
    panel: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
) -> None:
    assert spec.prepare_bars is not None
    ctx = _ctx(dict(panel), {_BENCHMARK: benchmark_bars})
    prepared = spec.prepare_bars(ctx)

    for symbol, frame in prepared.items():
        for rule in (spec.entry, spec.exit):
            if not rule:
                continue
            signal = evaluate(parse(rule), frame)
            assert len(signal) == len(frame), f"{spec.name}/{symbol}"
            assert signal.astype(bool).notna().all(), f"{spec.name}/{symbol}"


@pytest.mark.parametrize("spec", _WITH_PREPARE, ids=_ids(_WITH_PREPARE))
def test_prepare_bars_tolerates_empty_and_missing_data(
    spec: ExpressionStrategySpec,
) -> None:
    # The degenerate panel a real run hits when a symbol is freshly listed, is
    # suspended, or the benchmark fetch fails: hooks must warn, not raise.
    empty = pd.DataFrame()
    ctx = _ctx({"AAA": empty, "BBB": _bars(4)}, {})

    prepared = spec.prepare_bars(ctx)  # type: ignore[misc]

    # Cross-sectional rankers drop the unrankable symbol rather than pass an
    # empty frame through, so the output is a subset -- but the symbol that
    # does have bars must always survive, decorated and index-aligned.
    assert set(prepared) <= {"AAA", "BBB"}
    assert "BBB" in prepared
    assert prepared["BBB"].index.equals(ctx.bars_by_tv["BBB"].index)
    if "AAA" in prepared:
        assert prepared["AAA"].empty
