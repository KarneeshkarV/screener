"""Unit tests for the generic multi-factor combo strategy factory."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import screener.strategies.combo as combo
from screener.backtester.models import BacktestConfig
from screener.backtester.core import prepare_strategy_bars
from screener.strategies.combo import (
    combine_rank_scores,
    cross_sectional_zscore,
    is_combo_strategy,
    parse_combo_spec,
    resolve_combo_spec,
    validate_combo_components,
)
from screener.strategies.expressions import resolve_strategy
from screener.strategies.spec import PrepareCtx, discover_plugins


def test_parse_combo_spec_valid() -> None:
    parts = parse_combo_spec("combo:momentum_12_1=0.6,low_volatility=0.4")
    assert parts == [("momentum_12_1", 0.6), ("low_volatility", 0.4)]


def test_parse_combo_spec_rejects_bad_weights() -> None:
    # 1e999 overflows to inf → rejected as non-finite.
    with pytest.raises(ValueError, match="finite"):
        parse_combo_spec("combo:momentum_12_1=1e999")
    with pytest.raises(ValueError, match="at least one"):
        parse_combo_spec("combo:")
    with pytest.raises(ValueError, match="invalid combo component"):
        parse_combo_spec("combo:momentum_12_1")


def test_validate_requires_known_factor_components() -> None:
    discover_plugins()
    with pytest.raises(ValueError, match="unknown combo component"):
        validate_combo_components([("does_not_exist", 1.0)])
    # Expression-only strategies without prepare_bars cannot be combined.
    with pytest.raises(ValueError, match="prepare_bars"):
        validate_combo_components([("breakout", 1.0)])


def test_resolve_strategy_accepts_combo_prefix() -> None:
    named = resolve_strategy("combo:momentum_12_1=0.5,low_volatility=0.5")
    assert "mom_12_1" in named.entry
    assert "vol_252" in named.entry
    assert named.exit is None
    with pytest.raises(KeyError, match="at least one component"):
        resolve_strategy("combo:")


def test_combo_resolution_parses_and_validates_once(monkeypatch) -> None:
    parse = combo.parse_combo_spec
    validate = combo.validate_combo_components
    calls = {"parse": 0, "validate": 0}

    def counted_parse(name: str):
        calls["parse"] += 1
        return parse(name)

    def counted_validate(components):
        calls["validate"] += 1
        return validate(components)

    monkeypatch.setattr(combo, "parse_combo_spec", counted_parse)
    monkeypatch.setattr(combo, "validate_combo_components", counted_validate)
    resolve_strategy("combo:momentum_12_1=0.5,low_volatility=0.5")
    assert calls == {"parse": 1, "validate": 1}


def test_is_combo_strategy() -> None:
    assert is_combo_strategy("combo:momentum_12_1=1")
    assert not is_combo_strategy("momentum_12_1")
    assert not is_combo_strategy(None)


def test_cross_sectional_zscore_and_weighted_blend() -> None:
    idx = pd.bdate_range("2024-01-02", periods=1)
    # Factor 1 scores: A=3, B=1 → z = +1, -1 (pop std 1, mean 2)
    f1 = pd.DataFrame({"A": [3.0], "B": [1.0]}, index=idx)
    # Factor 2 scores: A=1, B=3 → z = -1, +1
    f2 = pd.DataFrame({"A": [1.0], "B": [3.0]}, index=idx)
    z1 = cross_sectional_zscore(f1)
    assert z1.loc[idx[0], "A"] == pytest.approx(1.0)
    assert z1.loc[idx[0], "B"] == pytest.approx(-1.0)

    # 0.75 * f1 + 0.25 * f2 → A: 0.75*1 + 0.25*(-1) = 0.5
    #                         B: 0.75*(-1) + 0.25*1 = -0.5
    blended = combine_rank_scores([(f1, 0.75), (f2, 0.25)])
    assert blended.loc[idx[0], "A"] == pytest.approx(0.5)
    assert blended.loc[idx[0], "B"] == pytest.approx(-0.5)
    # Ranking by blended score: A > B
    assert blended.loc[idx[0]].idxmax() == "A"


def test_combo_prepare_writes_weighted_rank_score() -> None:
    discover_plugins()
    idx = pd.bdate_range("2022-01-03", periods=320)

    # Distinct growth/noise so momentum and low-vol disagree.
    def frame(growth: float, noise: float) -> pd.DataFrame:
        drift = 50.0 * (1.0 + growth) ** np.arange(len(idx))
        wiggle = noise * 50.0 * np.sin(np.arange(len(idx)))
        close = pd.Series(drift + wiggle, index=idx)
        openp = close.shift(1).fillna(close.iloc[0])
        return pd.DataFrame(
            {
                "open": openp,
                "high": pd.concat([openp, close], axis=1).max(axis=1) + 0.5,
                "low": pd.concat([openp, close], axis=1).min(axis=1) - 0.5,
                "close": close,
                "volume": pd.Series(100_000.0, index=idx),
            }
        )

    bars_by_tv = {
        "HIMOM": frame(0.0020, 0.03),
        "LOVOL": frame(0.0006, 0.002),
        "MID": frame(0.0012, 0.01),
    }
    name = "combo:momentum_12_1=1.0,low_volatility=0.0"
    spec = resolve_combo_spec(name)
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() >= 252

    ctx = PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv=bars_by_tv,
        price_panel={},
        tv_symbols=list(bars_by_tv),
        start=idx[0].date(),
        end=idx[-1].date(),
        fetcher=lambda *a, **k: {},
        warnings=[],
    )
    prepared = spec.prepare_bars(ctx)
    # Pure-momentum weights → highest rank_score should be HIMOM on late bars.
    late = idx[-1]
    scores = {
        tv: float(prepared[tv].loc[late, "rank_score"])
        for tv in bars_by_tv
        if math.isfinite(float(prepared[tv].loc[late, "rank_score"]))
    }
    assert scores
    assert max(scores, key=scores.get) == "HIMOM"  # type: ignore[arg-type]


def test_combo_parser_and_validation_edge_cases(monkeypatch) -> None:
    with pytest.raises(ValueError, match="not a combo"):
        parse_combo_spec("momentum_12_1=1")
    with pytest.raises(ValueError, match="at least one"):
        parse_combo_spec("combo:,,")
    with pytest.raises(ValueError, match="finite"):
        validate_combo_components([("factor", float("nan"))])
    with pytest.raises(ValueError, match="nested"):
        validate_combo_components([("combo:nested", 1.0)])
    with pytest.raises(ValueError, match="at least one"):
        validate_combo_components([])

    fake_registry = SimpleNamespace(
        get_optional=lambda name: SimpleNamespace(
            prepare_bars=lambda ctx: {}, entry=None
        ),
        names=lambda: ["factor"],
    )
    monkeypatch.setattr(combo, "registry", fake_registry)
    monkeypatch.setattr(combo, "discover_plugins", lambda: None)
    with pytest.raises(ValueError, match="no entry expression"):
        validate_combo_components([("factor", 1.0)])


def test_combo_empty_matrices_and_component_extraction() -> None:
    empty = pd.DataFrame()
    assert cross_sectional_zscore(empty).empty
    with pytest.raises(ValueError, match="at least one component"):
        combine_rank_scores([])
    extracted = combo._component_score_matrix(
        {
            "NONE": None,  # type: ignore[dict-item]
            "EMPTY": empty,
            "NO_SCORE": pd.DataFrame({"close": [1.0]}),
            "SCORED": pd.DataFrame({"rank_score": [2.0]}),
        }
    )
    assert list(extracted.columns) == ["SCORED"]


def test_prepare_strategy_bars_reports_invalid_combo() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=1,
        top=1,
        strategy_name="combo:",
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=1_000.0,
        benchmark="SPY",
        tickers=("A",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    bars = {"A": pd.DataFrame({"close": [1.0, 2.0]}, index=idx)}
    warnings: list[str] = []
    prepared, lookback = prepare_strategy_bars(
        cfg.strategy_name,
        bars,
        bars,
        ["A"],
        idx[0].date(),
        idx[-1].date(),
        object(),  # type: ignore[arg-type]
        warnings,
        market=cfg.market,
        benchmark=cfg.benchmark,
    )
    assert prepared is bars and lookback == 0
    assert warnings and "strategy error" in warnings[0]


def test_combo_prepare_handles_missing_component_outputs(monkeypatch) -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    base = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)

    def empty_prepare(ctx):
        return {"A": pd.DataFrame(index=idx)}

    def scored_prepare(ctx):
        return {
            "A": pd.DataFrame({"rank_score": [1.0, 2.0], "aux": [3.0, 4.0]}, index=idx),
            "EXTRA": pd.DataFrame({"rank_score": [2.0, 1.0]}, index=idx),
        }

    specs = {
        "empty": SimpleNamespace(
            prepare_bars=empty_prepare, entry="close > 0", required_lookback=None
        ),
        "scored": SimpleNamespace(
            prepare_bars=scored_prepare, entry="close > 0", required_lookback=None
        ),
    }

    class Registry:
        def get_optional(self, name):
            return specs.get(name)

        def get(self, name):
            return specs[name]

        def names(self):
            return specs.keys()

    monkeypatch.setattr(combo, "registry", Registry())
    monkeypatch.setattr(combo, "discover_plugins", lambda: None)
    prepare = combo.make_combo_prepare([("empty", 0.0), ("scored", 1.0)])
    warnings: list[str] = []
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=1,
        top=1,
        strategy_name=None,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=1_000.0,
        benchmark="SPY",
        tickers=("A", "MISSING"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    ctx = PrepareCtx(
        market=cfg.market,
        benchmark=cfg.benchmark,
        bars_by_tv={"A": base, "MISSING": base.copy(), "EMPTY": pd.DataFrame()},
        price_panel={},
        tv_symbols=["A", "MISSING"],
        start=idx[0].date(),
        end=idx[-1].date(),
        fetcher=object(),
        warnings=warnings,
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    monkeypatch.setattr(combo, "PrepareCtx", lambda **kwargs: SimpleNamespace(**kwargs))
    out = prepare(ctx)
    assert any("produced no rank_score" in warning for warning in ctx.warnings)
    assert "aux" in out["A"].columns
    assert out["MISSING"]["rank_score"].isna().all()
    assert out["NONE"] is None and out["EMPTY"].empty
