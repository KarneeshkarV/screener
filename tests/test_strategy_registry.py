from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from screener.strategies.plugins.rs_breakout import (
    _prepare_rs_breakout,
    _rs_breakout_lookback,
)
from screener.strategies.plugins.vivek_equity_tool import (
    _prepare_vivek,
    _vivek_lookback,
)
from screener.strategies import spec as strategy_spec_module
from screener.strategies.plugins.ma_cross_st_entry import (
    strat_ma_cross_st_entry,
)
from screener.strategies.expressions import NAMED_STRATEGIES, resolve_strategy
from screener.strategies.registry import STRATEGIES
from screener.strategies.spec import (
    CallableStrategySpec,
    DerivedView,
    ExpressionStrategySpec,
    PrepareCtx,
    register_expression_strategy,
    strategy,
)


def test_strategy_registry_preserves_pine_runner_names():
    # Names ported from the pine runner are part of its public vocabulary:
    # saved configs and CLI invocations reference them, so a rename or a
    # dropped registration is a breaking change. Only their continued presence
    # is pinned -- asserting set equality would additionally forbid registering
    # any *new* callable strategy, which is ordinary growth rather than a
    # regression. test_all_callable_strategy_plugins_smoke covers whatever else
    # lands in the registry.
    pine_runner_names = {
        "bb_breakout",
        "ma_cross",
        "ma_cross_regime",
        "ma_cross_st_entry",
        "ma_cross_st_exit",
        "macd_rsi",
        "rsi_ema",
        "supertrend",
        "supertrend_rsi",
        "awesome_oscillator",
        "bb_pattern",
        "heikin_ashi",
        "macd_oscillator",
        "parabolic_sar",
        "rsi_pattern",
        "shooting_star",
    }

    assert pine_runner_names <= set(STRATEGIES)
    assert dict(STRATEGIES.items()) == dict(STRATEGIES)


def test_strategy_views_are_live_derived_over_single_registry():
    # Both public views are read-only projections of spec.registry, not stored
    # dicts, so a late registration is reflected without any rebuild.
    assert isinstance(STRATEGIES, DerivedView)
    assert isinstance(NAMED_STRATEGIES, DerivedView)

    name = "unit_live_view_probe"

    register_expression_strategy(name, entry="close > 0")

    # Visible immediately in the expression view (and its resolver)...
    assert name in NAMED_STRATEGIES
    assert resolve_strategy(name).entry == "close > 0"
    # ...but partitioned out of the callable view, since it has no callable_fn.
    assert name not in STRATEGIES


def test_strategy_registry_lookup_returns_callable():
    strategy = STRATEGIES["ma_cross_st_entry"]

    assert callable(strategy)

    with pytest.raises(KeyError, match="missing"):
        STRATEGIES["missing"]


def _ohlcv(n: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    x = np.linspace(0, 18, n)
    close = 100 + np.linspace(0, 80, n) + np.sin(x) * 8
    high = close + 1.5
    low = close - 1.5
    open_ = close + np.sin(x / 2) * 0.5
    volume = np.full(n, 10_000.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


def test_ma_cross_st_entry_smoke():
    trades = strat_ma_cross_st_entry(_ohlcv())

    assert isinstance(trades, list)
    assert all(trade.entry_idx <= trade.exit_idx for trade in trades)


def test_all_callable_strategy_plugins_smoke():
    bars = _ohlcv()

    for name, strategy_fn in STRATEGIES.items():
        trades = strategy_fn(bars)
        assert isinstance(trades, list), name
        assert all(trade.entry_idx <= trade.exit_idx for trade in trades), name


def test_strategy_spec_validation_and_decorator_metadata():
    with pytest.raises(ValidationError, match="strategy name must not be empty"):
        ExpressionStrategySpec(name=" ", entry="close > 0")
    with pytest.raises(ValidationError, match="strategy entry must not be empty"):
        ExpressionStrategySpec(name="empty", entry=" ")

    reg_size = len(strategy_spec_module.registry)

    expression = register_expression_strategy("unit_test_strategy", entry=" close > 0 ")

    assert expression.entry == "close > 0"
    assert isinstance(expression, ExpressionStrategySpec)
    assert len(strategy_spec_module.registry) == reg_size + 1


def test_callable_strategy_decorator_builds_explicit_callable_spec():
    name = "unit_test_callable_spec"

    @strategy(name)
    def callable_strategy(frame: pd.DataFrame) -> list:
        return []

    spec = strategy_spec_module.registry.get(name)
    assert isinstance(spec, CallableStrategySpec)
    assert spec.callable_fn is callable_strategy


def test_rs_breakout_prepare_handles_missing_benchmark():
    ctx = _prepare_ctx(price_panel={"SPY": pd.DataFrame()})

    prepared = _prepare_rs_breakout(ctx)

    assert prepared == ctx.bars_by_tv
    assert ctx.warnings == ["benchmark data unavailable for rs_breakout: SPY"]


def test_rs_breakout_prepare_uses_delivery_for_india(monkeypatch):
    bars = _ohlcv(50)
    delivery = pd.DataFrame({"symbol": ["AAA"]})
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "screener.rs_breakout.india_symbol",
        lambda symbol: f"NSE:{symbol}",
    )
    monkeypatch.setattr(
        "screener.unusual_volume.delivery.load_delivery_panel",
        lambda symbols, end, history_days: delivery,
    )

    def fake_prepare(bars_by_tv, benchmark_bars, *, market, delivery_panel):
        calls.append(("prepare", (benchmark_bars, market, delivery_panel)))
        return {"AAA": bars.assign(rs_breakout_entry=1)}

    monkeypatch.setattr("screener.rs_breakout.prepare_backtest_frames", fake_prepare)

    ctx = _prepare_ctx(
        market="india",
        bars_by_tv={"AAA": bars},
        price_panel={"^NSEI": bars},
        benchmark="^NSEI",
    )

    prepared = _prepare_rs_breakout(ctx)

    assert prepared["AAA"]["rs_breakout_entry"].iloc[0] == 1
    assert calls == [("prepare", (bars, "india", delivery))]
    assert ctx.warnings == []


def test_rs_breakout_prepare_warns_when_delivery_load_fails(monkeypatch):
    bars = _ohlcv(50)

    monkeypatch.setattr("screener.rs_breakout.india_symbol", lambda symbol: symbol)
    monkeypatch.setattr(
        "screener.unusual_volume.delivery.load_delivery_panel",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        "screener.rs_breakout.prepare_backtest_frames",
        lambda bars_by_tv, benchmark_bars, *, market, delivery_panel: bars_by_tv,
    )

    ctx = _prepare_ctx(
        market="india",
        bars_by_tv={"AAA": bars},
        price_panel={"^NSEI": bars},
        benchmark="^NSEI",
    )

    assert _prepare_rs_breakout(ctx) == {"AAA": bars}
    assert ctx.warnings == ["delivery panel unavailable for rs_breakout: offline"]


def test_strategy_prepare_lookback_hooks(monkeypatch):
    bars = _ohlcv(50)
    monkeypatch.setattr("screener.rs_breakout.required_history_bars", lambda: 123)
    monkeypatch.setattr(
        "screener.backtester.vivek_equity.required_history_bars",
        lambda: 456,
    )
    monkeypatch.setattr(
        "screener.backtester.vivek_equity.prepare_vivek_equity_tool_frame",
        lambda frame: frame.assign(vivek_equity_entry=1),
    )

    ctx = _prepare_ctx(bars_by_tv={"AAA": bars})

    assert _rs_breakout_lookback() == 123
    assert _vivek_lookback() == 456
    assert _prepare_vivek(ctx)["AAA"]["vivek_equity_entry"].iloc[0] == 1


def _prepare_ctx(
    *,
    market: str = "us",
    benchmark: str = "SPY",
    bars_by_tv: dict[str, pd.DataFrame] | None = None,
    price_panel: dict[str, pd.DataFrame] | None = None,
) -> PrepareCtx:
    bars = _ohlcv(50)
    return PrepareCtx(
        market=market,
        benchmark=benchmark,
        bars_by_tv=bars_by_tv or {"AAA": bars},
        price_panel=price_panel or {benchmark: bars},
        tv_symbols=["AAA"],
        start=pd.Timestamp("2024-01-01").date(),
        end=pd.Timestamp("2024-03-01").date(),
        fetcher=lambda *_args, **_kwargs: {},
        warnings=[],
    )
