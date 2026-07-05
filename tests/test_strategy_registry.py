from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from screener.backtester import pine_runner
from screener.backtester.models import BacktestConfig
from screener.strategies.plugins.breakout import _breakout
from screener.strategies.plugins.ema_trend import _ema_trend
from screener.strategies.plugins.rs_breakout import (
    _prepare_rs_breakout,
    _rs_breakout,
    _rs_breakout_lookback,
)
from screener.strategies.plugins.vivek_equity_tool import (
    _prepare_vivek,
    _vivek_equity_tool,
    _vivek_lookback,
)
from screener.strategies import spec as strategy_spec_module
from screener.strategies.pine_ports import (
    strat_ma_cross_st_entry,
)
from screener.strategies.expressions import NAMED_STRATEGIES, resolve_strategy
from screener.strategies.registry import STRATEGIES, get_strategy, iter_strategies
from screener.strategies.spec import (
    DerivedView,
    PrepareCtx,
    StrategySpec,
    strategy,
)


def test_strategy_registry_preserves_pine_runner_names():
    expected = {
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

    assert set(STRATEGIES) == expected
    assert set(pine_runner.STRATEGIES) == expected
    assert dict(iter_strategies()) == STRATEGIES


def test_strategy_views_are_live_derived_over_single_registry():
    # Both public views are read-only projections of spec.registry, not stored
    # dicts, so a late registration is reflected without any rebuild.
    assert isinstance(STRATEGIES, DerivedView)
    assert isinstance(NAMED_STRATEGIES, DerivedView)

    name = "unit_live_view_probe"

    @strategy(name, entry="close > 0")
    def placeholder() -> None:
        return None

    # Visible immediately in the expression view (and its resolver)...
    assert name in NAMED_STRATEGIES
    assert resolve_strategy(name).entry == "close > 0"
    # ...but partitioned out of the callable view, since it has no callable_fn.
    assert name not in STRATEGIES


def test_strategy_registry_lookup_returns_callable():
    strategy = get_strategy("ma_cross_st_entry")

    assert strategy is STRATEGIES["ma_cross_st_entry"]
    assert callable(strategy)

    with pytest.raises(KeyError, match="Unknown strategy 'missing'"):
        get_strategy("missing")


def test_backtester_pine_runner_reexports_legacy_helpers():
    assert pine_runner._ema is not None
    assert pine_runner._rsi is not None
    assert pine_runner.load_universe is not None


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


def test_expression_only_strategy_placeholders_are_noops():
    assert _breakout() is None
    assert _ema_trend() is None
    assert _rs_breakout() is None
    assert _vivek_equity_tool() is None


def test_strategy_spec_validation_and_decorator_metadata():
    with pytest.raises(ValidationError, match="strategy name must not be empty"):
        StrategySpec(name=" ", entry="close > 0")
    with pytest.raises(ValidationError, match="either callable_fn or entry"):
        StrategySpec(name="empty")

    reg_size = len(strategy_spec_module.registry)

    @strategy("unit_test_strategy", entry=" close > 0 ")
    def placeholder() -> None:
        return None

    assert placeholder() is None
    assert len(strategy_spec_module.registry) == reg_size + 1


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
        cfg=BacktestConfig(
            market=market,
            as_of=pd.Timestamp("2024-03-01").date(),
            hold=5,
            top=1,
            entry_expr="close > 0",
            exit_expr=None,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=10_000.0,
            tickers=("AAA",),
            benchmark=benchmark,
        ),
        bars_by_tv=bars_by_tv or {"AAA": bars},
        price_panel=price_panel or {benchmark: bars},
        tv_symbols=["AAA"],
        start=pd.Timestamp("2024-01-01").date(),
        end=pd.Timestamp("2024-03-01").date(),
        fetcher=lambda *_args, **_kwargs: {},
        warnings=[],
    )
