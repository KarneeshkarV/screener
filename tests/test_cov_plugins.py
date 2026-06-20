"""Offline line-coverage tests for strategy/criteria/indicator plugins.

These exercise the registration/build functions of small plugin modules so
they reach (or approach) 100% line coverage. Everything is deterministic and
offline: strategy plugins run on synthetic ``make_bars`` frames, filter
criteria just build their TV expression lists, and pipeline criteria are
driven with their downstream runners monkeypatched out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars

from screener.strategies.registry import STRATEGIES, get_strategy
from screener.strategies.spec import StrategySpec, registry as strategy_registry
from screener.strategies.expressions import NamedStrategy
from screener.strategies.trades import Trade, _walk


# --------------------------------------------------------------------------- #
# Strategy callable plugins
# --------------------------------------------------------------------------- #


def _bars(n: int = 800, **kw) -> pd.DataFrame:
    """make_bars + a ``date`` column (strategy plugins read df['date'])."""
    df = make_bars(n=n, **kw)
    df = df.reset_index().rename(columns={"index": "date"})
    return df


CALLABLE_STRATEGIES = [
    "macd_rsi",
    "bb_breakout",
    "ma_cross",
    "ma_cross_regime",
    "ma_cross_st_exit",
    "supertrend",
    "supertrend_rsi",
    "rsi_ema",
]


@pytest.mark.parametrize("name", CALLABLE_STRATEGIES)
def test_callable_strategy_runs_and_returns_trades(name: str) -> None:
    fn = get_strategy(name)
    # An oscillating series produces both entries and exits across strategies.
    n = 800
    t = np.arange(n)
    osc = 100.0 + np.sin(t / 9.0) * 12.0 + t * 0.02
    df = _bars(n=n)
    for col in ("open", "high", "low", "close"):
        df[col] = osc
    df["high"] = osc + 1.0
    df["low"] = osc - 1.0
    trades = fn(df)
    assert isinstance(trades, list)
    assert all(isinstance(tr, Trade) for tr in trades)
    assert all(tr.entry_idx <= tr.exit_idx for tr in trades)


def test_bb_breakout_with_short_series_is_all_nan_band() -> None:
    # window 350 > n: the band is all NaN so valid mask zeroes entries/exits.
    df = _bars(n=120)
    trades = STRATEGIES["bb_breakout"](df)
    assert trades == []


# --------------------------------------------------------------------------- #
# Strategy core: registry / spec / expressions / trades
# --------------------------------------------------------------------------- #


def test_get_strategy_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("does_not_exist")


def test_strategy_spec_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StrategySpec(name="   ", callable_fn=lambda df: [])


def test_strategy_spec_requires_callable_or_entry() -> None:
    with pytest.raises(ValueError, match="callable_fn or entry"):
        StrategySpec(name="x")


def test_named_strategy_empty_entry_rejected() -> None:
    with pytest.raises(ValueError, match="entry must not be empty"):
        NamedStrategy(entry="  ", exit=None)


def test_trade_ret_zero_when_entry_price_nonpositive() -> None:
    tr = Trade(
        entry_idx=0,
        exit_idx=1,
        entry_px=0.0,
        exit_px=5.0,
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-01-02"),
    )
    assert tr.ret == 0.0


def test_walk_closes_open_position_at_end() -> None:
    close = np.array([10.0, 11.0, 12.0, 13.0])
    dates = pd.bdate_range("2024-01-01", periods=4).values
    entries = np.array([True, False, False, False])
    exits = np.array([False, False, False, False])
    trades = _walk(entries, exits, close, dates)
    assert len(trades) == 1
    assert trades[0].exit_idx == 3
    assert trades[0].exit_px == 13.0


def _prepare_ctx(market: str, price_panel: dict, bars_by_tv: dict, end="2024-03-01"):
    from datetime import date as _date

    from screener.backtester.models import BacktestConfig
    from screener.strategies.spec import PrepareCtx

    cfg = BacktestConfig(
        market=market,
        as_of=_date(2024, 3, 1),
        hold=5,
        top=10,
        entry_expr="rs_breakout_entry > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="^NSEI",
    )
    return PrepareCtx(
        cfg=cfg,
        bars_by_tv=bars_by_tv,
        price_panel=price_panel,
        tv_symbols=list(bars_by_tv),
        start=_date(2024, 1, 1),
        end=_date.fromisoformat(end),
        fetcher=object(),
        warnings=[],
    )


def test_rs_breakout_prepare_missing_benchmark_warns() -> None:
    from screener.strategies.plugins.rs_breakout import _prepare_rs_breakout

    bars = {"AAA": make_bars(n=20)}
    ctx = _prepare_ctx("us", price_panel={}, bars_by_tv=bars)
    out = _prepare_rs_breakout(ctx)
    assert out is ctx.bars_by_tv
    assert any("benchmark data unavailable" in w for w in ctx.warnings)


def test_rs_breakout_prepare_india_loads_delivery(monkeypatch) -> None:
    import screener.rs_breakout as rsb
    import screener.unusual_volume.delivery as deliv
    from screener.strategies.plugins import rs_breakout as plugin

    bench = make_bars(n=30)
    bars = {"AAA": make_bars(n=30)}
    captured = {}

    monkeypatch.setattr(rsb, "india_symbol", lambda s: s + ".NS")
    monkeypatch.setattr(
        deliv,
        "load_delivery_panel",
        lambda syms, end, history_days: (
            captured.setdefault("syms", syms) or pd.DataFrame()
        ),
    )
    monkeypatch.setattr(
        rsb,
        "prepare_backtest_frames",
        lambda b, bm, *, market, delivery_panel: {"AAA": b["AAA"]},
    )

    ctx = _prepare_ctx("india", price_panel={"^NSEI": bench}, bars_by_tv=bars)
    out = plugin._prepare_rs_breakout(ctx)
    assert "AAA" in out
    assert captured["syms"] == ["AAA.NS"]


def test_rs_breakout_prepare_india_delivery_failure_warns(monkeypatch) -> None:
    import screener.rs_breakout as rsb
    import screener.unusual_volume.delivery as deliv
    from screener.strategies.plugins import rs_breakout as plugin

    bench = make_bars(n=30)
    bars = {"AAA": make_bars(n=30)}

    monkeypatch.setattr(rsb, "india_symbol", lambda s: s)

    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(deliv, "load_delivery_panel", boom)
    monkeypatch.setattr(
        rsb,
        "prepare_backtest_frames",
        lambda b, bm, *, market, delivery_panel: dict(b),
    )

    ctx = _prepare_ctx("india", price_panel={"^NSEI": bench}, bars_by_tv=bars)
    out = plugin._prepare_rs_breakout(ctx)
    assert "AAA" in out
    assert any("delivery panel unavailable" in w for w in ctx.warnings)


def test_rs_breakout_expression_strategy_registered() -> None:
    spec = strategy_registry.get("rs_breakout")
    assert spec.entry == "rs_breakout_entry > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    # required_lookback resolves an int without network access.
    assert isinstance(spec.required_lookback(), int)


# --------------------------------------------------------------------------- #
# Indicator registry + ema plugin
# --------------------------------------------------------------------------- #


def test_get_indicator_returns_registered_fn() -> None:
    from screener.indicators.registry import get_indicator

    ema = get_indicator("ema")
    out = ema(np.array([1.0, 2.0, 3.0]), 2)
    assert len(out) == 3


def test_ema_empty_input_returns_empty() -> None:
    from screener.indicators.plugins.ema import ema

    out = ema(np.array([], dtype=float), 5)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# Criteria core: combine / is_pipeline
# --------------------------------------------------------------------------- #


def test_combine_merges_filters() -> None:
    from screener.criteria import combine

    combined = combine(lambda: [1, 2], lambda: [3])
    assert combined() == [1, 2, 3]


def test_is_pipeline_true_and_false() -> None:
    from screener.criteria import is_pipeline

    assert is_pipeline("garp") is True
    assert is_pipeline("ema") is False


# --------------------------------------------------------------------------- #
# Filter (non-pipeline) criteria: build their TV expression lists
# --------------------------------------------------------------------------- #


FILTER_CRITERIA = [
    "dividend",
    "value",
    "undervalued",
    "quality",
    "intraday_breakout",
    "momentum_value",
    "ema_breakout",
    "ema",
    "breakout",
    "cheap_quality",
    "intraday_momentum",
    "near_52_high",
]


@pytest.mark.parametrize("name", FILTER_CRITERIA)
def test_filter_criterion_builds_nonempty_list(name: str) -> None:
    from screener.criteria import CRITERIA

    filters = CRITERIA[name]()
    assert isinstance(filters, list)
    assert filters  # non-empty


# --------------------------------------------------------------------------- #
# Pipeline criteria: downstream runners monkeypatched (offline, deterministic)
# --------------------------------------------------------------------------- #


def test_garp_pipeline_runs(monkeypatch) -> None:
    import screener.criteria.plugins.garp as g

    captured = {}

    def fake_run(market, size, *, limit, workers, cache_ttl, refresh, on_universe):
        on_universe(["AAA", "BBB"])
        captured["called"] = True
        return [{"ticker": "AAA"}]

    monkeypatch.setattr(g, "run_garp_screen", fake_run)
    monkeypatch.setattr(
        g, "print_garp_results", lambda r, m: captured.setdefault("rich", True)
    )
    monkeypatch.setattr(g, "print_csv", lambda r: captured.setdefault("csv", True))

    g.garp_pipeline(
        market="us", limit=10, output_csv=False, refresh=False, cache_ttl="1d"
    )
    assert captured["rich"] is True

    g.garp_pipeline(
        market="us", limit=10, output_csv=True, refresh=False, cache_ttl="1d"
    )
    assert captured["csv"] is True


def test_garp_pipeline_no_results(monkeypatch) -> None:
    import screener.criteria.plugins.garp as g

    monkeypatch.setattr(
        g,
        "run_garp_screen",
        lambda *a, **k: None,
    )
    echoed = []
    monkeypatch.setattr(g.click, "echo", lambda *a, **k: echoed.append(a))
    g.garp_pipeline(
        market="us", limit=10, output_csv=False, refresh=False, cache_ttl="1d"
    )
    assert any("No tickers" in str(a) for a in echoed)


def test_obv_trend_pipeline_runs(monkeypatch) -> None:
    import screener.commands.live_strategies as live
    from screener.criteria import CRITERIA

    called = {}
    monkeypatch.setattr(live, "run_obv_trend_live", lambda **kw: called.update(kw))
    CRITERIA["obv-trend"](market="india", limit=5)
    assert called["market"] == "india"
    assert called["limit"] == 5


def test_vol_breakout_pipeline_runs(monkeypatch) -> None:
    import screener.commands.live_strategies as live
    from screener.criteria import CRITERIA

    called = {}
    monkeypatch.setattr(live, "run_vol_breakout_live", lambda **kw: called.update(kw))
    CRITERIA["vol-breakout"](market="us", limit=7)
    assert called["limit"] == 7


def test_unusual_volume_pipeline_runs(monkeypatch) -> None:
    import screener.unusual_volume.cli as uv
    from screener.criteria import CRITERIA

    called = {}
    monkeypatch.setattr(uv, "run_unusual_volume", lambda **kw: called.update(kw))
    CRITERIA["unusual-volume"](market="us", limit=3, refresh=True)
    assert called["refresh"] is True


def test_promoter_buys_pipeline_runs(monkeypatch) -> None:
    import screener.commands.insiders as insiders
    from screener.criteria import CRITERIA

    called = {}
    monkeypatch.setattr(insiders, "run_promoter_buys", lambda **kw: called.update(kw))
    CRITERIA["promoter-buys"](
        market="india",
        limit=4,
        output_csv=False,
        refresh=False,
        cache_ttl="1d",
    )
    assert called["market"] == "india"
    assert called["limit"] == 4


def test_rs_breakout_pipeline_runs(monkeypatch) -> None:
    import screener.commands.rs_breakout as rs
    import screener.criteria.plugins.rs_breakout as plugin
    from screener.criteria import CRITERIA

    sentinel = object()
    monkeypatch.setattr(rs, "run_rs_breakout_screen", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        rs, "write_default_outputs", lambda *a, **k: ("out.json", "out.md")
    )
    rendered = {}
    monkeypatch.setattr(
        plugin, "render_result", lambda result, console, **kw: rendered.update(kw)
    )

    CRITERIA["rs-breakout"](market="india", limit=9, refresh=False, cache_ttl="15m")
    assert rendered["limit"] == 9
    assert rendered["market"] == "india"
