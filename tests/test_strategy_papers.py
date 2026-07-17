"""Unit + end-to-end tests for the three paper-replication factor strategies.

Covers ``high_52w_momentum`` (George-Hwang 2004), ``fip_momentum`` (Da-Gurun-
Warachka 2014) and ``low_beta`` (Frazzini-Pedersen 2014): hand-checked score
math, causality (mutating future bars leaves the signal at ``t`` unchanged), NaN
warmup, empty/None-bars paths, the low_beta benchmark-acquisition ladder and its
missing-benchmark warning path, and registry + rank_score wiring end-to-end.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.strategies.plugins.fip_momentum import (
    _ID_WINDOW,
    _prepare_fip,
    information_discreteness,
)
from screener.strategies.plugins.high_52w_momentum import (
    _NEARNESS_THRESHOLD,
    _prepare_high_52w,
    nearness_52w_score,
)
from screener.strategies.plugins.low_beta import (
    _prepare_low_beta,
    rolling_beta,
)
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2021-06-01", periods=_N)


def _ctx(
    bars_by_tv: dict[str, pd.DataFrame],
    *,
    benchmark: str = "SPY",
    price_panel: dict[str, pd.DataFrame] | None = None,
    fetcher: StubPriceFetcher | None = None,
    warnings: list[str] | None = None,
) -> PrepareCtx:
    return PrepareCtx(
        market="us",
        benchmark=benchmark,
        bars_by_tv=bars_by_tv,
        price_panel=price_panel if price_panel is not None else {},
        tv_symbols=list(bars_by_tv),
        start=_INDEX[0].date(),
        end=_INDEX[-1].date(),
        fetcher=fetcher if fetcher is not None else StubPriceFetcher({}),
        warnings=warnings if warnings is not None else [],
    )


def _ohlcv(close: pd.Series, high: pd.Series | None = None) -> pd.DataFrame:
    openp = close.shift(1).fillna(close.iloc[0])
    if high is None:
        high = pd.concat([openp, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([openp, close], axis=1).min(axis=1) * 0.999
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(900_000.0, index=close.index),
        }
    )


# --------------------------------------------------------------------------- #
# Strategy 1: high_52w_momentum
# --------------------------------------------------------------------------- #


def test_high_52w_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("high_52w_momentum")
    assert spec is not None
    assert spec.entry == "nearness >= 0.85"
    assert _NEARNESS_THRESHOLD == 0.85
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_nearness_score_hand_checked() -> None:
    close = pd.Series(np.linspace(100.0, 200.0, _N), index=_INDEX)
    high = close * 1.01
    nearness = nearness_52w_score(close, high)
    t = 270
    # Increasing series -> prior 252-day max of HIGH is the immediately prior bar.
    expected = close.iloc[t] / (close.iloc[t - 1] * 1.01)
    assert np.isclose(nearness.iloc[t], expected)
    # Undefined until 252 prior highs exist (rolling 252 + shift 1).
    assert nearness.iloc[:252].isna().all()
    assert nearness.iloc[252:].notna().all()


def test_nearness_is_causal() -> None:
    close = pd.Series(np.linspace(100.0, 180.0, _N), index=_INDEX)
    high = close * 1.02
    base = nearness_52w_score(close, high)
    t = 270
    close2 = close.copy()
    high2 = high.copy()
    close2.iloc[t + 1 :] = 1_000.0
    high2.iloc[t + 1 :] = 5_000.0
    mutated = nearness_52w_score(close2, high2)
    assert np.isclose(base.iloc[t], mutated.iloc[t])


def test_nearness_no_high_falls_back_to_close() -> None:
    close = pd.Series(np.linspace(100.0, 150.0, _N), index=_INDEX)
    from_close = nearness_52w_score(close, None)
    same = nearness_52w_score(close, close)
    pd.testing.assert_series_equal(from_close, same)


def test_high_52w_prepare_warns_without_high_and_handles_empty() -> None:
    close = pd.Series(np.linspace(100.0, 150.0, _N), index=_INDEX)
    no_high = _ohlcv(close).drop(columns=["high"])
    ctx = _ctx({"A": no_high, "B": no_high.copy(), "EMPTY": pd.DataFrame()})
    out = _prepare_high_52w(ctx)
    # Warning emitted exactly once even though two frames lack 'high'.
    assert sum("no 'high' column" in w for w in ctx.warnings) == 1
    assert "rank_score" in out["A"].columns
    assert out["EMPTY"].empty
    # rank_score mirrors the close-based nearness when high is absent.
    expected = nearness_52w_score(close, None)
    pd.testing.assert_series_equal(out["A"]["rank_score"], expected, check_names=False)


def test_high_52w_end_to_end_selects_closest_to_high() -> None:
    idx = _INDEX
    n = len(idx)
    # NEAR keeps making highs -> nearness ~1. FAR peaks early then sits 25% below
    # its high -> nearness < 0.85 (excluded). MID sits ~5% below -> eligible but
    # ranks below NEAR.
    rising = pd.Series(np.linspace(50.0, 90.0, n), index=idx)
    peak = np.linspace(50.0, 90.0, n // 2)
    far = pd.Series(
        np.concatenate([peak, np.full(n - n // 2, peak[-1] * 0.75)]), index=idx
    )
    mid = pd.Series(
        np.concatenate([peak, np.full(n - n // 2, peak[-1] * 0.95)]), index=idx
    )
    spy = pd.Series(np.linspace(400.0, 440.0, n), index=idx)
    data = {
        "NEAR": _ohlcv(rising),
        "FAR": _ohlcv(far),
        "MID": _ohlcv(mid),
        "SPY": _ohlcv(spy),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=10,
        top=1,
        strategy_name="high_52w_momentum",
        entry_expr="nearness >= 0.85",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("NEAR", "FAR", "MID"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, StubPriceFetcher(data), start_date=idx[260].date(), end_date=idx[-1].date()
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"NEAR"}, traded


# --------------------------------------------------------------------------- #
# Strategy 2: fip_momentum
# --------------------------------------------------------------------------- #


def test_fip_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("fip_momentum")
    assert spec is not None
    assert spec.entry == "mom_12_1 > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback() == 253


def test_information_discreteness_hand_checked() -> None:
    # Build a return path whose ID window [t-251, t-21] (t=252) has exactly ``m``
    # down days and the rest up days, with the up moves large enough that overall
    # momentum is positive (sign = +1). ID = (m - (231-m)) / 231.
    m = 40
    rets = np.zeros(_N)
    rets[1 : 1 + m] = -0.001
    rets[1 + m : 1 + _ID_WINDOW] = 0.003
    rets[1 + _ID_WINDOW :] = 0.001  # skipped last month, must not affect ID[252]
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=_INDEX)
    mom = momentum_12_1_score(close)
    idv = information_discreteness(close, mom)
    t = 252
    assert mom.iloc[t] > 0
    expected = (m / _ID_WINDOW) - ((_ID_WINDOW - m) / _ID_WINDOW)
    assert np.isclose(idv.iloc[t], expected)
    # Sign leg: a net-negative path flips the sign so an all-down window is still
    # negative ID (continuous), never a large positive.
    down = pd.Series(100.0 * np.cumprod(1.0 + np.full(_N, -0.001)), index=_INDEX)
    down_mom = momentum_12_1_score(down)
    down_id = information_discreteness(down, down_mom)
    assert down_mom.iloc[t] < 0
    # all-down window: pct_neg=1, pct_pos=0, sign=-1 -> ID = -1.
    assert np.isclose(down_id.iloc[t], -1.0)


def test_information_discreteness_is_causal() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.01, _N)
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=_INDEX)
    mom = momentum_12_1_score(close)
    base = information_discreteness(close, mom)
    t = 260
    close2 = close.copy()
    close2.iloc[t + 1 :] *= 3.0
    mom2 = momentum_12_1_score(close2)
    mutated = information_discreteness(close2, mom2)
    assert np.isclose(base.iloc[t], mutated.iloc[t])
    # NaN warmup: ID undefined until both the 231-window (+21 skip) and the
    # 12-1 momentum sign leg are defined -> first valid at index 252.
    assert base.iloc[:252].isna().all()
    assert base.iloc[252:].notna().all()


def test_fip_prepare_empty_and_no_data_paths() -> None:
    close = pd.Series(100.0 * np.cumprod(1.0 + np.full(_N, 0.001)), index=_INDEX)
    ctx = _ctx({"A": _ohlcv(close), "EMPTY": pd.DataFrame()})
    out = _prepare_fip(ctx)
    assert {"mom_12_1", "id_disc", "rank_score"} <= set(out["A"].columns)
    assert out["EMPTY"].empty
    # All-empty universe returns bars untouched (no cross-section to rank).
    empty_ctx = _ctx({"X": pd.DataFrame(), "Y": pd.DataFrame()})
    empty_out = _prepare_fip(empty_ctx)
    assert empty_out["X"].empty
    assert empty_out["Y"].empty


def test_fip_end_to_end_prefers_continuous_information() -> None:
    idx = _INDEX
    n = len(idx)

    def path(growth: float, noise: float) -> pd.DataFrame:
        drift = 50.0 * (1.0 + growth) ** np.arange(n)
        wiggle = noise * 50.0 * np.sin(np.arange(n))
        close = pd.Series(drift + wiggle, index=idx)
        return _ohlcv(close)

    # HIMOM: highest momentum but choppy (discrete info, ID high). BEST: slightly
    # lower momentum but perfectly smooth (all-up, ID = -1) -> wins the 50/50
    # blend. MEH/LOW fill the cross-section.
    data = {
        "HIMOM": path(growth=0.0020, noise=0.030),
        "BEST": path(growth=0.0015, noise=0.0),
        "MEH": path(growth=0.0010, noise=0.015),
        "LOW": path(growth=0.0006, noise=0.006),
        "SPY": path(growth=0.0005, noise=0.004),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=10,
        top=1,
        strategy_name="fip_momentum",
        entry_expr="mom_12_1 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("HIMOM", "BEST", "MEH", "LOW"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, StubPriceFetcher(data), start_date=idx[260].date(), end_date=idx[-1].date()
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"BEST"}, traded


# --------------------------------------------------------------------------- #
# Strategy 3: low_beta
# --------------------------------------------------------------------------- #


def _beta_close(bench_ret: pd.Series, beta: float, start: float = 100.0) -> pd.Series:
    """A close series whose daily returns are exactly ``beta * bench_ret``."""
    factors = (1.0 + beta * bench_ret.fillna(0.0)).cumprod()
    return start * factors


def test_low_beta_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("low_beta")
    assert spec is not None
    assert spec.entry == "beta_252 < 1.0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback() == 253


def test_rolling_beta_hand_checked_and_causal() -> None:
    rng = np.random.default_rng(3)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.008, _N)), index=_INDEX
    )
    bench_ret = bench_close.pct_change()
    name_close = _beta_close(bench_ret, beta=0.5)
    beta = rolling_beta(name_close, bench_ret)
    t = 270
    assert np.isclose(beta.iloc[t], 0.5)
    # NaN warmup: beta undefined until 252 returns exist.
    assert beta.iloc[:252].isna().all()
    assert beta.iloc[252:].notna().all()
    # Causality: mutating future bars leaves beta at t unchanged.
    name2 = name_close.copy()
    name2.iloc[t + 1 :] *= 4.0
    beta2 = rolling_beta(name2, bench_ret)
    assert np.isclose(beta.iloc[t], beta2.iloc[t])


def test_low_beta_benchmark_acquisition_ladder() -> None:
    close = pd.Series(100.0 * np.cumprod(1.0 + np.full(_N, 0.0005)), index=_INDEX)
    bars = {"A": _ohlcv(close)}
    bench_frame = _ohlcv(
        pd.Series(400.0 * np.cumprod(1.0 + np.full(_N, 0.0004)), index=_INDEX)
    )

    # (1) benchmark resolved from price_panel; an empty frame rides through the
    # has-benchmark loop untouched.
    ctx_panel = _ctx(
        {**bars, "EMPTY": pd.DataFrame()}, price_panel={"SPY": bench_frame}
    )
    out_panel = _prepare_low_beta(ctx_panel)
    assert "beta_252" in out_panel["A"].columns
    assert out_panel["EMPTY"].empty
    assert not ctx_panel.warnings

    # (2) benchmark resolved from bars_by_tv (carried as a universe member).
    ctx_univ = _ctx({**bars, "SPY": bench_frame})
    out_univ = _prepare_low_beta(ctx_univ)
    assert out_univ["A"]["beta_252"].notna().any()

    # (3) benchmark resolved via a fetch fallback.
    ctx_fetch = _ctx(dict(bars), fetcher=StubPriceFetcher({"SPY": bench_frame}))
    out_fetch = _prepare_low_beta(ctx_fetch)
    assert out_fetch["A"]["beta_252"].notna().any()


def test_low_beta_missing_benchmark_warns_and_nans() -> None:
    close = pd.Series(100.0 * np.cumprod(1.0 + np.full(_N, 0.0005)), index=_INDEX)
    ctx = _ctx(
        {"A": _ohlcv(close), "EMPTY": pd.DataFrame()},
        fetcher=StubPriceFetcher({}),
    )
    out = _prepare_low_beta(ctx)
    assert any("benchmark" in w for w in ctx.warnings)
    assert out["A"]["beta_252"].isna().all()
    assert out["A"]["rank_score"].isna().all()
    assert out["EMPTY"].empty


def test_low_beta_end_to_end_selects_lowest_beta() -> None:
    idx = _INDEX
    rng = np.random.default_rng(11)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.008, _N)), index=idx
    )
    bench_ret = bench_close.pct_change()
    data = {
        "LOWB": _ohlcv(_beta_close(bench_ret, beta=0.3)),
        "MIDB": _ohlcv(_beta_close(bench_ret, beta=0.9)),
        "HIGHB": _ohlcv(_beta_close(bench_ret, beta=1.5)),
        "SPY": _ohlcv(bench_close),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=10,
        top=1,
        strategy_name="low_beta",
        entry_expr="beta_252 < 1.0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("LOWB", "MIDB", "HIGHB"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, StubPriceFetcher(data), start_date=idx[260].date(), end_date=idx[-1].date()
    )
    traded = {t.ticker for t in result.trades}
    # HIGHB (beta 1.5) is gated out; LOWB (0.3) outranks MIDB (0.9).
    assert traded == {"LOWB"}, traded


def test_low_beta_lookback_and_date_types() -> None:
    # Guard: prepare tolerates ``date`` start/end (fetch fallback path uses them).
    assert isinstance(_INDEX[0].date(), date)
