"""Tests for the India-relevant dual- and time-series-momentum plugins.

The synthetic price paths cover hand-checked signal math, exact warmup
boundaries, causality, benchmark gating and failure behavior, empty/None input
frames, registry metadata, raw ``rank_score`` wiring, and rolling top-N
selection for both price-only strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.strategies.plugins.dual_momentum import _prepare_dual_momentum
from screener.strategies.plugins.time_series_momentum import (
    _prepare_time_series_momentum,
    annualized_volatility,
    trailing_12m_return,
)
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 340
_INDEX = pd.bdate_range("2021-06-01", periods=_N)


def _ctx(
    bars_by_tv: dict[str, pd.DataFrame],
    *,
    benchmark: str = "^NSEI",
    price_panel: dict[str, pd.DataFrame] | None = None,
    fetcher: StubPriceFetcher | None = None,
    warnings: list[str] | None = None,
) -> PrepareCtx:
    return PrepareCtx(
        market="india",
        benchmark=benchmark,
        bars_by_tv=bars_by_tv,
        price_panel=price_panel if price_panel is not None else {},
        tv_symbols=list(bars_by_tv),
        start=_INDEX[0].date(),
        end=_INDEX[-1].date(),
        fetcher=fetcher if fetcher is not None else StubPriceFetcher({}),
        warnings=warnings if warnings is not None else [],
    )


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    openp = close.shift(1).fillna(close.iloc[0])
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


def _geom(daily: float, start: float = 100.0) -> pd.Series:
    return pd.Series(start * (1.0 + daily) ** np.arange(_N), index=_INDEX)


def _backtest_config(
    strategy: str, entry: str, tickers: tuple[str, ...]
) -> BacktestConfig:
    return BacktestConfig(
        market="india",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name=strategy,
        entry_expr=entry,
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="^NSEI",
        tickers=tickers,
        min_price=None,
        min_avg_dollar_volume=None,
    )


# --------------------------------------------------------------------------- #
# Strategy 1: dual_momentum
# --------------------------------------------------------------------------- #


def test_dual_momentum_registered_with_documented_contract() -> None:
    discover_plugins()
    spec = registry.get_optional("dual_momentum")
    assert spec is not None
    assert spec.entry == "dual_ok > 0"
    assert spec.exit is None
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_dual_momentum_score_math_warmup_and_rank_score() -> None:
    name_close = _geom(0.001)
    bench_close = _geom(0.0005, start=20_000.0)
    out = _prepare_dual_momentum(
        _ctx(
            {"NSE:NAME": _ohlcv(name_close)},
            price_panel={"^NSEI": _ohlcv(bench_close)},
        )
    )["NSE:NAME"]
    t = 300
    expected = (1.001**231) - 1.0
    assert np.isclose(out["mom_12_1"].iloc[t], expected)
    pd.testing.assert_series_equal(
        out["rank_score"], out["mom_12_1"], check_names=False
    )
    assert out["mom_12_1"].iloc[:252].isna().all()
    assert out["mom_12_1"].iloc[252:].notna().all()
    assert out["dual_ok"].iloc[:252].eq(0.0).all()
    assert out["dual_ok"].iloc[252:].eq(1.0).all()


@pytest.mark.parametrize("benchmark_daily", [-0.0005, 0.0])
def test_dual_momentum_requires_positive_name_and_benchmark(
    benchmark_daily: float,
) -> None:
    bars = {
        "NSE:UP": _ohlcv(_geom(0.001)),
        "NSE:DOWN": _ohlcv(_geom(-0.001)),
        "EMPTY": pd.DataFrame(),
    }
    ctx = _ctx(
        bars,
        price_panel={"^NSEI": _ohlcv(_geom(benchmark_daily, start=20_000.0))},
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    out = _prepare_dual_momentum(ctx)
    t = 300
    # Even the positive name is risk-off when benchmark momentum is <= 0.
    assert out["NSE:UP"]["mom_12_1"].iloc[t] > 0
    assert out["NSE:UP"]["dual_ok"].iloc[t] == 0.0
    assert out["NSE:DOWN"]["dual_ok"].iloc[t] == 0.0
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_dual_momentum_positive_benchmark_still_gates_negative_name() -> None:
    ctx = _ctx(
        {
            "NSE:UP": _ohlcv(_geom(0.001)),
            "NSE:DOWN": _ohlcv(_geom(-0.001)),
        },
        price_panel={"^NSEI": _ohlcv(_geom(0.0004, start=20_000.0))},
    )
    out = _prepare_dual_momentum(ctx)
    t = 300
    assert out["NSE:UP"]["dual_ok"].iloc[t] == 1.0
    assert out["NSE:DOWN"]["dual_ok"].iloc[t] == 0.0


def test_dual_momentum_is_causal_for_name_and_benchmark() -> None:
    name_close = _geom(0.001)
    bench_close = _geom(0.0004, start=20_000.0)
    t = 280
    base = _prepare_dual_momentum(
        _ctx(
            {"NSE:A": _ohlcv(name_close)},
            price_panel={"^NSEI": _ohlcv(bench_close)},
        )
    )["NSE:A"]

    changed_name = name_close.copy()
    changed_bench = bench_close.copy()
    changed_name.iloc[t + 1 :] *= 8.0
    changed_bench.iloc[t + 1 :] *= 0.1
    changed = _prepare_dual_momentum(
        _ctx(
            {"NSE:A": _ohlcv(changed_name)},
            price_panel={"^NSEI": _ohlcv(changed_bench)},
        )
    )["NSE:A"]
    assert np.isclose(base["mom_12_1"].iloc[t], changed["mom_12_1"].iloc[t])
    assert base["dual_ok"].iloc[t] == changed["dual_ok"].iloc[t]


def test_dual_momentum_missing_benchmark_warns_and_sets_nan() -> None:
    ctx = _ctx(
        {"NSE:A": _ohlcv(_geom(0.001)), "EMPTY": pd.DataFrame()},
        fetcher=StubPriceFetcher({}),
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    out = _prepare_dual_momentum(ctx)
    assert any(
        "dual_momentum" in warning and "benchmark" in warning
        for warning in ctx.warnings
    )
    assert out["NSE:A"][["mom_12_1", "dual_ok", "rank_score"]].isna().all().all()
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_dual_momentum_end_to_end_ranks_relative_winners() -> None:
    data = {
        "STRONG.NS": _ohlcv(_geom(0.0015)),
        "WEAK.NS": _ohlcv(_geom(0.0004)),
        "DOWN.NS": _ohlcv(_geom(-0.0010)),
        "^NSEI": _ohlcv(_geom(0.0005, start=20_000.0)),
    }
    cfg = _backtest_config(
        "dual_momentum", "dual_ok > 0", ("NSE:STRONG", "NSE:WEAK", "NSE:DOWN")
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[280].date(),
        end_date=_INDEX[-1].date(),
    )
    assert {trade.ticker for trade in result.trades} == {"NSE:STRONG"}


# --------------------------------------------------------------------------- #
# Strategy 2: time_series_momentum
# --------------------------------------------------------------------------- #


def test_time_series_momentum_registered_with_documented_contract() -> None:
    discover_plugins()
    spec = registry.get_optional("time_series_momentum")
    assert spec is not None
    assert spec.entry == "ts_ret > 0"
    assert spec.exit is None
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_time_series_return_hand_checked_and_warmup() -> None:
    ts_ret = trailing_12m_return(_geom(0.001))
    t = 300
    assert np.isclose(ts_ret.iloc[t], (1.001**252) - 1.0)
    assert ts_ret.iloc[:252].isna().all()
    assert ts_ret.iloc[252:].notna().all()


def test_time_series_volatility_hand_checked_and_warmup() -> None:
    rng = np.random.default_rng(14)
    log_rets = rng.normal(0.0004, 0.012, _N)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=_INDEX)
    vol = annualized_volatility(close)
    t = 300
    expected = np.std(np.diff(np.log(close.to_numpy()))[t - 252 : t], ddof=1)
    expected *= np.sqrt(252)
    assert np.isclose(vol.iloc[t], expected)
    assert vol.iloc[:252].isna().all()
    assert vol.iloc[252:].notna().all()


def test_time_series_signals_are_causal() -> None:
    rng = np.random.default_rng(8)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, _N))), index=_INDEX
    )
    t = 290
    base_ret = trailing_12m_return(close)
    base_vol = annualized_volatility(close)
    changed = close.copy()
    changed.iloc[t + 1 :] *= 6.0
    assert np.isclose(base_ret.iloc[t], trailing_12m_return(changed).iloc[t])
    assert np.isclose(base_vol.iloc[t], annualized_volatility(changed).iloc[t])


def test_time_series_prepare_rank_score_guard_empty_and_none() -> None:
    rng = np.random.default_rng(19)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, _N))), index=_INDEX
    )
    ctx = _ctx(
        {"NSE:A": _ohlcv(close), "FLAT": _ohlcv(_geom(0.0)), "EMPTY": pd.DataFrame()}
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    out = _prepare_time_series_momentum(ctx)
    assert {"ts_ret", "vol_ann", "rank_score"} <= set(out["NSE:A"].columns)
    expected = out["NSE:A"]["ts_ret"] / out["NSE:A"]["vol_ann"]
    pd.testing.assert_series_equal(
        out["NSE:A"]["rank_score"], expected, check_names=False
    )
    # A flat price has zero volatility once warm; the score guard keeps it NaN.
    assert out["FLAT"]["vol_ann"].iloc[252:].eq(0.0).all()
    assert out["FLAT"]["rank_score"].isna().all()
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_time_series_momentum_end_to_end_uses_risk_scaled_rank_score() -> None:
    def trend(drift: float, sigma: float, seed: int) -> pd.Series:
        rng = np.random.default_rng(seed)
        log_rets = rng.normal(drift, sigma, _N)
        return pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=_INDEX)

    data = {
        "BEST.NS": _ohlcv(trend(0.0018, 0.003, 1)),
        "MID.NS": _ohlcv(trend(0.0006, 0.012, 2)),
        "DOWN.NS": _ohlcv(trend(-0.0015, 0.003, 3)),
        "^NSEI": _ohlcv(trend(0.0004, 0.008, 4)),
    }
    cfg = _backtest_config(
        "time_series_momentum",
        "ts_ret > 0",
        ("NSE:BEST", "NSE:MID", "NSE:DOWN"),
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[280].date(),
        end_date=_INDEX[-1].date(),
    )
    assert {trade.ticker for trade in result.trades} == {"NSE:BEST"}
