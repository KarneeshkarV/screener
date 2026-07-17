"""Unit + end-to-end tests for the two India-relevant price-only factor plugins.

Covers ``risk_adjusted_momentum`` (NSE Nifty200 Momentum 30 methodology) and
``low_ivol`` (Ang-Hodrick-Xing-Zhang 2006 idiosyncratic-volatility anomaly):
hand-checked score math on synthetic series, causality (mutating future bars
leaves the signal at ``t`` unchanged), NaN warmup boundaries, empty/None-bars
paths, the low_ivol missing-benchmark warning path, cross-sectional Z-score
correctness, and registry + rank_score wiring end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.strategies.plugins.low_ivol import (
    _WINDOW,
    _prepare_low_ivol,
    idiosyncratic_volatility,
)
from screener.strategies.plugins.risk_adjusted_momentum import (
    _prepare_risk_adjusted_momentum,
    annualized_volatility,
    cross_sectional_zscore,
    six_month_return,
    twelve_month_return,
)
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 340
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
    """A close series with a constant daily gross return ``1 + daily``."""
    return pd.Series(start * (1.0 + daily) ** np.arange(_N), index=_INDEX)


# --------------------------------------------------------------------------- #
# Strategy 1: risk_adjusted_momentum
# --------------------------------------------------------------------------- #


def test_risk_adjusted_momentum_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("risk_adjusted_momentum")
    assert spec is not None
    assert spec.entry == "mom_score > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_returns_hand_checked_and_warmup() -> None:
    close = _geom(0.001)
    r6 = six_month_return(close)
    r12 = twelve_month_return(close)
    t = 300
    assert np.isclose(r6.iloc[t], (1.001**126) - 1.0)
    assert np.isclose(r12.iloc[t], (1.001**252) - 1.0)
    # Warmup: 6m needs 126 prior closes, 12m needs 252.
    assert r6.iloc[:126].isna().all()
    assert r6.iloc[126:].notna().all()
    assert r12.iloc[:252].isna().all()
    assert r12.iloc[252:].notna().all()


def test_annualized_volatility_hand_checked_and_warmup() -> None:
    rng = np.random.default_rng(5)
    log_rets = rng.normal(0.0, 0.01, _N)
    log_rets[0] = 0.0  # first return is dropped by diff anyway
    close = pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=_INDEX)
    vol = annualized_volatility(close)
    t = 300
    # Independent computation: std (ddof=1) of the trailing 252 log returns.
    window = np.diff(np.log(close.to_numpy()))[t - 252 : t]
    assert window.size == 252
    expected = np.std(window, ddof=1) * np.sqrt(252)
    assert np.isclose(vol.iloc[t], expected)
    # Warmup: log-return diff drops bar 0, so the 252-return window is first full
    # at index 252.
    assert vol.iloc[:252].isna().all()
    assert vol.iloc[252:].notna().all()


def test_risk_adjusted_signals_are_causal() -> None:
    rng = np.random.default_rng(9)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, _N)), index=_INDEX
    )
    base_vol = annualized_volatility(close)
    base_r6 = six_month_return(close)
    t = 300
    close2 = close.copy()
    close2.iloc[t + 1 :] *= 5.0
    assert np.isclose(base_vol.iloc[t], annualized_volatility(close2).iloc[t])
    assert np.isclose(base_r6.iloc[t], six_month_return(close2).iloc[t])


def test_cross_sectional_zscore_mean0_std1_and_zero_std_guard() -> None:
    frame = pd.DataFrame(
        {
            "A": [1.0, 5.0, np.nan],
            "B": [2.0, 5.0, 4.0],
            "C": [6.0, 5.0, 10.0],
        }
    )
    z = cross_sectional_zscore(frame)
    # Row 0 fully defined -> cross-sectional mean ~0, sample std (ddof=1) ~1.
    assert np.isclose(z.iloc[0].mean(), 0.0)
    assert np.isclose(z.iloc[0].std(ddof=1), 1.0)
    # Row 1 all equal -> std == 0 -> guarded to NaN.
    assert z.iloc[1].isna().all()
    # Row 2 has a NaN name; the two defined names still get finite z-scores.
    assert z.iloc[2]["B"] == -z.iloc[2]["C"]
    assert np.isnan(z.iloc[2]["A"])


def test_risk_adjusted_prepare_columns_and_zscore_mean_zero() -> None:
    # Three fully-defined names with distinct risk-adjusted momentum.
    data = {
        "HI": _ohlcv(_geom(0.0015)),
        "MID": _ohlcv(_geom(0.0008)),
        "LO": _ohlcv(_geom(0.0002)),
    }
    ctx = _ctx(data)
    out = _prepare_risk_adjusted_momentum(ctx)
    for tv in data:
        cols = out[tv].columns
        assert {"ret_6m", "ret_12m", "vol_ann", "mom_score", "rank_score"} <= set(cols)
        pd.testing.assert_series_equal(
            out[tv]["rank_score"], out[tv]["mom_score"], check_names=False
        )
    t = 300
    # z6 and z12 each have cross-sectional mean 0, so 0.5*z6+0.5*z12 does too:
    # the three names' mom_score sums to ~0 on a fully-defined date.
    scores = np.array([out[tv]["mom_score"].iloc[t] for tv in data])
    assert np.isclose(scores.sum(), 0.0)
    # Higher risk-adjusted momentum -> higher score.
    assert scores[0] > scores[1] > scores[2]


def test_risk_adjusted_prepare_empty_none_and_all_empty() -> None:
    close = _geom(0.001)
    bars = {"A": _ohlcv(close), "EMPTY": pd.DataFrame()}
    ctx = _ctx(bars)
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    out = _prepare_risk_adjusted_momentum(ctx)
    assert {"ret_6m", "ret_12m", "vol_ann", "mom_score"} <= set(out["A"].columns)
    assert out["EMPTY"].empty
    assert out["NONE"] is None
    # All-empty universe: nothing to cross-sectionally rank -> bars untouched.
    empty_ctx = _ctx({"X": pd.DataFrame(), "Y": pd.DataFrame()})
    empty_out = _prepare_risk_adjusted_momentum(empty_ctx)
    assert empty_out["X"].empty
    assert empty_out["Y"].empty


def test_risk_adjusted_end_to_end_selects_highest_radj() -> None:
    idx = _INDEX
    # WINNER: strong smooth trend (high return, low vol) -> top risk-adj momentum.
    # LOSER: declining -> negative momentum, gated out. MID fills the middle.
    data = {
        "WINNER": _ohlcv(_geom(0.0018)),
        "MID": _ohlcv(_geom(0.0005)),
        "LOSER": _ohlcv(_geom(-0.0010)),
        "SPY": _ohlcv(_geom(0.0004)),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=10,
        top=1,
        strategy_name="risk_adjusted_momentum",
        entry_expr="mom_score > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("WINNER", "MID", "LOSER"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, StubPriceFetcher(data), start_date=idx[300].date(), end_date=idx[-1].date()
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"WINNER"}, traded


# --------------------------------------------------------------------------- #
# Strategy 2: low_ivol
# --------------------------------------------------------------------------- #


def _iv_close(
    bench_ret: pd.Series,
    beta: float,
    idio_scale: float,
    seed: int,
    start: float = 100.0,
) -> pd.Series:
    """Close whose daily returns are ``beta*bench_ret + idio_scale*eps``."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, len(bench_ret))
    rets = beta * bench_ret.fillna(0.0).to_numpy() + idio_scale * eps
    return pd.Series(start * np.cumprod(1.0 + rets), index=bench_ret.index)


def test_low_ivol_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("low_ivol")
    assert spec is not None
    assert spec.entry == "ivol > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_idiosyncratic_volatility_hand_checked_and_warmup() -> None:
    rng = np.random.default_rng(3)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.008, _N)), index=_INDEX
    )
    bench_ret = bench_close.pct_change()
    name_close = _iv_close(bench_ret, beta=0.8, idio_scale=0.01, seed=21)
    ivol = idiosyncratic_volatility(name_close, bench_ret)
    t = 300
    # Independent OLS over the trailing 252 returns: residual std (ddof=1) * √252.
    r_i = name_close.pct_change().to_numpy()[t - 251 : t + 1]
    r_m = bench_ret.to_numpy()[t - 251 : t + 1]
    slope, intercept = np.polyfit(r_m, r_i, 1)
    resid = r_i - (slope * r_m + intercept)
    expected = np.sqrt(np.sum(resid**2) / (r_i.size - 1)) * np.sqrt(252)
    assert np.isclose(ivol.iloc[t], expected)
    # Warmup: first full 252-return window at index 252.
    assert ivol.iloc[:252].isna().all()
    assert ivol.iloc[252:].notna().all()


def test_idiosyncratic_volatility_zero_for_pure_market_and_causal() -> None:
    rng = np.random.default_rng(4)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.007, _N)), index=_INDEX
    )
    bench_ret = bench_close.pct_change()
    # Pure market exposure (no idiosyncratic term) -> residual vol ~ 0.
    pure = _iv_close(bench_ret, beta=1.2, idio_scale=0.0, seed=0)
    ivol_pure = idiosyncratic_volatility(pure, bench_ret)
    t = 300
    assert np.isclose(ivol_pure.iloc[t], 0.0, atol=1e-9)
    # Causality: mutating future bars leaves ivol at t unchanged.
    noisy = _iv_close(bench_ret, beta=0.5, idio_scale=0.02, seed=7)
    base = idiosyncratic_volatility(noisy, bench_ret)
    noisy2 = noisy.copy()
    noisy2.iloc[t + 1 :] *= 6.0
    assert np.isclose(base.iloc[t], idiosyncratic_volatility(noisy2, bench_ret).iloc[t])


def test_low_ivol_prepare_with_benchmark_columns_and_empty_none() -> None:
    rng = np.random.default_rng(8)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.008, _N)), index=_INDEX
    )
    bench_ret = bench_close.pct_change()
    name = _iv_close(bench_ret, beta=0.9, idio_scale=0.01, seed=1)
    ctx = _ctx(
        {"A": _ohlcv(name), "EMPTY": pd.DataFrame()},
        price_panel={"SPY": _ohlcv(bench_close)},
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]
    out = _prepare_low_ivol(ctx)
    assert not ctx.warnings
    assert "ivol" in out["A"].columns
    assert out["A"]["ivol"].notna().any()
    pd.testing.assert_series_equal(
        out["A"]["rank_score"], -out["A"]["ivol"], check_names=False
    )
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_low_ivol_missing_benchmark_warns_and_nans() -> None:
    rng = np.random.default_rng(2)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, _N)), index=_INDEX
    )
    ctx = _ctx(
        {"A": _ohlcv(close), "EMPTY": pd.DataFrame()},
        fetcher=StubPriceFetcher({}),
    )
    out = _prepare_low_ivol(ctx)
    assert any("benchmark" in w for w in ctx.warnings)
    assert out["A"]["ivol"].isna().all()
    assert out["A"]["rank_score"].isna().all()
    assert out["EMPTY"].empty


def test_low_ivol_end_to_end_selects_lowest_ivol() -> None:
    idx = _INDEX
    rng = np.random.default_rng(11)
    bench_close = pd.Series(
        400.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.008, _N)), index=idx
    )
    bench_ret = bench_close.pct_change()
    data = {
        "LOWIV": _ohlcv(_iv_close(bench_ret, beta=0.9, idio_scale=0.004, seed=31)),
        "MIDIV": _ohlcv(_iv_close(bench_ret, beta=0.9, idio_scale=0.015, seed=32)),
        "HIGHIV": _ohlcv(_iv_close(bench_ret, beta=0.9, idio_scale=0.030, seed=33)),
        "SPY": _ohlcv(bench_close),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=idx[-1].date(),
        hold=10,
        top=1,
        strategy_name="low_ivol",
        entry_expr="ivol > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("LOWIV", "MIDIV", "HIGHIV"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, StubPriceFetcher(data), start_date=idx[300].date(), end_date=idx[-1].date()
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"LOWIV"}, traded


def test_low_ivol_lookback_matches_window() -> None:
    assert _WINDOW + 1 == 253
