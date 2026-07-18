"""Tests for the two additional US price-only factor strategies.

Covers ``residual_momentum`` (Blitz-Huij-Martens 2011) and
``short_term_reversal`` (Jegadeesh 1990 / Lehmann 1990): independently checked
score math, exact NaN warmup boundaries, causality, empty/None frame handling,
the missing-benchmark path, registry metadata, and end-to-end factor selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.strategies.plugins.residual_momentum import (
    _BETA_WINDOW,
    _FORMATION_WINDOW,
    _SKIP,
    _prepare_residual_momentum,
    residual_momentum_score,
)
from screener.strategies.plugins.short_term_reversal import (
    _prepare_short_term_reversal,
    short_term_reversal_score,
)
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 620
_INDEX = pd.bdate_range("2021-01-04", periods=_N)


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


def _close_from_returns(returns: pd.Series, start: float = 100.0) -> pd.Series:
    factors = 1.0 + returns.fillna(0.0).to_numpy()
    return pd.Series(start * np.cumprod(factors), index=returns.index)


def _market_series(seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0004, 0.008, _N), index=_INDEX)
    returns.iloc[0] = 0.0
    close = _close_from_returns(returns, start=400.0)
    return close, close.pct_change()


# --------------------------------------------------------------------------- #
# Strategy 1: residual_momentum
# --------------------------------------------------------------------------- #


def test_strategy_registry_metadata() -> None:
    discover_plugins()

    residual = registry.get_optional("residual_momentum")
    assert residual is not None
    assert residual.entry == "resid_mom > 0"
    assert residual.exit is None
    assert residual.prepare_bars is not None
    assert residual.required_lookback is not None
    assert residual.required_lookback() == 505

    reversal = registry.get_optional("short_term_reversal")
    assert reversal is not None
    assert reversal.entry == "st_rev > 0"
    assert reversal.exit is None
    assert reversal.prepare_bars is not None
    assert reversal.required_lookback is not None
    assert reversal.required_lookback() == 23


def test_residual_momentum_hand_checked_and_warmup() -> None:
    bench_close, bench_returns = _market_series(seed=3)
    rng = np.random.default_rng(7)
    idiosyncratic = pd.Series(rng.normal(0.0002, 0.011, _N), index=_INDEX)
    stock_returns = 0.8 * bench_returns.fillna(0.0) + idiosyncratic
    stock_returns.iloc[0] = 0.0
    close = _close_from_returns(stock_returns)

    score = residual_momentum_score(close, bench_returns)
    t = 560

    # Independent vectorized market-model calculation. Each row is one trailing
    # 252-return window; no pandas rolling operation from the implementation is
    # reused here.
    stock_ret = close.pct_change().to_numpy()[1:]
    market_ret = bench_close.pct_change().to_numpy()[1:]
    stock_windows = sliding_window_view(stock_ret, _BETA_WINDOW)
    market_windows = sliding_window_view(market_ret, _BETA_WINDOW)
    stock_mean = stock_windows.mean(axis=1)
    market_mean = market_windows.mean(axis=1)
    centered_stock = stock_windows - stock_mean[:, None]
    centered_market = market_windows - market_mean[:, None]
    beta = np.sum(centered_stock * centered_market, axis=1) / np.sum(
        centered_market**2, axis=1
    )
    alpha = stock_mean - beta * market_mean
    residual = stock_ret[_BETA_WINDOW - 1 :] - (
        alpha + beta * market_ret[_BETA_WINDOW - 1 :]
    )

    formation_end = t - _SKIP
    residual_start = formation_end - _FORMATION_WINDOW + 1 - _BETA_WINDOW
    residual_stop = formation_end - _BETA_WINDOW + 1
    formation_residual = residual[residual_start:residual_stop]
    assert formation_residual.size == _FORMATION_WINDOW
    expected = formation_residual.mean() / formation_residual.std(ddof=1)
    assert np.isclose(score.iloc[t], expected)

    # First beta/alpha/residual is at index 252; 231 residuals end at index 482,
    # then the 21-day exclusion moves the first score to index 503.
    assert score.iloc[:503].isna().all()
    assert score.iloc[503:].notna().all()


def test_residual_momentum_is_causal() -> None:
    bench_close, bench_returns = _market_series(seed=5)
    rng = np.random.default_rng(9)
    stock_returns = 0.6 * bench_returns.fillna(0.0) + pd.Series(
        rng.normal(0.0003, 0.012, _N), index=_INDEX
    )
    stock_returns.iloc[0] = 0.0
    close = _close_from_returns(stock_returns)
    base = residual_momentum_score(close, bench_returns)
    t = 540

    close_mutated = close.copy()
    close_mutated.iloc[t + 1 :] *= 6.0
    bench_mutated = bench_close.copy()
    bench_mutated.iloc[t + 1 :] *= 0.4
    mutated = residual_momentum_score(close_mutated, bench_mutated.pct_change())
    assert np.isclose(base.iloc[t], mutated.iloc[t])


def test_residual_prepare_columns_rank_score_empty_and_none() -> None:
    bench_close, bench_returns = _market_series(seed=11)
    rng = np.random.default_rng(13)
    stock_returns = 0.7 * bench_returns.fillna(0.0) + pd.Series(
        rng.normal(0.0002, 0.01, _N), index=_INDEX
    )
    stock_returns.iloc[0] = 0.0
    bars = {"A": _ohlcv(_close_from_returns(stock_returns)), "EMPTY": pd.DataFrame()}
    ctx = _ctx(bars, price_panel={"SPY": _ohlcv(bench_close)})
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]

    out = _prepare_residual_momentum(ctx)
    assert not ctx.warnings
    assert "resid_mom" in out["A"].columns
    pd.testing.assert_series_equal(
        out["A"]["rank_score"], out["A"]["resid_mom"], check_names=False
    )
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_residual_prepare_missing_benchmark_warns_and_nans() -> None:
    close = pd.Series(np.linspace(80.0, 120.0, _N), index=_INDEX)
    ctx = _ctx(
        {"A": _ohlcv(close), "EMPTY": pd.DataFrame()},
        fetcher=StubPriceFetcher({}),
    )
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]

    out = _prepare_residual_momentum(ctx)
    assert any("residual_momentum" in warning for warning in ctx.warnings)
    assert any("benchmark" in warning for warning in ctx.warnings)
    assert out["A"]["resid_mom"].isna().all()
    assert out["A"]["rank_score"].isna().all()
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_residual_momentum_end_to_end_selects_best_name() -> None:
    bench_close, bench_returns = _market_series(seed=17)
    rng = np.random.default_rng(19)
    step = np.where(np.arange(_N) < 280, -1.0, 1.0)
    best_idio = 0.0025 * step + rng.normal(0.0, 0.0004, _N)
    middle_idio = 0.0012 * step + rng.normal(0.0, 0.0015, _N)
    loser_idio = -0.0020 * step + rng.normal(0.0, 0.0005, _N)

    def stock(idiosyncratic: np.ndarray) -> pd.Series:
        returns = 0.8 * bench_returns.fillna(0.0) + pd.Series(
            idiosyncratic, index=_INDEX
        )
        returns.iloc[0] = 0.0
        return _close_from_returns(returns)

    best = stock(best_idio)
    middle = stock(middle_idio)
    loser = stock(loser_idio)
    signal_day = 560
    scores = {
        "BEST": residual_momentum_score(best, bench_returns).iloc[signal_day],
        "MIDDLE": residual_momentum_score(middle, bench_returns).iloc[signal_day],
        "LOSER": residual_momentum_score(loser, bench_returns).iloc[signal_day],
    }
    assert scores["BEST"] > scores["MIDDLE"] > 0.0 > scores["LOSER"]

    data = {
        "BEST": _ohlcv(best),
        "MIDDLE": _ohlcv(middle),
        "LOSER": _ohlcv(loser),
        "SPY": _ohlcv(bench_close),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[570].date(),
        hold=20,
        top=1,
        strategy_name="residual_momentum",
        entry_expr="resid_mom > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("BEST", "MIDDLE", "LOSER"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        # The rolling setup initially pads expression history by one calendar
        # year, then applies the strategy's stricter 505-bar eligibility floor.
        # Starting the simulation earlier lets that fetched panel accumulate the
        # full residual-history requirement before the asserted selection date.
        start_date=_INDEX[300].date(),
        end_date=_INDEX[570].date(),
    )
    traded = {trade.ticker for trade in result.trades}
    assert traded == {"BEST"}, traded


# --------------------------------------------------------------------------- #
# Strategy 2: short_term_reversal
# --------------------------------------------------------------------------- #


def test_short_term_reversal_hand_checked_warmup_and_causal() -> None:
    close = pd.Series(100.0 * 0.997 ** np.arange(_N), index=_INDEX)
    score = short_term_reversal_score(close)
    t = 80
    expected = -(close.iloc[t - 1] / close.iloc[t - 22] - 1.0)
    assert np.isclose(score.iloc[t], expected)
    assert score.iloc[:22].isna().all()
    assert score.iloc[22:].notna().all()

    mutated = close.copy()
    mutated.iloc[t + 1 :] *= 8.0
    assert np.isclose(score.iloc[t], short_term_reversal_score(mutated).iloc[t])


def test_short_term_reversal_prepare_rank_score_empty_and_none() -> None:
    close = pd.Series(100.0 * 0.998 ** np.arange(_N), index=_INDEX)
    ctx = _ctx({"A": _ohlcv(close), "EMPTY": pd.DataFrame()})
    ctx.bars_by_tv["NONE"] = None  # type: ignore[assignment]

    out = _prepare_short_term_reversal(ctx)
    assert "st_rev" in out["A"].columns
    pd.testing.assert_series_equal(
        out["A"]["rank_score"], out["A"]["st_rev"], check_names=False
    )
    assert out["EMPTY"].empty
    assert out["NONE"] is None


def test_short_term_reversal_end_to_end_selects_biggest_loser() -> None:
    biggest_loser = pd.Series(100.0 * 0.990 ** np.arange(_N), index=_INDEX)
    mild_loser = pd.Series(100.0 * 0.997 ** np.arange(_N), index=_INDEX)
    winner = pd.Series(100.0 * 1.002 ** np.arange(_N), index=_INDEX)
    spy = pd.Series(400.0 * 1.001 ** np.arange(_N), index=_INDEX)
    data = {
        "BIGGEST_LOSER": _ohlcv(biggest_loser),
        "MILD_LOSER": _ohlcv(mild_loser),
        "WINNER": _ohlcv(winner),
        "SPY": _ohlcv(spy),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[50].date(),
        hold=20,
        top=1,
        strategy_name="short_term_reversal",
        entry_expr="st_rev > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("BIGGEST_LOSER", "MILD_LOSER", "WINNER"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[40].date(),
        end_date=_INDEX[50].date(),
    )
    traded = {trade.ticker for trade in result.trades}
    assert traded == {"BIGGEST_LOSER"}, traded
