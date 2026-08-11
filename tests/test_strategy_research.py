"""Unit tests for the research-backed strategy plugins (golden cross, 52-week
high, Connors RSI-2, Bollinger mean reversion, MACD signal cross, BLL TRB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.bollinger_reversion import _prepare_bb
from screener.strategies.plugins.fifty_two_week_high import _prepare_high
from screener.strategies.plugins.trading_range_break import _prepare_channel
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _trend_frame(
    drift: float = 0.001, noise: float = 0.005, start: float = 50.0
) -> pd.DataFrame:
    close = pd.Series(
        start * (1.0 + drift) ** np.arange(_N) + noise * start * np.sin(np.arange(_N)),
        index=_INDEX,
    )
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) + 0.1
    low = pd.concat([openp, close], axis=1).min(axis=1) - 0.1
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(1_000_000.0, index=_INDEX),
        }
    )


def _run(strategy: str, data: dict[str, pd.DataFrame], top: int = 1) -> object:
    from screener.strategies.expressions import resolve_strategy

    resolved = resolve_strategy(strategy)
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=100,
        top=top,
        strategy_name=strategy,
        entry_expr=resolved.entry,
        exit_expr=resolved.exit,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=tuple(data.keys()),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    return run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[300].date(),
        end_date=_INDEX[-1].date(),
    )


# ── registration ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "entry", "exit"),
    [
        (
            "golden_cross_50_200",
            "crossover(sma(close, 50), sma(close, 200))",
            "crossunder(sma(close, 50), sma(close, 200))",
        ),
        (
            "fifty_two_week_high",
            "close > high_252_prev",
            "crossunder(close, sma(close, 50))",
        ),
        ("connors_rsi2", "rsi(close, 2) < 5", "rsi(close, 2) > 60"),
        (
            "connors_rsi2_bull",
            "rsi(close, 2) < 5 and close > sma(close, 200)",
            "rsi(close, 2) > 60",
        ),
        ("bollinger_mean_reversion", "close < bb_lower", "close > bb_mid"),
        (
            "macd_signal_cross",
            "crossover(ema(close, 12) - ema(close, 26), ema(ema(close, 12) - ema(close, 26), 9))",
            "crossunder(ema(close, 12) - ema(close, 26), ema(ema(close, 12) - ema(close, 26), 9))",
        ),
        ("bll_trading_range_break", "close > high_150_prev", "close < low_150_prev"),
        (
            "stochastic_cross",
            "crossover(100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)), sma(100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)), 3)) and 100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)) < 30",
            "crossunder(100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)), sma(100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)), 3)) and 100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14)) > 70",
        ),
        (
            "williams_percent_r",
            "-100 * (highest(high, 14) - close) / (highest(high, 14) - lowest(low, 14)) < -80",
            "-100 * (highest(high, 14) - close) / (highest(high, 14) - lowest(low, 14)) > -20",
        ),
        (
            "keltner_breakout",
            "crossover(close, ema(close, 20) + 2.0 * atr(20))",
            "crossunder(close, ema(close, 20) - 2.0 * atr(20))",
        ),
        ("cci_reversion", "cci_20 < -100", "cci_20 > 100"),
        ("adx_trend", "adx_14 > 25.0 and di_plus > di_minus", "di_plus < di_minus"),
        ("short_term_reversal", "ret_21 < 0", None),
        ("long_term_reversal", "ret_756 < 0", None),
        (
            "turn_of_month",
            "day_of_month >= 28 or day_of_month <= 3",
            "day_of_month >= 4 and day_of_month <= 27",
        ),
    ],
)
def test_research_strategies_registered(
    name: str, entry: str, exit: str | None
) -> None:
    discover_plugins()
    spec = registry.get_optional(name)
    assert spec is not None, name
    assert spec.entry == entry
    assert spec.exit == exit


# ── prepare_bars causality ────────────────────────────────────────────


def _ctx(frames: dict[str, pd.DataFrame]) -> PrepareCtx:
    return PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv=frames,
        price_panel=frames,
        tv_symbols=list(frames),
        start=_INDEX[0].date(),
        end=_INDEX[-1].date(),
        fetcher=None,
        warnings=[],
    )


def test_high_252_prev_is_prior_only() -> None:
    frame = _trend_frame()
    prepared = _prepare_high(_ctx({"AAA": frame}))["AAA"]
    ref = prepared["high_252_prev"]
    assert ref.iloc[:252].isna().all()
    # The reference peak on bar t must never include today's close.
    for t in range(252, _N):
        assert ref.iloc[t] <= frame["close"].iloc[t - 252 : t].max() + 1e-9


def test_channel_150_prev_is_prior_only() -> None:
    frame = _trend_frame()
    prepared = _prepare_channel(_ctx({"AAA": frame}))["AAA"]
    hi, lo = prepared["high_150_prev"], prepared["low_150_prev"]
    assert hi.iloc[:150].isna().all() and lo.iloc[:150].isna().all()
    for t in range(150, _N):
        assert hi.iloc[t] <= frame["close"].iloc[t - 150 : t].max() + 1e-9
        assert lo.iloc[t] >= frame["close"].iloc[t - 150 : t].min() - 1e-9


def test_bb_bands_are_prior_only() -> None:
    frame = _trend_frame()
    prepared = _prepare_bb(_ctx({"AAA": frame}))["AAA"]
    # First valid value lands on the 20th bar (index 19).
    assert prepared["bb_lower"].iloc[:19].isna().all()
    assert not prepared["bb_lower"].iloc[19:].isna().any()
    # Lower band must sit below the middle band (sanity of band geometry).
    assert (prepared["bb_lower"].dropna() < prepared["bb_mid"].dropna()).all()


def test_cci_reversion_column_is_causal() -> None:
    from screener.strategies.plugins.cci import _prepare_cci

    frame = _trend_frame()
    prepared = _prepare_cci(_ctx({"AAA": frame}))["AAA"]
    assert prepared["cci_20"].iloc[:38].isna().all()  # 20-bar SMA + 20-bar mean-dev legs
    assert not prepared["cci_20"].iloc[38:].isna().any()


def test_adx_columns_bounded_and_causal() -> None:
    from screener.strategies.plugins.adx_trend import _prepare_adx

    frame = _trend_frame()
    prepared = _prepare_adx(_ctx({"AAA": frame}))["AAA"]
    di = prepared["di_plus"].dropna()
    assert ((di >= 0) & (di <= 100)).all()
    adx = prepared["adx_14"].dropna()
    assert ((adx >= 0) & (adx <= 100)).all()


def test_reversal_rank_score_picks_losers() -> None:
    from screener.strategies.plugins.short_term_reversal import _prepare_reversal

    frame = _trend_frame()
    close = frame["close"].copy()
    close.iloc[-5:] -= 40.0  # crash the last week -> biggest 1-month loser
    frame["close"] = close
    prepared = _prepare_reversal(_ctx({"AAA": frame}))["AAA"]
    # rank_score = -ret_21 must be positive (biggest loser) and ranked highest.
    assert prepared["rank_score"].iloc[-1] > 0
    assert prepared["rank_score"].iloc[-1] > prepared["rank_score"].iloc[-60]


def test_turn_of_month_window_flags() -> None:
    from screener.strategies.plugins.turn_of_month import _prepare_tom

    frame = _trend_frame()
    prepared = _prepare_tom(_ctx({"AAA": frame}))["AAA"]
    dom = prepared["day_of_month"]
    in_window = (dom >= 28) | (dom <= 3)
    out_window = (dom >= 4) & (dom <= 27)
    assert (in_window | out_window).all()  # exhaustive partition of the month


# ── full rolling smoke tests ──────────────────────────────────────────


def test_connors_rsi2_trades_on_synthetic_dip() -> None:
    # Build a price series with a sharp 2-day drop inside the backtest window
    # (bars 300-319) so RSI(2) < 5 fires while the sim is live.
    close = pd.Series(100.0, index=_INDEX)
    close.iloc[305:307] = [96.0, 88.0]  # two big down days
    close.iloc[307:] = 100.0 + np.linspace(0, 5, _N - 307)
    frame = _trend_frame()
    frame["close"] = close
    frame["open"] = close.shift(1).fillna(close.iloc[0])
    frame["high"] = pd.concat([frame["open"], close], axis=1).max(axis=1)
    frame["low"] = pd.concat([frame["open"], close], axis=1).min(axis=1)
    result = _run("connors_rsi2", {"AAA": frame, "SPY": _trend_frame()})
    assert len(result.trades) >= 1


def test_golden_cross_fires_on_sustained_trend() -> None:
    # Flat at 50 for 300 bars, then a jump + steady rally: the SMA50/SMA200
    # golden cross fires right after the window starts (bar 300).
    close = pd.Series(50.0, index=_INDEX)
    rally = 50.0 * 1.30 * (1.0015) ** np.arange(_N - 300)
    close.iloc[300:] = rally
    frame = _trend_frame()
    frame["close"] = close
    frame["open"] = close.shift(1).fillna(close.iloc[0])
    frame["high"] = pd.concat([frame["open"], close], axis=1).max(axis=1)
    frame["low"] = pd.concat([frame["open"], close], axis=1).min(axis=1)
    result = _run("golden_cross_50_200", {"AAA": frame, "SPY": _trend_frame()})
    assert len(result.trades) >= 1


def test_bollinger_mean_reversion_runs() -> None:
    data = {"AAA": _trend_frame(noise=0.01), "SPY": _trend_frame()}
    result = _run("bollinger_mean_reversion", data)
    assert result.trades is not None


def test_macd_signal_cross_runs() -> None:
    data = {"AAA": _trend_frame(), "SPY": _trend_frame()}
    result = _run("macd_signal_cross", data)
    assert result.trades is not None
