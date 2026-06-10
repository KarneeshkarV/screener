from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import run_rolling_backtest
from screener.regime import classify_regimes, vol_regime


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# classify_regimes
# ---------------------------------------------------------------------------


def test_classify_regimes_warmup_is_unknown():
    close = _series([100.0 + i for i in range(150)])
    labels = classify_regimes(close)
    assert (labels == "unknown").all()


def test_classify_regimes_empty_series():
    labels = classify_regimes(pd.Series(dtype=float))
    assert labels.empty


def test_classify_regimes_uptrend_is_bull_after_warmup():
    close = _series([100.0 + 0.5 * i for i in range(260)])
    labels = classify_regimes(close)
    assert (labels.iloc[:199] == "unknown").all()
    assert (labels.iloc[199:] == "bull").all()


def test_classify_regimes_downtrend_is_bear_after_warmup():
    close = _series([500.0 - 0.5 * i for i in range(260)])
    labels = classify_regimes(close)
    assert (labels.iloc[:199] == "unknown").all()
    assert (labels.iloc[199:] == "bear").all()


def test_classify_regimes_flat_series_is_pullback():
    close = _series([100.0] * 260)
    labels = classify_regimes(close)
    assert (labels.iloc[:199] == "unknown").all()
    assert (labels.iloc[199:] == "pullback").all()


def test_classify_regimes_no_lookahead():
    # Uptrend that crashes at the end: labels computed on a truncated series
    # must match the full-series prefix (each date uses only past data).
    values = [100.0 + 0.5 * i for i in range(240)] + [50.0] * 20
    close = _series(values)
    full = classify_regimes(close)
    cut = 230
    truncated = classify_regimes(close.iloc[:cut])
    assert truncated.equals(full.iloc[:cut])


# ---------------------------------------------------------------------------
# vol_regime
# ---------------------------------------------------------------------------


def test_vol_regime_warmup_is_unknown():
    close = _series([100.0 + (i % 2) for i in range(271)])
    labels = vol_regime(close)
    assert (labels == "unknown").all()


def test_vol_regime_flat_series_is_normal_after_warmup():
    # Constant returns: realized vol is identical across the trailing window,
    # so the percentile rank sits mid-distribution -> 'normal'.
    close = _series([100.0] * 300)
    labels = vol_regime(close)
    assert (labels.iloc[:271] == "unknown").all()
    assert (labels.iloc[271:] == "normal").all()


def test_vol_regime_spike_is_high_vol():
    calm = [100.0 + 0.1 * (i % 2) for i in range(280)]
    wild = [100.0 + 8.0 * (i % 2) for i in range(20)]
    close = _series(calm + wild)
    labels = vol_regime(close)
    assert labels.iloc[-1] == "high_vol"
    # Calm stretch (past warmup, before the spike) stays normal.
    assert (labels.iloc[271:279] == "normal").all()


# ---------------------------------------------------------------------------
# rolling backtest integration (--regime-filter + per-regime metrics)
# ---------------------------------------------------------------------------


def _trend_bars(
    start: str = "2024-01-01", n: int = 60, step: float = 1.0, base: float = 100.0
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    close = pd.Series([base + step * i for i in range(n)], index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - step)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=3,
        top=2,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB"),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _stub_data() -> dict[str, pd.DataFrame]:
    # Benchmark needs >200 bars of pre-window history so its regime is
    # defined ('bull': monotonic uptrend) by the backtest window.
    return {
        "AAA": _trend_bars(),
        "BBB": _trend_bars(),
        "SPY": _trend_bars(start="2022-12-01", n=330, step=0.5),
    }


def test_regime_filter_suppresses_entries_when_regime_not_allowed(
    stub_fetcher_factory,
):
    fetcher = stub_fetcher_factory(_stub_data())
    start, end = date(2024, 2, 1), date(2024, 3, 1)

    baseline = run_rolling_backtest(_cfg(), fetcher, start_date=start, end_date=end)
    assert baseline.trades, "baseline should produce trades"

    filtered = run_rolling_backtest(
        _cfg(regime_filter=("bear",)), fetcher, start_date=start, end_date=end
    )
    assert filtered.trades == []
    assert filtered.selection.empty


def test_regime_filter_allows_entries_in_matching_regime(stub_fetcher_factory):
    fetcher = stub_fetcher_factory(_stub_data())
    start, end = date(2024, 2, 1), date(2024, 3, 1)

    baseline = run_rolling_backtest(_cfg(), fetcher, start_date=start, end_date=end)
    allowed = run_rolling_backtest(
        _cfg(regime_filter=("bull",)), fetcher, start_date=start, end_date=end
    )
    assert [(t.ticker, t.entry_date) for t in allowed.trades] == [
        (t.ticker, t.entry_date) for t in baseline.trades
    ]


def test_rolling_metrics_include_per_regime_trade_stats(stub_fetcher_factory):
    fetcher = stub_fetcher_factory(_stub_data())
    result = run_rolling_backtest(
        _cfg(), fetcher, start_date=date(2024, 2, 1), end_date=date(2024, 3, 1)
    )
    assert result.trades
    assert result.metrics["regime_bull_trades"] == len(result.trades)
    assert result.metrics["regime_bull_win_rate"] == 1.0
    assert result.metrics["regime_bull_avg_return"] > 0
