"""Regressions from the backtest correctness audit.

Each test pins one defect that produced a wrong number on a synthetic panel
small enough to check by hand.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.data import ExchangeFallbackPriceFetcher
from screener.backtester.fundamentals import merge_fundamentals_into_bars
from screener.backtester.metrics import (
    _cagr,
    _paired_returns,
    _session_returns,
    _trade_return_stats,
    bar_returns,
    compute_metrics,
    position_count,
    positive_pnl_rate,
)
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization.metrics import score_result
from screener.backtester.price_frames import (
    apply_full_price_adjustment,
    normalize_price_frame,
)
from screener.backtester.rolling_simulation import run_rolling_backtest


class _Fetcher:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, tickers, start, end) -> dict[str, pd.DataFrame]:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) + pd.Timedelta(days=1)
        return {
            t: self.frames[t]
            .loc[(self.frames[t].index >= lo) & (self.frames[t].index < hi)]
            .copy()
            for t in tickers
        }


def _bars(idx: pd.DatetimeIndex, price: float = 100.0, volume: float = 100.0):
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": volume},
        index=idx,
    )


def _cfg(**overrides) -> BacktestConfig:
    args: dict = {
        "market": "us",
        "as_of": date(2024, 1, 1),
        "benchmark": "SPY",
        "tickers": ("AAA",),
        "entry_expr": "close > 0",
        "exit_expr": None,
        "hold": 1000,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "top": 1,
        "initial_capital": 10000.0,
        "price_adjustment": "none",
    }
    args.update(overrides)
    return BacktestConfig(**args)


def _run(
    frames: dict[str, pd.DataFrame],
    config: BacktestConfig,
    *,
    start: date | None = None,
    benchmark: pd.DataFrame | None = None,
) -> BacktestResult:
    idx = max((f.index for f in frames.values()), key=len)
    spy = _bars(idx) if benchmark is None else benchmark
    return run_rolling_backtest(
        config,
        _Fetcher({**frames, "SPY": spy}),
        start_date=start or idx[0].date(),
        end_date=idx[-1].date(),
    )


def test_bar_with_missing_high_does_not_fill_the_target() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    frame = _bars(idx)
    frame.loc[idx[2], ["open", "high", "low", "close"]] = np.nan

    result = _run({"AAA": frame}, _cfg(take_profit=0.1))

    assert [t.exit_reason for t in result.trades] != ["target"]
    assert result.equity_curve.iloc[-1] == 10000.0


def test_full_adjustment_drops_bars_without_an_adjusted_close() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    raw = _bars(idx)
    raw["adj_close"] = 100.0
    raw.loc[idx[2], "adj_close"] = np.nan

    adjusted = apply_full_price_adjustment(normalize_price_frame(raw))

    assert list(adjusted.index) == [idx[0], idx[1], idx[3]]
    assert not adjusted[["open", "high", "low", "close"]].isna().any().any()


def test_high_exactly_at_the_target_level_fills() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    frame = _bars(idx)
    frame.loc[idx[3], "high"] = 110.0

    result = _run({"AAA": frame}, _cfg(take_profit=0.1))

    first = result.trades[0]
    assert (first.exit_reason, first.exit_price) == ("target", 110.0)


def test_mid_month_start_holds_the_membership_of_the_months_first_session() -> None:
    idx = pd.bdate_range("2023-12-01", "2024-01-31")
    leader = _bars(idx, volume=1000.0)
    laggard = _bars(idx, volume=100.0)
    # The ranking flips on 8 January. A monthly universe does not see that
    # until February, whatever day the run starts on.
    leader.loc["2024-01-08":, "volume"] = 10.0
    laggard.loc["2024-01-08":, "volume"] = 2000.0
    config = _cfg(
        tickers=("AAA", "BBB"),
        dynamic_universe_size=1,
        dynamic_universe_lookback=2,
        dynamic_universe_rebalance="monthly",
    )
    frames = {"AAA": leader, "BBB": laggard}

    from_month_start = _run(frames, config, start=date(2024, 1, 1))
    from_mid_month = _run(frames, config, start=date(2024, 1, 15))

    assert from_month_start.trades[0].ticker == "AAA"
    assert from_mid_month.trades[0].ticker == "AAA"


def test_intraday_flat_does_not_read_a_halted_tickers_next_bar() -> None:
    # 14:30-20:30 UTC is the US session. AAA prints three bars, halts, and
    # reopens the next day at 1. At 16:30 nobody knows the halt is coming, so
    # the position cannot be flattened there at 100.
    stamps = pd.DatetimeIndex(
        [f"2024-01-02 {hour}:30" for hour in range(14, 21)]
        + ["2024-01-03 14:30", "2024-01-03 15:30"]
    )
    halted = _bars(stamps[:3].append(stamps[7:]), volume=200.0)
    halted.loc["2024-01-03":, ["open", "high", "low", "close"]] = 1.0
    config = _cfg(tickers=("AAA", "BBB"), interval="1h", intraday_only=True)

    result = _run({"AAA": halted, "BBB": _bars(stamps)}, config)

    first = result.trades[0]
    assert (first.ticker, first.exit_reason) == ("AAA", "session")
    assert pd.Timestamp(first.exit_date) == pd.Timestamp("2024-01-03 14:30")
    assert first.exit_price == 1.0


def _trade(pnl: float, *, open_seq: int = 0, entry_cost: float = 1000.0) -> Trade:
    return Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        exit_date=date(2024, 1, 5),
        exit_price=100.0,
        exit_reason="time",
        shares=entry_cost / 100.0,
        entry_cost=entry_cost,
        exit_value=entry_cost + pnl,
        pnl=pnl,
        return_pct=-0.01,
        open_seq=open_seq,
    )


def test_cagr_of_a_short_intraday_curve_does_not_overflow() -> None:
    equity = pd.Series([100.0, 200.0])

    assert _cagr(equity, periods_per_year=252 * 390) == float("inf")


def test_one_session_run_reports_its_intraday_loss_as_tail_risk() -> None:
    stamps = pd.date_range("2024-01-02 14:30", periods=4, freq="1h")
    equity = pd.Series([100.0, 99.0, 95.0, 90.0], index=stamps)

    session = _session_returns(equity)

    assert session.tolist() == pytest.approx([-0.10])


def test_bar_returns_drops_a_move_from_zero_equity() -> None:
    assert bar_returns(pd.Series([100.0, 0.0, 50.0])).tolist() == [-1.0]


def test_partial_exit_tranches_count_as_one_position() -> None:
    tranches = [_trade(50.0, open_seq=1), _trade(30.0, open_seq=1), _trade(-20.0)]

    stats = _trade_return_stats(tranches)

    assert position_count(tranches) == 2
    assert positive_pnl_rate(tranches) == 0.5
    assert (stats["winning_trades"], stats["losing_trades"]) == (1, 1)


def test_trade_return_stats_agree_with_pnl_when_a_dividend_makes_the_win() -> None:
    # Capital return is -1% but the dividend makes the trade a 3% winner.
    stats = _trade_return_stats([_trade(30.0)])

    assert stats["winning_trades"] == 1
    assert stats["best_trade_return"] == pytest.approx(0.03)
    assert stats["expectancy"] == pytest.approx(0.03)


def test_benchmark_gap_is_not_paired_as_a_flat_bar() -> None:
    idx = pd.bdate_range("2024-01-01", periods=7)
    equity = pd.Series([100.0, 101, 103, 102, 104, 107, 106], index=idx)
    benchmark = pd.Series([100.0, 102, np.nan, 101, 103, 105, 104], index=idx)
    filled = benchmark.ffill()

    with_marks = compute_metrics(equity, filled, [], 1, benchmark_marks=benchmark)
    without_gap_bars = compute_metrics(
        equity.drop(idx[[2, 3]]), benchmark.drop(idx[[2, 3]]), [], 1
    )
    filled_only = compute_metrics(equity, filled, [], 1)

    paired_equity, paired_bench = _paired_returns(equity, benchmark)
    assert len(paired_bench) == 4
    assert with_marks["beta"] == pytest.approx(
        float(
            np.cov(paired_equity, paired_bench, ddof=0)[0, 1] / paired_bench.var(ddof=0)
        )
    )
    assert with_marks["beta"] != pytest.approx(filled_only["beta"])
    assert with_marks["benchmark_return"] == filled_only["benchmark_return"]
    assert without_gap_bars["beta"] != 0.0


def test_unknown_optimization_metric_is_an_error_not_a_zero_score() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    result = _run({"AAA": _bars(idx)}, _cfg())

    with pytest.raises(ValueError, match="Unknown optimization metric 'sharp'"):
        score_result(result, "sharp")
    assert score_result(result, "sharpe") == result.metrics["sharpe"]


def test_fundamental_value_expires_when_the_feed_stops() -> None:
    idx = pd.bdate_range("2019-01-01", "2022-12-30")
    filings = pd.DataFrame(
        {"pe": [10.0, 12.0], "roe": [0.2, np.nan]},
        index=pd.DatetimeIndex(["2019-01-15", "2020-01-15"]),
    )

    merged = merge_fundamentals_into_bars(
        {"NSE:AAA": _bars(idx)},
        {"AAA.NS": filings},
        {"NSE:AAA": "AAA.NS"},
        filing_lag_days=0,
        effective_date_kind="filing_timestamp",
    )["NSE:AAA"]

    assert merged.loc["2021-06-30", "pe"] == 12.0
    assert np.isnan(merged.loc["2021-08-02", "pe"])
    # ``roe`` was last reported a year before ``pe``: it expires a year sooner.
    assert merged.loc["2020-06-30", "roe"] == 0.2
    assert np.isnan(merged.loc["2020-08-03", "roe"])


def test_rolling_run_names_the_historical_only_settings_it_ignores() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)

    quiet = _run({"AAA": _bars(idx)}, _cfg())
    loud = _run({"AAA": _bars(idx)}, _cfg(allow_reentry=True, max_reentries=2))

    assert not any("historical-only" in w for w in quiet.warnings)
    assert any(
        "historical-only settings: allow_reentry, max_reentries" in w
        for w in loud.warnings
    )


def test_config_rejects_an_unimplemented_concurrent_lot_count() -> None:
    with pytest.raises(ValueError, match="max_concurrent_per_ticker"):
        _cfg(max_concurrent_per_ticker=2)


def test_price_only_benchmark_is_named_when_the_book_earns_dividends() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    spy = _bars(idx)
    spy["dividend"] = 0.0
    spy.loc[idx[3], "dividend"] = 1.5

    credited = _run({"AAA": _bars(idx)}, _cfg(), benchmark=spy)
    plain = _run({"AAA": _bars(idx)}, _cfg())

    assert any("benchmark SPY is price-only" in w for w in credited.warnings)
    assert not any("price-only" in w for w in plain.warnings)


def test_nse_symbol_served_from_bse_is_logged(monkeypatch) -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    events: list[tuple[str, dict]] = []

    class _Log:
        def warning(self, event: str, **fields) -> None:
            events.append((event, fields))

    monkeypatch.setattr("screener.logging_config.get_logger", lambda _name: _Log())
    inner = _Fetcher({"AAA.NS": _bars(idx[:0]), "AAA.BO": _bars(idx)})

    out = ExchangeFallbackPriceFetcher(inner).fetch(["AAA.NS"], idx[0], idx[-1])

    assert len(out["AAA.NS"]) == 3
    assert events == [
        ("nse_symbols_served_from_bse", {"count": 1, "tickers": ["AAA.NS"]})
    ]
