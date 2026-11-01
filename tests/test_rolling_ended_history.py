"""A ticker whose price history ends inside the rolling window.

Delistings, mergers and feeds that simply stop all look the same to the
engine: the ticker's bars end before the master calendar does. Nothing after
that last bar can reach the exit checks, the fill checks or the liquidity
ranking, so each of them used to treat the ticker as still present forever:

* a held position kept its slot (and its capital) to the end of the window
  and was then force-closed with an exit stamp months in the past;
* a pending limit order on it kept its slot reserved for an order that could
  never fill;
* the dynamic universe kept ranking its final lagged turnover, so it held a
  top-N seat no candidate could use.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from screener.backtester import BacktestConfig, run_rolling_backtest
from screener.backtester.rolling_candidates import _build_rolling_candidate_matrices
from tests.conftest import StubPriceFetcher

DATES = pd.bdate_range("2024-01-01", periods=80)


def _bars(n: int, *, base: float, volume: float, dip: float = 0.0) -> pd.DataFrame:
    """Gently rising bars; ``dip`` sets how far each bar's low sits below close."""
    close = base + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - dip,
            "close": close,
            "volume": np.full(n, volume),
        },
        index=DATES[:n],
    )


def _config(**overrides) -> BacktestConfig:
    values = dict(
        market="us",
        as_of=DATES[-1].date(),
        benchmark="SPY",
        tickers=("AAA", "BBB"),
        entry_expr="close > 0",
        exit_expr=None,
        hold=500,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=100_000.0,
        max_universe=0,
        min_price=0,
        min_avg_dollar_volume=0,
    )
    values.update(overrides)
    return BacktestConfig(**values)


def _run(cfg: BacktestConfig, data: dict[str, pd.DataFrame]):
    return run_rolling_backtest(
        cfg,
        StubPriceFetcher({**data, "SPY": _bars(80, base=400.0, volume=1e6)}),
        start_date=DATES[5].date(),
        end_date=DATES[-1].date(),
    )


def test_a_holding_whose_history_ends_frees_its_slot_at_its_last_close():
    # AAA is the most liquid name, so it takes the only slot, then stops
    # trading at bar 39 with a 500-bar hold still running.
    data = {
        "AAA": _bars(40, base=100.0, volume=1e6),
        "BBB": _bars(80, base=50.0, volume=1e4),
    }

    result = _run(_config(), data)

    first, second = result.trades
    assert (first.ticker, first.exit_reason) == ("AAA", "eod")
    assert first.exit_date == DATES[39].date()
    assert first.exit_price == data["AAA"]["close"].iloc[-1]
    # The first session past AAA's last bar freed the slot and ranked BBB, so
    # BBB fills at the next open instead of never trading at all.
    assert second.ticker == "BBB"
    assert second.signal_date == DATES[40].date()
    assert second.entry_date == DATES[41].date()


def test_a_limit_order_on_a_ticker_with_no_bars_left_is_dropped():
    # AAA's lows never reach a limit 5% under the signal close, so its order
    # waits. Once AAA has no bars left it can never fill, and the slot it
    # reserved must go back to the ranking; BBB's deep lows fill at once.
    data = {
        "AAA": _bars(40, base=100.0, volume=1e6),
        "BBB": _bars(80, base=50.0, volume=1e4, dip=10.0),
    }

    result = _run(_config(entry_order_type="limit", entry_limit_bps=500), data)

    assert [trade.ticker for trade in result.trades] == ["BBB"]
    assert result.trades[0].signal_date == DATES[40].date()


def test_a_ticker_whose_history_ended_gives_up_its_dynamic_universe_seat():
    bars = {
        "AAA": _bars(40, base=100.0, volume=1e6),
        "BBB": _bars(80, base=50.0, volume=1e4),
    }
    always = {tv: np.ones(len(frame), dtype=bool) for tv, frame in bars.items()}

    matrices = _build_rolling_candidate_matrices(
        bars,
        always,
        {},
        list(DATES),
        0,
        dynamic_universe_size=1,
        dynamic_universe_lookback=5,
        dynamic_universe_rebalance="daily",
    )

    signal = matrices.signal_mat
    # While AAA trades it is the one liquid enough to hold the single seat.
    assert signal["AAA"].iloc[6:40].all()
    assert not signal["BBB"].iloc[6:40].any()
    # Past its last bar AAA's final lagged turnover no longer ranks, so the
    # seat passes to BBB instead of sitting empty for the rest of the window.
    assert not signal["AAA"].iloc[40:].any()
    assert signal["BBB"].iloc[40:].all()
