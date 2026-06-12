"""Characterization / regression tests for the unified backtest day-loop.

These scenarios pin the numerical output (trades + equity curve) of both the
historical (event-driven) and rolling backtest flows on deterministic synthetic
data. They were written BEFORE the day-loop refactor as a safety net: identical
results must hold before and after extracting the shared per-day orchestration.

Each scenario exercises a distinct mechanic: partial exits, gap fills, trailing
stops, reserve promotion (historical), daily refill (rolling), dividends, and
slippage. The assertions are exact (entry/exit prices, reasons, dates, equity
points), so any behavioural drift surfaces immediately.
"""

from __future__ import annotations

from datetime import date

import pytest

from screener.backtester.engine import run_backtest, run_rolling_backtest
from screener.backtester.models import BacktestConfig

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=10,
        entry_expr="entry_signal > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=None,
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _trade_tuples(result) -> list[tuple]:
    """Canonical, comparable representation of a result's trades."""
    return [
        (
            t.ticker,
            t.rank,
            t.signal_date.isoformat(),
            t.entry_date.isoformat(),
            round(t.entry_price, 8),
            t.exit_date.isoformat(),
            round(t.exit_price, 8),
            t.exit_reason,
            round(t.shares, 8),
            round(t.entry_cost, 8),
            round(t.exit_value, 8),
            round(t.pnl, 8),
            round(t.return_pct, 10),
            round(t.dividend_income, 8),
        )
        for t in result.trades
    ]


def _as_tuples(rows) -> list[tuple]:
    """Normalize golden literal lists to tuples for comparison."""
    return [tuple(row) for row in rows]


def _equity_tuples(result) -> list[tuple]:
    return [
        (ts.date().isoformat(), round(float(v), 6))
        for ts, v in result.equity_curve.items()
    ]


# ── Scenario builders ────────────────────────────────────────────────


def _scn_partial_exits():
    """Historical: a partial-exit tier fires, then a time exit closes the rest."""
    bars = make_bars(n=30, seed=21, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[5, bars.columns.get_loc("entry_signal")] = 1.0
    # entry on bar 6 ~ open. Make bar 8 spike to trigger the +5% partial tier.
    bars.iat[8, bars.columns.get_loc("high")] = 130.0
    spy = make_bars(n=30, seed=22, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[5].date(),
        hold=6,
        top=1,
        tickers=("AAA",),
        partial_exits=((0.05, 0.5),),
    )
    return cfg, fetcher, bars


def _scn_gap_fills():
    """Historical: a gap-down through the stop fills at the (worse) open."""
    bars = make_bars(n=20, seed=31, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[5, bars.columns.get_loc("entry_signal")] = 1.0
    bars.iat[6, bars.columns.get_loc("open")] = 100.0
    # bar 8 gaps down hard through the 5% stop (ref 95): open 90 < 95.
    bars.iat[8, bars.columns.get_loc("open")] = 90.0
    bars.iat[8, bars.columns.get_loc("high")] = 91.0
    bars.iat[8, bars.columns.get_loc("low")] = 85.0
    bars.iat[8, bars.columns.get_loc("close")] = 88.0
    spy = make_bars(n=20, seed=32, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[5].date(),
        hold=10,
        top=1,
        tickers=("AAA",),
        stop_loss=0.05,
        gap_fills=True,
    )
    return cfg, fetcher, bars


def _scn_trailing_stop():
    """Historical: peak lifts then price falls back through the trail."""
    bars = make_bars(n=20, seed=41, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[5, bars.columns.get_loc("entry_signal")] = 1.0
    bars.iat[6, bars.columns.get_loc("open")] = 100.0
    bars.iat[7, bars.columns.get_loc("high")] = 120.0
    bars.iat[7, bars.columns.get_loc("close")] = 118.0
    bars.iat[8, bars.columns.get_loc("open")] = 118.0
    bars.iat[8, bars.columns.get_loc("low")] = 100.0
    bars.iat[8, bars.columns.get_loc("close")] = 101.0
    spy = make_bars(n=20, seed=42, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[5].date(),
        hold=10,
        top=1,
        tickers=("AAA",),
        trailing_stop=0.10,
    )
    return cfg, fetcher, bars


def _scn_slippage_commission():
    """Historical: non-trivial slippage + commission on entry and exit."""
    bars = make_bars(n=20, seed=51, open_base=100.0, drift=0.2)
    bars["entry_signal"] = 0.0
    bars.iat[4, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=20, seed=52, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[4].date(),
        hold=5,
        top=1,
        tickers=("AAA",),
        slippage_bps=50.0,
        commission_bps=10.0,
    )
    return cfg, fetcher, bars


def _scn_reserve_promotion():
    """Historical-only: a stop-out frees a slot and a reserve is promoted."""
    bars_active = make_bars(n=60, seed=1, open_base=100.0)
    bars_reserve = make_bars(n=60, seed=2, open_base=100.0)
    for i in range(30, 60):
        bars_reserve.iat[i, bars_reserve.columns.get_loc("open")] = 100.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("high")] = 101.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("low")] = 99.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("close")] = 100.0
    bars_reserve.iat[41, bars_reserve.columns.get_loc("close")] = 105.0
    bars_reserve.iat[39, bars_reserve.columns.get_loc("close")] = 105.0
    bars_active.iat[39, bars_active.columns.get_loc("close")] = (
        float(bars_active.iloc[39]["close"]) + 5
    )
    bars_active.iat[40, bars_active.columns.get_loc("open")] = 100.0
    bars_active.iat[41, bars_active.columns.get_loc("open")] = 99.0
    bars_active.iat[41, bars_active.columns.get_loc("high")] = 99.5
    bars_active.iat[41, bars_active.columns.get_loc("low")] = 90.0
    bars_active.iat[41, bars_active.columns.get_loc("close")] = 91.0
    bars_active["volume"] = 1_000_000.0
    bars_reserve["volume"] = 100_000.0
    fetcher = StubPriceFetcher(
        {"ACTIVE": bars_active, "RESERVE": bars_reserve, "SPY": bars_reserve.copy()}
    )
    cfg = _cfg(
        as_of=bars_active.index[39].date(),
        hold=10,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("ACTIVE", "RESERVE"),
        stop_loss=0.05,
        reserve_multiple=3,
        reinvest=True,
    )
    return cfg, fetcher, bars_active


def _scn_dividends():
    """Historical: splits_only regime credits a cash dividend mid-hold."""
    bars = make_bars(n=20, seed=61, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[4, bars.columns.get_loc("entry_signal")] = 1.0
    bars["dividend"] = 0.0
    bars.iat[7, bars.columns.get_loc("dividend")] = 1.25
    spy = make_bars(n=20, seed=62, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[4].date(),
        hold=6,
        top=1,
        tickers=("AAA",),
        price_adjustment="splits_only",
    )
    return cfg, fetcher, bars


# ── Historical characterization ──────────────────────────────────────


HISTORICAL_SCENARIOS = {
    "partial_exits": _scn_partial_exits,
    "gap_fills": _scn_gap_fills,
    "trailing_stop": _scn_trailing_stop,
    "slippage_commission": _scn_slippage_commission,
    "reserve_promotion": _scn_reserve_promotion,
    "dividends": _scn_dividends,
}


# Golden values captured from the pre-refactor implementation. These pin exact
# numerical output; regenerate intentionally (and review the diff) only if a
# behaviour change is deliberate.
HISTORICAL_GOLDEN: dict[str, dict] = {
    "dividends": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-05",
                "2024-01-08",
                98.26499225,
                "2024-01-16",
                98.88412561,
                "time",
                1017.6564177,
                100000.0,
                100630.06503841,
                630.06503841,
                0.0063006504,
                1272.07052212,
            ]
        ],
        "equity": [
            ["2024-01-05", 100000.0],
            ["2024-01-08", 100333.260208],
            ["2024-01-09", 100761.312555],
            ["2024-01-10", 100278.378513],
            ["2024-01-11", 99990.722495],
            ["2024-01-12", 101405.126147],
            ["2024-01-15", 100887.424812],
            ["2024-01-16", 100630.065038],
        ],
    },
    "gap_fills": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.0,
                "2024-01-11",
                90.0,
                "stop",
                1000.0,
                100000.0,
                90000.0,
                -10000.0,
                -0.1,
                0.0,
            ]
        ],
        "equity": [
            ["2024-01-08", 100000.0],
            ["2024-01-09", 100654.642156],
            ["2024-01-10", 100790.851273],
            ["2024-01-11", 90000.0],
        ],
    },
    "partial_exits": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.4612186,
                "2024-01-11",
                105.48427953,
                "target",
                497.70449432,
                50000.0,
                52500.0,
                2500.0,
                0.05,
                0.0,
            ],
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.4612186,
                "2024-01-11",
                99.51833198,
                "stop",
                497.70449432,
                50000.0,
                49530.72109284,
                -469.27890716,
                -0.0093855781,
                0.0,
            ],
        ],
        "equity": [
            ["2024-01-08", 100000.0],
            ["2024-01-09", 99600.364833],
            ["2024-01-10", 99061.442186],
            ["2024-01-11", 102030.721093],
        ],
    },
    "reserve_promotion": {
        "trades": [
            [
                "ACTIVE",
                1,
                "2024-02-23",
                "2024-02-26",
                100.0,
                "2024-02-27",
                95.0,
                "stop",
                1000.0,
                100000.0,
                95000.0,
                -5000.0,
                -0.05,
                0.0,
            ],
            [
                "RESERVE",
                2,
                "2024-02-27",
                "2024-02-28",
                100.0,
                "2024-03-13",
                100.0,
                "time",
                950.0,
                95000.0,
                95000.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        "equity": [
            ["2024-02-23", 100000.0],
            ["2024-02-26", 99442.347714],
            ["2024-02-27", 95000.0],
            ["2024-02-28", 95000.0],
            ["2024-02-29", 95000.0],
            ["2024-03-01", 95000.0],
            ["2024-03-04", 95000.0],
            ["2024-03-05", 95000.0],
            ["2024-03-06", 95000.0],
            ["2024-03-07", 95000.0],
            ["2024-03-08", 95000.0],
            ["2024-03-11", 95000.0],
            ["2024-03-12", 95000.0],
            ["2024-03-13", 95000.0],
        ],
    },
    "slippage_commission": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-05",
                "2024-01-08",
                101.41033924,
                "2024-01-15",
                100.62055329,
                "time",
                985.10763941,
                100000.0,
                99022.95364741,
                -977.04635259,
                -0.0097704635,
                0.0,
            ]
        ],
        "equity": [
            ["2024-01-05", 100000.0],
            ["2024-01-08", 99057.487165],
            ["2024-01-09", 98861.088666],
            ["2024-01-10", 99933.600528],
            ["2024-01-11", 99958.02634],
            ["2024-01-12", 99572.048822],
            ["2024-01-15", 99022.953647],
        ],
    },
    "trailing_stop": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.0,
                "2024-01-11",
                108.0,
                "trail",
                1000.0,
                100000.0,
                108000.0,
                8000.0,
                0.08,
                0.0,
            ]
        ],
        "equity": [
            ["2024-01-08", 100000.0],
            ["2024-01-09", 99702.428953],
            ["2024-01-10", 118000.0],
            ["2024-01-11", 108000.0],
        ],
    },
}


@pytest.mark.parametrize("name", sorted(HISTORICAL_SCENARIOS))
def test_historical_scenarios_are_stable(name):
    cfg, fetcher, _bars = HISTORICAL_SCENARIOS[name]()
    result = run_backtest(cfg, fetcher)
    trades = _trade_tuples(result)
    equity = _equity_tuples(result)
    # Sanity: every scenario must actually produce at least one trade so the
    # characterization is meaningful.
    assert trades, f"scenario {name} produced no trades"
    golden = HISTORICAL_GOLDEN.get(name)
    if golden is not None:
        assert trades == _as_tuples(golden["trades"]), f"{name}: trade drift"
        assert equity == _as_tuples(golden["equity"]), f"{name}: equity drift"


# ── Rolling characterization ─────────────────────────────────────────


def _scn_rolling_partial_exits():
    cfg, fetcher, bars = _scn_partial_exits()
    return cfg, fetcher, bars.index[0].date(), bars.index[14].date()


def _scn_rolling_trailing_stop():
    cfg, fetcher, bars = _scn_trailing_stop()
    return cfg, fetcher, bars.index[0].date(), bars.index[14].date()


def _scn_rolling_daily_refill():
    """Rolling-only: a freed slot is refilled from the same-day candidate scan."""
    active = make_bars(n=30, seed=6, open_base=100.0)
    reserve = make_bars(n=30, seed=7, open_base=50.0)
    spy = make_bars(n=30, seed=8, open_base=400.0)
    active["entry_signal"] = 0.0
    reserve["entry_signal"] = 0.0
    active.iat[5, active.columns.get_loc("entry_signal")] = 1.0
    reserve.iat[7, reserve.columns.get_loc("entry_signal")] = 1.0
    active["volume"] = 1_000_000.0
    reserve["volume"] = 500_000.0
    fetcher = StubPriceFetcher({"ACTIVE": active, "RESERVE": reserve, "SPY": spy})
    cfg = _cfg(
        as_of=active.index[15].date(),
        hold=1,
        top=1,
        tickers=("ACTIVE", "RESERVE"),
    )
    return cfg, fetcher, active.index[0].date(), active.index[15].date()


def _scn_rolling_dividends():
    cfg, fetcher, bars = _scn_dividends()
    return cfg, fetcher, bars.index[0].date(), bars.index[14].date()


ROLLING_SCENARIOS = {
    "partial_exits": _scn_rolling_partial_exits,
    "trailing_stop": _scn_rolling_trailing_stop,
    "daily_refill": _scn_rolling_daily_refill,
    "dividends": _scn_rolling_dividends,
}


ROLLING_GOLDEN: dict[str, dict] = {
    "daily_refill": {
        "trades": [
            [
                "ACTIVE",
                1,
                "2024-01-08",
                "2024-01-09",
                101.2521057,
                "2024-01-10",
                102.32755881,
                "time",
                987.63378117,
                100000.0,
                101062.15382989,
                1062.15382989,
                0.0106215383,
                0.0,
            ],
            [
                "RESERVE",
                1,
                "2024-01-10",
                "2024-01-11",
                49.54464375,
                "2024-01-12",
                48.98830304,
                "time",
                2018.38165392,
                100000.0,
                98877.09211925,
                -1122.90788075,
                -0.0112290788,
                0.0,
            ],
        ],
        "equity": [
            ["2024-01-01", 100000.0],
            ["2024-01-02", 100000.0],
            ["2024-01-03", 100000.0],
            ["2024-01-04", 100000.0],
            ["2024-01-05", 100000.0],
            ["2024-01-08", 100000.0],
            ["2024-01-09", 100322.851747],
            ["2024-01-10", 101062.15383],
            ["2024-01-11", 100565.423526],
            ["2024-01-12", 99939.245949],
            ["2024-01-15", 99939.245949],
            ["2024-01-16", 99939.245949],
            ["2024-01-17", 99939.245949],
            ["2024-01-18", 99939.245949],
            ["2024-01-19", 99939.245949],
            ["2024-01-22", 99939.245949],
        ],
    },
    "dividends": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-05",
                "2024-01-08",
                98.26499225,
                "2024-01-16",
                98.88412561,
                "time",
                1017.6564177,
                100000.0,
                100630.06503841,
                630.06503841,
                0.0063006504,
                1272.07052212,
            ]
        ],
        "equity": [
            ["2024-01-01", 100000.0],
            ["2024-01-02", 100000.0],
            ["2024-01-03", 100000.0],
            ["2024-01-04", 100000.0],
            ["2024-01-05", 100000.0],
            ["2024-01-08", 100333.260208],
            ["2024-01-09", 100761.312555],
            ["2024-01-10", 100278.378513],
            ["2024-01-11", 99990.722495],
            ["2024-01-12", 101405.126147],
            ["2024-01-15", 100887.424812],
            ["2024-01-16", 100630.065038],
            ["2024-01-17", 100630.065038],
            ["2024-01-18", 100630.065038],
            ["2024-01-19", 100630.065038],
        ],
    },
    "partial_exits": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.4612186,
                "2024-01-11",
                105.48427953,
                "target",
                497.70449432,
                50000.0,
                52500.0,
                2500.0,
                0.05,
                0.0,
            ],
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.4612186,
                "2024-01-11",
                99.51833198,
                "stop",
                497.70449432,
                50000.0,
                49530.72109284,
                -469.27890716,
                -0.0093855781,
                0.0,
            ],
        ],
        "equity": [
            ["2024-01-01", 100000.0],
            ["2024-01-02", 100000.0],
            ["2024-01-03", 100000.0],
            ["2024-01-04", 100000.0],
            ["2024-01-05", 100000.0],
            ["2024-01-08", 100000.0],
            ["2024-01-09", 99600.364833],
            ["2024-01-10", 99061.442186],
            ["2024-01-11", 102030.721093],
            ["2024-01-12", 102030.721093],
            ["2024-01-15", 102030.721093],
            ["2024-01-16", 102030.721093],
            ["2024-01-17", 102030.721093],
            ["2024-01-18", 102030.721093],
            ["2024-01-19", 102030.721093],
        ],
    },
    "trailing_stop": {
        "trades": [
            [
                "AAA",
                1,
                "2024-01-08",
                "2024-01-09",
                100.0,
                "2024-01-11",
                108.0,
                "trail",
                1000.0,
                100000.0,
                108000.0,
                8000.0,
                0.08,
                0.0,
            ]
        ],
        "equity": [
            ["2024-01-01", 100000.0],
            ["2024-01-02", 100000.0],
            ["2024-01-03", 100000.0],
            ["2024-01-04", 100000.0],
            ["2024-01-05", 100000.0],
            ["2024-01-08", 100000.0],
            ["2024-01-09", 99702.428953],
            ["2024-01-10", 118000.0],
            ["2024-01-11", 108000.0],
            ["2024-01-12", 108000.0],
            ["2024-01-15", 108000.0],
            ["2024-01-16", 108000.0],
            ["2024-01-17", 108000.0],
            ["2024-01-18", 108000.0],
            ["2024-01-19", 108000.0],
        ],
    },
}


@pytest.mark.parametrize("name", sorted(ROLLING_SCENARIOS))
def test_rolling_scenarios_are_stable(name):
    cfg, fetcher, start, end = ROLLING_SCENARIOS[name]()
    result = run_rolling_backtest(cfg, fetcher, start_date=start, end_date=end)
    trades = _trade_tuples(result)
    equity = _equity_tuples(result)
    assert trades, f"rolling scenario {name} produced no trades"
    golden = ROLLING_GOLDEN.get(name)
    if golden is not None:
        assert trades == _as_tuples(golden["trades"]), f"{name}: trade drift"
        assert equity == _as_tuples(golden["equity"]), f"{name}: equity drift"
