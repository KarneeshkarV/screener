"""Entry-day protection and ambiguous-bar fills through both portfolio engines."""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.slippage import VolumeImpactSlippage
from tests.conftest import StubPriceFetcher


def protection_run(
    engine,
    *,
    spike_day=1,
    order="moo",
    partials=(),
    low=80.0,
    capital=100_000,
    take_profit=None,
    entry_limit_bps=None,
    slippage_bps=0,
    exit_expr=None,
    slippage_model=None,
    volumes=None,
):
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0, 0, 0, 0, 0],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    bars.loc[bars.index[spike_day], ["high", "low"]] = [120.0, low]
    if volumes is not None:
        bars["volume"] = volumes
    cfg = BacktestConfig(
        market="india",
        as_of=date(2024, 1, 1),
        benchmark="^NSEI",
        tickers=("TEST.NS",),
        entry_expr="entry_signal > 0",
        exit_expr=exit_expr,
        hold=2,
        stop_loss=0.1,
        take_profit=take_profit,
        trailing_stop=None,
        top=1,
        initial_capital=capital,
        slippage_bps=slippage_bps,
        slippage_model=slippage_model,
        commission_bps=0,
        entry_order_type=order,
        partial_exits=partials,
        entry_limit_bps=entry_limit_bps,
    )
    fetcher = StubPriceFetcher({"TEST.NS": bars, "^NSEI": bars})
    if engine == "historical":
        return run_backtest(cfg, fetcher)
    return run_rolling_backtest(
        cfg, fetcher, start_date=bars.index[0].date(), end_date=bars.index[-1].date()
    )


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_open_entry_honors_stop_on_entry_day(engine):
    result = protection_run(engine)
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].exit_date == date(2024, 1, 2)
    assert result.trades[0].exit_price == pytest.approx(90)
    assert result.metrics["final_equity"] == pytest.approx(90_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_close_entry_does_not_use_pre_entry_intraday_low(engine):
    result = protection_run(engine, order="moc")
    assert result.trades[0].exit_reason == "time"
    assert result.metrics["final_equity"] == pytest.approx(100_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_stop_precedes_ambiguous_partial_profit_target(engine):
    result = protection_run(engine, spike_day=2, partials=((0.1, 0.5),))
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"
    assert result.metrics["final_equity"] == pytest.approx(90_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_new_breakeven_stop_does_not_reuse_pre_target_low(engine):
    result = protection_run(engine, spike_day=2, partials=((0.1, 0.5),), low=95)
    assert result.trades[0].exit_reason == "target"
    # The original 90 stop did not trigger. The new 100 stop starts next bar.
    assert result.trades[-1].exit_date > date(2024, 1, 3)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_entry_skips_unaffordable_whole_share(engine):
    result = protection_run(engine, capital=99)
    assert not result.trades
    assert result.metrics["final_equity"] == pytest.approx(99)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_partial_exit_keeps_integer_shares_and_cash(engine):
    result = protection_run(engine, capital=350, low=95, partials=((0.1, 0.5),))
    assert [trade.shares for trade in result.trades] == [1, 2]
    assert result.metrics["final_equity"] == pytest.approx(360)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_one_share_partial_does_not_raise_stop_without_a_sale(engine):
    result = protection_run(engine, capital=150, low=95, partials=((0.1, 0.5),))
    assert len(result.trades) == 1
    assert result.trades[0].shares == 1
    assert result.trades[0].exit_reason == "time"


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_entry_day_target_only_for_entry_at_open(engine):
    result = protection_run(engine, take_profit=0.1, low=95)
    assert result.trades[0].exit_reason == "target"
    assert result.trades[0].exit_date == date(2024, 1, 2)
    limit = protection_run(
        engine, order="limit", entry_limit_bps=400, take_profit=0.1, low=95
    )
    assert limit.trades[0].exit_reason == "time"


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_slippage_cannot_create_zero_share_position(engine):
    result = protection_run(engine, capital=100, slippage_bps=10)
    assert not result.trades
    assert result.metrics["final_equity"] == pytest.approx(100)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_close_derived_exit_waits_for_next_open(engine):
    result = protection_run(engine, low=95, exit_expr="high > 110")
    assert result.trades[0].exit_reason == "exit_expr"
    assert result.trades[0].exit_date == date(2024, 1, 3)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_capital_exposure_tracks_partial_sale_and_idle_cash(engine):
    result = protection_run(engine, low=95, capital=350, partials=((0.1, 0.5),))
    # One share sells at 110; two remain at 100 with cash 160.
    assert result.metrics["max_capital_exposure"] == pytest.approx(200 / 360)
    assert result.metrics["avg_capital_exposure"] < result.metrics["exposure"]


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_exit_liquidity_uses_prior_completed_bars(engine):
    @dataclass
    class RecordingSlippage:
        observed: list = field(default_factory=list)

        def adverse_fraction(self, side, shares, adv, sigma_daily, half_spread=0):
            self.observed.append((side, adv, sigma_daily))
            return 0.0

    model = RecordingSlippage()
    protection_run(
        engine,
        low=95,
        slippage_model=model,
        volumes=[1e6, 2e6, 3e6, 999e6, 999e6, 999e6],
    )
    assert model.observed[0] == ("buy", 1_000_000, 0)
    assert model.observed[-1] == ("sell", 2_000_000, 0)


def _slippage_stop_run(
    engine: str,
    bars: pd.DataFrame,
    *,
    as_of: date,
    stop_kind: str,
    stop_pct: float = 0.01,
    slippage_bps: float = 200.0,
    slippage_model=None,
    gap_fills: bool = True,
    hold: int = 2,
):
    cfg = BacktestConfig(
        market="us",
        as_of=as_of,
        benchmark="SPY",
        tickers=("TEST",),
        entry_expr="entry_signal > 0",
        exit_expr=None,
        hold=hold,
        stop_loss=stop_pct if stop_kind == "stop_loss" else None,
        take_profit=None,
        trailing_stop=stop_pct if stop_kind == "trailing_stop" else None,
        top=1,
        initial_capital=100_000.0,
        slippage_bps=slippage_bps,
        slippage_model=slippage_model,
        commission_bps=0.0,
        gap_fills=gap_fills,
        entry_order_type="moo",
    )
    fetcher = StubPriceFetcher({"TEST": bars, "SPY": bars.copy()})
    if engine == "historical":
        return run_backtest(cfg, fetcher)
    return run_rolling_backtest(
        cfg, fetcher, start_date=bars.index[0].date(), end_date=bars.index[-1].date()
    )


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_rally_entry_bar_does_not_false_stop_with_fixed_slippage(engine, stop_kind):
    """Buy slippage pushing entry_fill*(1-stop) >= bar_open must not stop a rally bar."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 110.0, 110.0, 110.0],
            "high": [101.0, 110.0, 110.5, 110.5, 110.5, 110.5],
            "low": [99.0, 100.0, 109.8, 109.8, 109.8, 109.8],
            "close": [100.0, 110.0, 110.0, 110.0, 110.0, 110.0],
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    # 200 bps buy slippage + 1% stop: (1 + 0.02) * (1 - 0.01) = 1.0098 > 1.
    result = _slippage_stop_run(
        engine,
        bars,
        as_of=idx[0].date(),
        stop_kind=stop_kind,
        stop_pct=0.01,
        slippage_bps=200.0,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == idx[1].date()
    assert trade.entry_price == pytest.approx(102.0)
    assert trade.exit_reason == "time"
    assert trade.exit_date == idx[3].date()
    assert trade.exit_price == pytest.approx(110.0 * 0.98)
    assert trade.pnl > 0


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_rally_entry_bar_does_not_false_stop_with_volume_impact(engine, stop_kind):
    """Thin-stock VolumeImpactSlippage pushing stop above open must not stop a rally bar."""
    idx = pd.bdate_range("2024-01-01", periods=8)
    # Trailing bars 0..2 have distinct returns (+16.7%, -4.8%) so sigma_daily is
    # large enough for VolumeImpactSlippage to exceed the 1% stop distance.
    bars = pd.DataFrame(
        {
            "open": [90.0, 105.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0],
            "high": [91.0, 106.0, 101.0, 110.0, 110.5, 110.5, 110.5, 110.5],
            "low": [89.0, 104.0, 99.0, 100.0, 109.8, 109.8, 109.8, 109.8],
            "close": [90.0, 105.0, 100.0, 110.0, 110.0, 110.0, 110.0, 110.0],
            "volume": 500.0,
            "entry_signal": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    result = _slippage_stop_run(
        engine,
        bars,
        as_of=idx[2].date(),
        stop_kind=stop_kind,
        stop_pct=0.01,
        slippage_bps=0.0,
        slippage_model=VolumeImpactSlippage(k=0.2),
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Confirm buy impact actually pushed entry_fill * (1 - 0.01) > bar_open (100.0).
    assert trade.entry_price * (1.0 - 0.01) > 100.0
    assert trade.entry_date == idx[3].date()
    assert trade.exit_reason == "time"
    assert trade.exit_date == idx[5].date()


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_entry_bar_genuine_drop_still_triggers_stop_with_high_slippage(
    engine, stop_kind
):
    """An entry bar whose low drops below entry_ref*(1-stop) must still stop out."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 98.0, 99.5, 99.5, 99.5, 99.5],
            "close": [100.0, 98.5, 100.0, 100.0, 100.0, 100.0],
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    result = _slippage_stop_run(
        engine,
        bars,
        as_of=idx[0].date(),
        stop_kind=stop_kind,
        stop_pct=0.01,
        slippage_bps=200.0,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    expected_reason = "stop" if stop_kind == "stop_loss" else "trail"
    assert trade.exit_reason == expected_reason
    assert trade.exit_date == idx[1].date()
    # Triggered at unslipped level 99.0 (not bar_open 100.0), then 200 bps sell slip.
    assert trade.exit_price == pytest.approx(99.0 * 0.98)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_subsequent_bar_gap_down_below_stop_fills_at_open_with_high_slippage(
    engine, stop_kind
):
    """After surviving the entry bar, a next-bar gap-down below stop fills at open."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 95.0, 95.0, 95.0],
            "high": [101.0, 100.5, 96.0, 96.0, 96.0, 96.0],
            "low": [99.0, 100.0, 94.0, 94.0, 94.0, 94.0],
            "close": [100.0, 100.5, 95.0, 95.0, 95.0, 95.0],
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    result = _slippage_stop_run(
        engine,
        bars,
        as_of=idx[0].date(),
        stop_kind=stop_kind,
        stop_pct=0.01,
        slippage_bps=200.0,
        gap_fills=True,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    expected_reason = "stop" if stop_kind == "stop_loss" else "trail"
    assert trade.exit_reason == expected_reason
    assert trade.exit_date == idx[2].date()
    # Gap-down open (95.0) is below stop_ref (99.0) and trail_ref (99.495) -> open fill.
    assert trade.exit_price == pytest.approx(95.0 * 0.98)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_mild_rally_does_not_false_stop_on_entry_or_next_bar(engine, stop_kind):
    """A small bar-1 gain below entry_fill*(1-stop) must not stop out on bar 2 open."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.5, 105.0, 105.0, 105.0],
            "high": [101.0, 100.5, 105.0, 106.0, 106.0, 106.0],
            "low": [99.0, 100.0, 100.5, 104.5, 104.5, 104.5],
            "close": [100.0, 100.5, 105.0, 105.0, 105.0, 105.0],
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    # entry_fill = 102.0 -> old stop_ref = 100.98 > bar-2 open (100.5).
    result = _slippage_stop_run(
        engine,
        bars,
        as_of=idx[0].date(),
        stop_kind=stop_kind,
        stop_pct=0.01,
        slippage_bps=200.0,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "time"
    assert trade.exit_date == idx[3].date()


@pytest.mark.parametrize("engine", ["historical", "rolling"])
@pytest.mark.parametrize("stop_kind", ["stop_loss", "trailing_stop"])
def test_gap_down_limit_entry_with_slippage_does_not_false_stop(engine, stop_kind):
    """A buy limit that gaps below limit with buy slippage anchors stops to bar_open."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    # Signal bar close = 100.0, entry_limit_bps = 100 -> limit = 99.0.
    # Entry bar opens at 95.0 (< 99.0) and rallies to 105.0; 200 bps buy slip fills at
    # min(95.0 * 1.02, 99.0) = 96.9, where 96.9 * 0.99 = 95.931 > bar_open (95.0).
    bars = pd.DataFrame(
        {
            "open": [100.0, 95.0, 105.0, 105.0, 105.0, 105.0],
            "high": [101.0, 105.0, 105.5, 105.5, 105.5, 105.5],
            "low": [99.0, 95.0, 104.8, 104.8, 104.8, 104.8],
            "close": [100.0, 105.0, 105.0, 105.0, 105.0, 105.0],
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    cfg = BacktestConfig(
        market="us",
        as_of=idx[0].date(),
        benchmark="SPY",
        tickers=("TEST",),
        entry_expr="entry_signal > 0",
        exit_expr=None,
        hold=2,
        stop_loss=0.01 if stop_kind == "stop_loss" else None,
        take_profit=None,
        trailing_stop=0.01 if stop_kind == "trailing_stop" else None,
        top=1,
        initial_capital=100_000.0,
        slippage_bps=200.0,
        commission_bps=0.0,
        gap_fills=True,
        entry_order_type="limit",
        entry_limit_bps=100.0,
    )
    fetcher = StubPriceFetcher({"TEST": bars, "SPY": bars.copy()})
    result = (
        run_backtest(cfg, fetcher)
        if engine == "historical"
        else run_rolling_backtest(
            cfg, fetcher, start_date=idx[0].date(), end_date=idx[-1].date()
        )
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(95.0 * 1.02)
    assert trade.exit_reason == "time"
    assert trade.exit_date == idx[3].date()
