"""Characterization snapshot for ``run_rolling_backtest``.

This test pins the EXACT behavior of the rolling backtester on deterministic
synthetic data so a behavior-preserving refactor of
``screener/backtester/rolling.py`` can be verified byte-for-byte. It asserts the
full trade ledger (tickers, entry/exit dates, fill prices) and the final equity.

The scenario is engineered to exercise every code path that the refactor will
touch:

* four tickers in the universe (AAA, BBB, CCC, DDD) -> at least three trade,
* ``top=2`` with four candidates -> ranking by as-of dollar-volume matters,
* a stop-loss exit (DDD crashes below its 8% stop within the hold window),
* a hold-limit exit (``hold=5`` -> "time" exits dominate),
* an end-of-data force close ("eod") on the final open slot.

If any asserted value changes during the refactor, revert that step entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.book import BOOK_CONFIG_FIELDS
from screener.backtester.models import BacktestConfig
from screener.backtester.price_panel import PRICE_PANEL_CONFIG_FIELDS
from screener.backtester.rolling_simulation import (
    _preparation_fingerprint,
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
    run_rolling_backtest,
)
from screener.backtester.signal_panel import SIGNAL_PANEL_CONFIG_FIELDS
from tests.conftest import StubPriceFetcher

_START = "2024-01-01"
_N = 40
_INDEX = pd.bdate_range(_START, periods=_N)


def _ramp(start_px: float, end_px: float, volume: float) -> pd.DataFrame:
    """A steadily trending OHLCV frame with constant volume."""
    close = pd.Series(np.linspace(start_px, end_px, _N), index=_INDEX, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _crashing_ddd() -> pd.DataFrame:
    """Rises briefly (so it enters as a top-2 candidate) then crashes into its stop."""
    vals = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 80.0, 78.0, 76.0, 74.0]
    vals += list(np.linspace(74.0, 60.0, _N - len(vals)))
    close = pd.Series(vals, index=_INDEX, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(400_000.0, index=_INDEX, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# Distinct dollar-volumes make daily ranking deterministic:
# AAA (highest) > DDD (early) > BBB > CCC.
_DATA = {
    "AAA": _ramp(100.0, 160.0, 500_000.0),
    "BBB": _ramp(100.0, 150.0, 300_000.0),
    "CCC": _ramp(100.0, 150.0, 200_000.0),
    "DDD": _crashing_ddd(),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}

# Captured against the UNMODIFIED rolling.py. Each tuple is
# (ticker, signal_date, entry_date, entry_price, exit_date, exit_price, reason).
_EXPECTED_LEDGER = [
    ("DDD", "2024-01-04", "2024-01-05", 103.000000, "2024-01-09", 94.760000, "stop"),
    ("AAA", "2024-01-04", "2024-01-05", 104.615385, "2024-01-12", 113.846154, "time"),
    ("BBB", "2024-01-09", "2024-01-10", 107.692308, "2024-01-17", 115.384615, "time"),
    ("AAA", "2024-01-12", "2024-01-15", 113.846154, "2024-01-22", 123.076923, "time"),
    ("BBB", "2024-01-17", "2024-01-18", 115.384615, "2024-01-25", 123.076923, "time"),
    ("AAA", "2024-01-22", "2024-01-23", 123.076923, "2024-01-30", 132.307692, "time"),
    ("BBB", "2024-01-25", "2024-01-26", 123.076923, "2024-02-02", 130.769231, "time"),
    ("AAA", "2024-01-30", "2024-01-31", 132.307692, "2024-02-07", 141.538462, "time"),
    ("BBB", "2024-02-02", "2024-02-05", 130.769231, "2024-02-12", 138.461538, "time"),
    ("AAA", "2024-02-07", "2024-02-08", 141.538462, "2024-02-15", 150.769231, "time"),
    ("BBB", "2024-02-12", "2024-02-13", 138.461538, "2024-02-20", 146.153846, "time"),
    ("AAA", "2024-02-15", "2024-02-16", 150.769231, "2024-02-23", 160.000000, "time"),
    ("BBB", "2024-02-20", "2024-02-21", 146.153846, "2024-02-23", 150.000000, "eod"),
]
_EXPECTED_FINAL_EQUITY = 134805.07624907081
_EXPECTED_UNIQUE_TICKERS = 3


def _cfg() -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=5,
        top=2,  # top < 4 candidates -> ranking is load-bearing
        strategy_name=None,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=0.08,  # DDD's crash trips this
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB", "CCC", "DDD"),
    )


def test_rolling_backtest_ledger_snapshot():
    fetcher = StubPriceFetcher(_DATA)
    cfg = _cfg()
    result = run_rolling_backtest(
        cfg,
        fetcher,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )

    actual = [
        (
            t.ticker,
            t.signal_date.isoformat(),
            t.entry_date.isoformat(),
            t.entry_price,
            t.exit_date.isoformat(),
            t.exit_price,
            str(t.exit_reason),
        )
        for t in result.trades
    ]

    assert len(actual) == len(_EXPECTED_LEDGER)
    for got, want in zip(actual, _EXPECTED_LEDGER):
        (g_tk, g_sig, g_entry, g_epx, g_exit, g_xpx, g_reason) = got
        (w_tk, w_sig, w_entry, w_epx, w_exit, w_xpx, w_reason) = want
        assert g_tk == w_tk
        assert g_sig == w_sig
        assert g_entry == w_entry
        assert g_exit == w_exit
        assert g_reason == w_reason
        assert g_epx == pytest.approx(w_epx)
        assert g_xpx == pytest.approx(w_xpx)

    assert float(result.equity_curve.iloc[-1]) == pytest.approx(_EXPECTED_FINAL_EQUITY)
    assert result.metrics["unique_tickers"] == _EXPECTED_UNIQUE_TICKERS


def test_rolling_backtest_exercises_all_exit_reasons():
    """Guard rails: the snapshot must keep exercising ranking, stop, hold, eod."""
    fetcher = StubPriceFetcher(_DATA)
    result = run_rolling_backtest(
        _cfg(),
        fetcher,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    reasons = {str(t.exit_reason) for t in result.trades}
    assert {"stop", "time", "eod"} <= reasons


def test_prepared_backtest_reuses_signals_for_runtime_only_changes():
    fetcher = StubPriceFetcher(_DATA)
    cfg = _cfg()
    prepared = prepare_rolling_backtest(
        cfg,
        fetcher,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    changed = cfg.model_copy(update={"hold": 8, "top": 3, "initial_capital": 125_000.0})

    reused = run_prepared_rolling_backtest(prepared, changed)
    fresh = run_rolling_backtest(
        changed,
        StubPriceFetcher(_DATA),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )

    assert reused.trades == fresh.trades
    pd.testing.assert_series_equal(reused.equity_curve, fresh.equity_curve)
    assert reused.metrics == fresh.metrics
    pd.testing.assert_frame_equal(reused.selection, fresh.selection)


def test_prepared_backtest_rejects_signal_changes():
    cfg = _cfg()
    prepared = prepare_rolling_backtest(
        cfg,
        StubPriceFetcher(_DATA),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    changed = cfg.model_copy(update={"entry_expr": "close > sma(close, 10)"})

    with pytest.raises(ValueError, match="prepared-data fields"):
        run_prepared_rolling_backtest(prepared, changed)


# ---------------------------------------------------------------------------
# The reuse fingerprint must not have a silent failure mode.
# ---------------------------------------------------------------------------

# The hand-maintained allowlist this stage deleted, kept verbatim so the test
# below can show what it would have done with a config field nobody added to
# it. Do not extend it: it exists only as the "before" side of the comparison.
_LEGACY_PREPARATION_CONFIG_FIELDS = {
    "market",
    "as_of",
    "benchmark",
    "tickers",
    "universe_file",
    "membership_added",
    "membership_windows",
    "dynamic_universe_size",
    "dynamic_universe_lookback",
    "dynamic_universe_rebalance",
    "max_universe",
    "entry_expr",
    "exit_expr",
    "strategy_name",
    "regime_filter",
    "earnings_blackout_days",
    "fundamentals_provider",
    "fundamental_fields",
    "fundamental_lag_days",
    "sector_neutral",
    "interval",
    "price_adjustment",
    "min_price",
    "min_avg_dollar_volume",
    "avg_dollar_volume_window",
}


class _TruncatingConfig(BacktestConfig):
    """A config carrying a bars-affecting knob added after the allowlist was written.

    ``truncate_leading_bars`` stands in for the next data setting somebody adds
    (an extended-hours toggle, a stale-bar policy, another adjustment regime):
    it changes which bars the fetcher returns, exactly as ``interval`` and
    ``price_adjustment`` already do, and nobody remembered to register it
    anywhere.
    """

    truncate_leading_bars: int = 0


class _TruncatingStubFetcher:
    """Stub fetcher configured from the config, as the real fetchers are."""

    def __init__(self, data: dict[str, pd.DataFrame], cfg: _TruncatingConfig) -> None:
        self._inner = StubPriceFetcher(data)
        self._truncate = cfg.truncate_leading_bars

    def fetch(self, tickers, start, end) -> dict[str, pd.DataFrame]:
        panel = self._inner.fetch(tickers, start, end)
        return {t: frame.iloc[self._truncate :] for t, frame in panel.items()}


def _truncating_cfg(truncate_leading_bars: int) -> _TruncatingConfig:
    return _TruncatingConfig(
        **_cfg().model_dump(exclude={"slippage_model"}),
        truncate_leading_bars=truncate_leading_bars,
    )


def test_unregistered_bars_affecting_field_invalidates_reuse():
    """A config field the fingerprint was never told about must still block reuse.

    This is the hole the derived fingerprint closes. Under the old allowlist the
    sweep would have reused panels built from different bars and reported wrong
    numbers with no error.
    """
    base = _truncating_cfg(0)
    changed = base.model_copy(update={"truncate_leading_bars": 5})
    window = {"start_date": _INDEX[0].date(), "end_date": _INDEX[-1].date()}

    prepared = prepare_rolling_backtest(
        base, _TruncatingStubFetcher(_DATA, base), **window
    )
    rebuilt = prepare_rolling_backtest(
        changed, _TruncatingStubFetcher(_DATA, changed), **window
    )

    # The field really does change the bars, so reusing ``prepared`` for
    # ``changed`` would be a stale-panel bug and not merely a missed cache hit.
    assert not prepared.bars_by_tv["AAA"].equals(rebuilt.bars_by_tv["AAA"])

    # The deleted allowlist could not see the field, so it declared the two
    # configs interchangeable.
    assert "truncate_leading_bars" not in _LEGACY_PREPARATION_CONFIG_FIELDS
    assert base.model_dump(
        mode="json", include=_LEGACY_PREPARATION_CONFIG_FIELDS
    ) == changed.model_dump(mode="json", include=_LEGACY_PREPARATION_CONFIG_FIELDS)

    # The derived fingerprint claims every field no module declared as
    # runtime-only, so the unregistered field lands in the key by default.
    assert "truncate_leading_bars" in prepared.config_fingerprint
    assert not prepared.supports(changed)
    with pytest.raises(ValueError, match="prepared-data fields"):
        run_prepared_rolling_backtest(prepared, changed)

    # Rebuilt panels still run, so the guard costs a rebuild, not a failure.
    assert run_prepared_rolling_backtest(rebuilt, changed).trades


def test_every_config_field_is_claimed_by_a_knowledge_module():
    """Each config field belongs to a panel or to the book; nothing is unowned.

    An unowned field is not a correctness bug (the fingerprint treats unknown
    fields as panel-affecting), but it does mean a sweep silently stops reusing
    prepared data, so make the classification explicit instead.
    """
    claimed = (
        PRICE_PANEL_CONFIG_FIELDS | SIGNAL_PANEL_CONFIG_FIELDS | BOOK_CONFIG_FIELDS
    )
    assert set(BacktestConfig.model_fields) == claimed


def test_book_fields_are_the_only_ones_left_out_of_the_reuse_key():
    """Runtime-only settings stay out of the key; everything else stays in."""
    fingerprint = _preparation_fingerprint(_cfg())
    panel_fields = PRICE_PANEL_CONFIG_FIELDS | SIGNAL_PANEL_CONFIG_FIELDS

    assert panel_fields <= set(fingerprint)
    assert set(fingerprint) & (BOOK_CONFIG_FIELDS - panel_fields) == set()
