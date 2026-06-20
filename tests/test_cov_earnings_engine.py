"""Line-coverage tests for the earnings-backtest engine, strategies, CLI, and PEAD.

All offline and deterministic: every provider / fetcher / earnings source is
monkeypatched. CLI paths use click's CliRunner.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import pytest
from click.testing import CliRunner

import screener.earnings_backtest.cli as cli_module
import screener.earnings_backtest.engine as engine_module
import screener.earnings_backtest.pead as pead_module
from screener.cli import cli
from screener.earnings_backtest.engine import (
    EarningsTrade,
    _can_use_current_snapshot,
    _find_entry_exit,
    _historical_snapshot_unavailable,
    _resolve_strategies,
    compute_backtest_summary,
    run_earnings_backtest,
)
from screener.earnings_backtest.strategies import (
    SignalResult,
    analyst_sentiment,
    combined_score,
    iv_sentiment,
    price_momentum,
    volume_surge,
)

# Anchor synthetic data to today so the years-cutoff filter never bites.
IDX = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)


def _bars(idx: pd.DatetimeIndex = IDX, drift: float = 1.0) -> pd.DataFrame:
    close = [100.0 + drift * i for i in range(len(idx))]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10_000.0] * len(idx),
        },
        index=idx,
    )


def _events(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _event(ticker: str, ed: date, surprise: float = 10.0) -> dict:
    return {
        "ticker": ticker,
        "earnings_date": ed,
        "eps_estimate": 1.0,
        "reported_eps": 1.1,
        "surprise_pct": surprise,
    }


# ── strategies.py ────────────────────────────────────────────────────────


def test_price_momentum_insufficient_total_bars():
    short = _bars(IDX[:10])
    res = price_momentum("AAA", IDX[5], short)
    assert res.score == 0.0 and res.passed is False
    assert res.details["reason"] == "insufficient_data"


def test_price_momentum_insufficient_signal_bars():
    # 60 bars total (passes first guard) but as_of cuts to < 21 bars.
    res = price_momentum("AAA", IDX[59], _bars(), as_of_date=IDX[10])
    assert res.details["reason"] == "insufficient_data"


def test_price_momentum_scores_positive_drift():
    res = price_momentum("AAA", IDX[59], _bars(drift=1.0), threshold=0.5)
    assert res.score == 1.0 and res.passed is True
    assert "ret_5d" in res.details and "ret_20d" in res.details


def test_volume_surge_insufficient_total_bars():
    res = volume_surge("AAA", IDX[5], _bars(IDX[:10]))
    assert res.details["reason"] == "insufficient_data"


def test_volume_surge_insufficient_signal_bars():
    res = volume_surge("AAA", IDX[59], _bars(), as_of_date=IDX[10])
    assert res.details["reason"] == "insufficient_data"


def test_volume_surge_zero_avg_volume():
    bars = _bars()
    bars["volume"] = 0.0
    res = volume_surge("AAA", IDX[59], bars)
    assert res.details["reason"] == "zero_avg_volume"


def test_volume_surge_surge_and_partial_credit():
    # Spike on last bar -> ratio >= surge_factor -> score path.
    bars = _bars()
    bars.iat[-1, bars.columns.get_loc("volume")] = 100_000.0
    surged = volume_surge("AAA", IDX[59], bars)
    assert surged.score > 0.5

    # Flat volume -> ratio ~1 < surge_factor 1.5 -> partial-credit branch.
    flat = volume_surge("AAA", IDX[59], _bars())
    assert 0.0 < flat.score < 0.5


def test_analyst_sentiment_no_data():
    res = analyst_sentiment("AAA", IDX[59], None)
    assert res.score == 0.0 and res.details["reason"] == "no_data"


def test_analyst_sentiment_positive_net():
    data = {"net": 5, "upgrades": 6, "downgrades": 1}
    res = analyst_sentiment("AAA", IDX[59], data, threshold=0.5)
    assert res.score > 0.5 and bool(res.passed) is True
    assert res.details["net"] == 5


def test_iv_sentiment_no_data_is_neutral():
    res = iv_sentiment("AAA", IDX[59], None, threshold=0.5)
    assert res.score == 0.5
    assert res.passed is True  # threshold <= 0.5
    assert res.details["reason"] == "no_options_data"


def test_iv_sentiment_bullish_low_pc_high_iv():
    res = iv_sentiment("AAA", IDX[59], {"pc_ratio": 0.5, "median_iv": 60.0})
    assert res.score > 0.5
    assert res.details["median_iv_pct"] == 60.0


def test_iv_sentiment_mid_pc_moderate_iv():
    # 0.7 <= pc < 1.0 branch and 20 <= iv < 50 branch.
    res = iv_sentiment("AAA", IDX[59], {"pc_ratio": 0.85, "median_iv": 35.0})
    assert 0.0 < res.score < 1.0


def test_iv_sentiment_bearish_high_pc_low_iv():
    # pc >= 1.0 branch and iv < 20 branch (bearish relative to the bullish case).
    res = iv_sentiment("AAA", IDX[59], {"pc_ratio": 1.5, "median_iv": 10.0})
    bullish = iv_sentiment("AAA", IDX[59], {"pc_ratio": 0.5, "median_iv": 60.0})
    assert res.score < bullish.score


def test_iv_sentiment_nan_iv_uses_neutral():
    res = iv_sentiment("AAA", IDX[59], {"pc_ratio": 0.5, "median_iv": float("nan")})
    assert res.details["median_iv_pct"] is None


def test_iv_sentiment_missing_median_iv_key():
    # median_iv defaults to NaN when absent.
    res = iv_sentiment("AAA", IDX[59], {"pc_ratio": 0.5})
    assert res.details["median_iv_pct"] is None


def test_combined_score_skips_unknown_and_neutral_iv():
    scores = {
        "price_momentum": 1.0,
        "iv_sentiment": 0.5,  # neutral skip
        "not_a_strategy": 0.9,  # not in weights -> skipped
    }
    out = combined_score(scores)
    # Only price_momentum contributes -> weighted avg == its score.
    assert out == 1.0


def test_combined_score_empty_returns_zero():
    assert combined_score({"iv_sentiment": 0.5}) == 0.0


def test_signal_result_is_frozen_dataclass():
    sr = SignalResult("AAA", IDX[0], "x", 0.5, True, {})
    assert sr.ticker == "AAA"


# ── engine.py ────────────────────────────────────────────────────────────


def test_resolve_strategies_variants():
    assert _resolve_strategies("combined_score")[0] == "price_momentum"
    assert _resolve_strategies("volume_surge") == ["volume_surge"]
    with pytest.raises(ValueError, match="Unknown strategy"):
        _resolve_strategies("nope")


def test_can_use_current_snapshot_and_unavailable_helper():
    assert _can_use_current_snapshot(date.today()) is True
    assert _can_use_current_snapshot(date.today() - timedelta(days=1)) is False
    info = _historical_snapshot_unavailable(date(2024, 1, 5))
    assert info["as_of_date"] == "2024-01-05"


def test_find_entry_exit_no_bar_on_or_before():
    bars = _bars()
    early = bars.index[0] - pd.Timedelta(days=10)
    assert _find_entry_exit(bars, early, 1) == (None, None)


def test_find_entry_exit_exit_idx_below_days_before():
    bars = _bars()
    # earnings on the first bar -> exit_idx 0 < days_before -> (None, None)
    assert _find_entry_exit(bars, bars.index[0], 2) == (None, None)


def test_find_entry_exit_happy_path():
    bars = _bars()
    entry, exit_ = _find_entry_exit(bars, bars.index[30], 2)
    assert entry == bars.index[28].date()
    assert exit_ == bars.index[30].date()


def test_run_backtest_no_events_at_all(monkeypatch):
    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: {})
    assert run_earnings_backtest("us", tickers=["AAA"]) == []


def test_run_backtest_loads_universe_when_no_tickers(monkeypatch):
    called = {}

    def fake_universe(market):
        called["market"] = market
        return ["AAA"]

    monkeypatch.setattr(engine_module, "load_universe", fake_universe)
    monkeypatch.setattr(
        engine_module, "collect_earnings_events", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: {})
    run_earnings_backtest("us", tickers=None)
    assert called["market"] == "us"


def test_run_backtest_events_all_outside_window(monkeypatch):
    old = date.today() - timedelta(days=10 * 365)
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", old)]),
    )
    monkeypatch.setattr(engine_module, "fetch_price_data", lambda *a, **kw: {})
    assert run_earnings_backtest("us", tickers=["AAA"]) == []


def test_run_backtest_skips_ticker_without_price_data(monkeypatch):
    ed = IDX[30].date()
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed), _event("ZZZ", ed)]),
    )
    # ZZZ has empty frame -> filtered out by the engine's non-empty keep.
    monkeypatch.setattr(
        engine_module,
        "fetch_price_data",
        lambda *a, **kw: {"AAA": _bars(), "ZZZ": pd.DataFrame()},
    )
    trades = run_earnings_backtest("us", tickers=["AAA", "ZZZ"], min_score=0.0)
    assert [t.ticker for t in trades] == ["AAA"]


def test_run_backtest_skips_event_without_entry_exit(monkeypatch):
    # Earnings on the very first bar -> _find_entry_exit returns (None, None).
    ed = IDX[0].date()
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed)]),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars()}
    )
    assert run_earnings_backtest("us", tickers=["AAA"], days_before=2) == []


def test_run_backtest_single_strategy_in_scores(monkeypatch):
    ed = IDX[30].date()
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed)]),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars()}
    )
    trades = run_earnings_backtest(
        "us", tickers=["AAA"], strategy="price_momentum", min_score=0.0
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.strategy == "price_momentum"
    assert t.score == t.details["scores"]["price_momentum"]


def test_run_backtest_uses_live_snapshots_when_pit_safe(monkeypatch):
    """Force the current-snapshot path (analyst + iv) by allowing today entries."""
    ed = IDX[30].date()
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed)]),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars()}
    )
    # Make every entry date "current" so the live-snapshot branch executes.
    monkeypatch.setattr(engine_module, "_can_use_current_snapshot", lambda d: True)

    analyst_calls = {"n": 0}

    def fake_analyst(ticker, market):
        analyst_calls["n"] += 1
        return {"net": 3, "upgrades": 3, "downgrades": 0}

    def fake_iv(ticker, market):
        return {"pc_ratio": 0.5, "median_iv": 40.0}

    monkeypatch.setattr(engine_module, "fetch_analyst_sentiment", fake_analyst)
    monkeypatch.setattr(engine_module, "fetch_iv_sentiment", fake_iv)

    trades = run_earnings_backtest(
        "us", tickers=["AAA"], strategy="combined_score", min_score=0.0
    )
    assert len(trades) == 1
    scores = trades[0].details["scores"]
    assert set(scores) == {
        "price_momentum",
        "volume_surge",
        "analyst_sentiment",
        "iv_sentiment",
    }
    assert analyst_calls["n"] == 1  # cached: fetched once


def test_run_backtest_single_strategy_skipped_falls_back_to_combined(monkeypatch):
    # strategy is a single (valid) strategy, but for a historical entry the
    # analyst snapshot is skipped -> scores is empty -> strategy not in scores
    # -> the final `else: combined_score(scores)` fallback runs (engine line 211).
    ed = IDX[30].date()
    monkeypatch.setattr(
        engine_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed)]),
    )
    monkeypatch.setattr(
        engine_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars()}
    )
    # Historical entry -> analyst_sentiment branch is skipped (not PIT-safe).
    monkeypatch.setattr(engine_module, "_can_use_current_snapshot", lambda d: False)
    trades = run_earnings_backtest(
        "us", tickers=["AAA"], strategy="analyst_sentiment", min_score=0.0
    )
    assert len(trades) == 1
    assert trades[0].details["scores"] == {}
    assert trades[0].score == 0.0  # combined_score of empty


def test_compute_summary_empty_trades():
    s = compute_backtest_summary([], strategy="x")
    assert s["total_events"] == 0 and s["strategy"] == "x"


def _trade(ret: float, passed: bool = True, entry=None, exit_=None) -> EarningsTrade:
    entry = entry or date(2024, 1, 1)
    exit_ = exit_ or date(2024, 1, 3)
    return EarningsTrade(
        ticker="AAA",
        earnings_date=date(2024, 1, 3),
        entry_date=entry,
        exit_date=exit_,
        entry_price=100.0,
        exit_price=100.0 * (1 + ret / 100),
        return_pct=ret,
        strategy="combined_score",
        score=0.9,
        passed_filter=passed,
    )


def test_compute_summary_no_taken_trades():
    s = compute_backtest_summary([_trade(5.0, passed=False)], strategy="x")
    assert s["total_events"] == 1 and s["trades_taken"] == 0


def test_compute_summary_full_stats_and_sharpe():
    trades = [_trade(float(i - 3)) for i in range(1, 8)]
    s = compute_backtest_summary(trades, strategy="combined_score")
    assert s["trades_taken"] == 7
    assert s["max_winner_pct"] > 0 and s["max_loser_pct"] < 0
    assert isinstance(s["sharpe_approx"], float)
    assert s["profit_factor"] > 0


def test_compute_summary_profit_factor_inf_all_winners():
    s = compute_backtest_summary([_trade(2.0), _trade(3.0)], strategy="x")
    assert s["profit_factor"] == float("inf")


def test_compute_summary_zero_holding_days_branch():
    # entry == exit -> avg_holding == 0 -> sharpe annualization guards hit.
    same = date(2024, 1, 1)
    trades = [
        _trade(1.0, entry=same, exit_=same),
        _trade(-2.0, entry=same, exit_=same),
    ]
    s = compute_backtest_summary(trades, strategy="x")
    assert s["avg_holding_days"] == 0.0


# ── pead.py (remaining gaps: 71, 133, 144, 205-206) ──────────────────────


def test_pead_loads_universe_when_no_tickers(monkeypatch):
    captured = {}

    def fake_universe(market):
        captured["market"] = market
        return ["AAA"]

    monkeypatch.setattr(pead_module, "load_universe", fake_universe)
    monkeypatch.setattr(
        pead_module, "collect_earnings_events", lambda *a, **kw: pd.DataFrame()
    )
    pead_module.run_pead_backtest("us", tickers=None)
    assert captured["market"] == "us"


def test_pead_skips_event_with_no_post_bars(monkeypatch):
    # Earnings on the very last bar -> no bars strictly after -> skipped (line 133).
    ed = IDX[-1].date()
    monkeypatch.setattr(
        pead_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed, 20.0)]),
    )
    monkeypatch.setattr(
        pead_module, "fetch_price_data", lambda *a, **kw: {"AAA": _bars()}
    )
    trades = pead_module.run_pead_backtest(
        "us", tickers=["AAA"], min_surprise=5.0, hold_days=2
    )
    assert trades == []


def test_pead_skips_non_positive_entry_price(monkeypatch):
    ed = IDX[9].date()
    bars = _bars()
    # Force the next-open entry bar to <= 0 (line 144 branch).
    bars.iat[10, bars.columns.get_loc("open")] = 0.0
    monkeypatch.setattr(
        pead_module,
        "collect_earnings_events",
        lambda *a, **kw: _events([_event("AAA", ed, 20.0)]),
    )
    monkeypatch.setattr(pead_module, "fetch_price_data", lambda *a, **kw: {"AAA": bars})
    trades = pead_module.run_pead_backtest(
        "us", tickers=["AAA"], min_surprise=5.0, hold_days=3
    )
    assert trades == []


def test_pead_surprise_quintiles_qcut_value_error(monkeypatch):
    # qcut raising ValueError -> early empty return (lines 205-206).
    def boom(*a, **kw):
        raise ValueError("forced")

    monkeypatch.setattr(pead_module.pd, "qcut", boom)
    trades = [
        pead_module.PeadTrade(
            ticker="AAA",
            earnings_date=date(2024, 1, 3),
            entry_date=date(2024, 1, 4),
            exit_date=date(2024, 1, 10),
            entry_price=100.0,
            exit_price=101.0,
            return_pct=1.0,
            surprise_pct=float(i),
            holding_days=5,
        )
        for i in range(6)
    ]
    assert pead_module.surprise_quintiles(trades) == {}


# ── cli.py (earnings-backtest + earnings-pead rich/csv/no-trade paths) ────


def _patch_backtest(monkeypatch, trades):
    monkeypatch.setattr(cli_module, "run_earnings_backtest", lambda **kw: trades)


def _cli_trade(ret: float, passed: bool = True) -> EarningsTrade:
    return EarningsTrade(
        ticker="AAA",
        earnings_date=date(2024, 1, 3),
        entry_date=date(2024, 1, 2),
        exit_date=date(2024, 1, 3),
        entry_price=100.0,
        exit_price=100.0 * (1 + ret / 100),
        return_pct=ret,
        strategy="combined_score",
        score=0.9,
        passed_filter=passed,
        details={
            "scores": {
                "price_momentum": 1.0,
                "volume_surge": 0.6,
                "analyst_sentiment": 0.7,
                "iv_sentiment": 0.55,
            }
        },
    )


def test_cli_earnings_backtest_no_events(monkeypatch):
    _patch_backtest(monkeypatch, [])
    res = CliRunner().invoke(
        cli, ["earnings-backtest", "--tickers", "AAA"], catch_exceptions=False
    )
    assert res.exit_code == 0
    assert "No earnings events found" in res.output


def test_cli_earnings_backtest_rich_summary_and_table(monkeypatch):
    _patch_backtest(monkeypatch, [_cli_trade(5.0), _cli_trade(-3.0)])
    res = CliRunner().invoke(
        cli,
        ["earnings-backtest", "--tickers", " aaa , ", "--min-score", "0.1"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "Earnings-Drift Backtest Summary" in res.output
    assert "Top Trades" in res.output


def test_cli_earnings_backtest_summary_without_taken(monkeypatch):
    # All trades fail the filter -> summary prints but trade table is skipped.
    _patch_backtest(monkeypatch, [_cli_trade(5.0, passed=False)])
    res = CliRunner().invoke(
        cli, ["earnings-backtest", "--tickers", "AAA"], catch_exceptions=False
    )
    assert res.exit_code == 0
    assert "Earnings-Drift Backtest Summary" in res.output
    assert "Top Trades" not in res.output


def test_cli_earnings_backtest_csv(monkeypatch):
    _patch_backtest(monkeypatch, [_cli_trade(5.0)])
    res = CliRunner().invoke(
        cli,
        ["earnings-backtest", "--tickers", "AAA", "--csv"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    df = pd.read_csv(io.StringIO(res.output))
    assert "price_momentum_score" in df.columns
    assert df.iloc[0]["ticker"] == "AAA"


def test_cli_earnings_pead_quintiles_table(monkeypatch):
    # Enough trades with surprise dispersion -> quintile table renders.
    ed = IDX[9].date()
    events = _events([_event("S%d" % i, ed, float(i * 3 + 6)) for i in range(8)])
    monkeypatch.setattr(pead_module, "collect_earnings_events", lambda *a, **kw: events)
    data = {"S%d" % i: _bars() for i in range(8)}
    monkeypatch.setattr(pead_module, "fetch_price_data", lambda *a, **kw: data)
    res = CliRunner().invoke(
        cli,
        ["earnings-pead", "--tickers", "S0", "--hold-days", "5", "--min-surprise", "5"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "PEAD Backtest Summary" in res.output
    assert "Quintile" in res.output


def test_print_pead_quintiles_empty_is_noop():
    # Directly exercise the early-return on empty quintiles.
    cli_module._print_pead_quintiles({})
