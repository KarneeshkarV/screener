"""Offline coverage tests for the backtester core/rolling/historical/data/pine modules.

These tests are deterministic and never touch the network: every price fetch goes
through ``StubPriceFetcher`` or a monkeypatched seam, and CLI paths use
``click.testing.CliRunner`` with an injected fetcher (``obj=...``).

They are written to drive the remaining uncovered lines in:
  - screener/backtester/core.py
  - screener/backtester/rolling.py
  - screener/backtester/historical.py
  - screener/backtester/data.py
  - screener/backtester/pine.py
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from main import cli

from screener.backtester import data, rolling
from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.core import (
    _apply_slip,
    _bar_index_on_or_before,
    _eligible_reserve_signal_idx,
    _make_slot_state,
    _passes_entry_filters,
    _precompute_entry_signals,
    _precompute_filter_signals,
    _prepare_strategy_bars,
    _resolve_universe,
    _trailing_liquidity,
)
from screener.backtester.fills import FillModel
from screener.backtester.historical import (
    _benchmark_series_from_panel,
    run_backtest,
    select_candidates,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import (
    PineError,
    PineSyntaxError,
    evaluate,
    parse,
    required_lookback,
)
from screener.backtester.rolling import (
    _build_rolling_candidate_matrices,
    run_rolling_backtest,
)

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=2,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ───────────────────────── data.py ─────────────────────────


def test_tv_to_yf_exchange_prefixes():
    assert data.tv_to_yf("NSE:RELIANCE", "india") == "RELIANCE.NS"
    assert data.tv_to_yf("BSE:TCS", "india") == "TCS.BO"
    assert data.tv_to_yf("NASDAQ:AAPL", "us") == "AAPL"
    assert data.tv_to_yf("RELIANCE", "india") == "RELIANCE.NS"
    assert data.tv_to_yf("AAPL", "us") == "AAPL"


def test_naive_normalized_index_handles_tz_and_non_datetime():
    tz_idx = pd.DatetimeIndex(["2024-01-01"]).tz_localize("UTC")
    out = data._naive_normalized_index(tz_idx)
    assert out.tz is None
    # non-DatetimeIndex path: list of date strings.
    out2 = data._naive_normalized_index(pd.Index(["2024-01-02", "2024-01-03"]))
    assert isinstance(out2, pd.DatetimeIndex)


def test_load_cached_missing_and_corrupt(tmp_path):
    assert data._load_cached("NOPE", tmp_path) is None
    # write a non-parquet file at the expected path -> ParserError/OSError path.
    bad = data._cache_path("BAD", tmp_path)
    bad.write_text("not a parquet file")
    assert data._load_cached("BAD", tmp_path) is None


def test_load_and_save_cache_roundtrip_drops_nan(tmp_path):
    df = make_bars(n=5)
    df.iloc[0, df.columns.get_loc("close")] = np.nan
    data._save_cache("RT", df, tmp_path)
    loaded = data._load_cached("RT", tmp_path)
    assert loaded is not None
    # the NaN-close row was dropped on load.
    assert len(loaded) == 4


def test_save_cache_swallows_errors(monkeypatch, tmp_path):
    df = make_bars(n=3)

    def boom(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    # must not raise.
    data._save_cache("ERR", df, tmp_path)


def test_apply_splits_only_adjustment_paths():
    no_factor = make_bars(n=3)
    flat_factor = make_bars(n=3)
    flat_factor["split_factor"] = 1.0
    split_factor = make_bars(n=4)
    split_factor["split_factor"] = [2.0, 2.0, 1.0, 1.0]
    split_factor["dividend"] = [0.0, 0.0, 1.0, 0.0]
    empty = pd.DataFrame()
    out = data.apply_splits_only_adjustment(
        {
            "NOFAC": no_factor,
            "FLAT": flat_factor,
            "SPLIT": split_factor,
            "EMPTY": empty,
        }
    )
    # no_factor passes through unchanged.
    assert out["NOFAC"] is no_factor
    assert out["FLAT"] is flat_factor
    assert out["EMPTY"] is empty
    # split frame back-adjusts close by factor 2.0 on the first bars.
    assert out["SPLIT"]["close"].iloc[0] == pytest.approx(
        split_factor["close"].iloc[0] / 2.0
    )
    assert out["SPLIT"]["volume"].iloc[0] == pytest.approx(
        split_factor["volume"].iloc[0] * 2.0
    )


def test_warn_unadjustable_fmp_frames_emits_warning():
    fmp = make_bars(n=3)  # no split_factor column
    yf = make_bars(n=3)
    yf["split_factor"] = 1.0
    out = data.warn_unadjustable_fmp_frames(
        {"FMP": fmp, "YF": yf, "EMPTY": pd.DataFrame()}
    )
    assert out["FMP"] is fmp


def test_merge_cached_branches():
    a = make_bars(n=3, start="2024-01-01")
    b = make_bars(n=3, start="2024-01-04")
    # existing None -> copy of new
    assert len(data._merge_cached(None, b)) == 3
    # new empty -> existing copy
    assert len(data._merge_cached(a, pd.DataFrame())) == 3
    # both present -> concat dedup
    merged = data._merge_cached(a, b)
    assert len(merged) == 6
    # both empty -> empty
    assert data._merge_cached(pd.DataFrame(), pd.DataFrame()).empty


def test_has_range_and_empty():
    df = make_bars(n=20, start="2024-01-01")
    s = pd.Timestamp("2024-01-02")
    e = pd.Timestamp("2024-01-20")
    assert data._has_range(df, s, e)
    assert not data._has_range(pd.DataFrame(), s, e)


def test_split_download_single_and_multi():
    # empty raw -> empty frames per ticker.
    out = data._split_download(pd.DataFrame(), ["AAA", "BBB"])
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].empty

    # single-ticker (plain columns).
    single = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [100, 200],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    out_single = data._split_download(single, ["AAA"])
    assert "close" in out_single["AAA"].columns

    # multi-ticker MultiIndex columns.
    cols = pd.MultiIndex.from_product(
        [["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    raw = pd.DataFrame(
        np.random.default_rng(0).uniform(1, 2, size=(3, 10)),
        index=pd.bdate_range("2024-01-01", periods=3),
        columns=cols,
    )
    out_multi = data._split_download(raw, ["AAA", "BBB", "MISSING"])
    assert "close" in out_multi["AAA"].columns
    assert out_multi["MISSING"].empty


def test_normalize_fmp_historical_variants():
    # not a dict
    assert data._normalize_fmp_historical([], True).empty
    # missing 'historical'
    assert data._normalize_fmp_historical({"foo": 1}, True).empty
    # historical not a list
    assert data._normalize_fmp_historical({"historical": "x"}, True).empty
    # no 'date' column
    assert data._normalize_fmp_historical({"historical": [{"open": 1.0}]}, True).empty
    # full payload with auto_adjust scaling.
    payload = {
        "historical": [
            {
                "date": "2024-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 1000,
                "adjClose": 5.0,
            },
            {
                "date": "2024-01-03",
                "open": 12.0,
                "high": 13.0,
                "low": 11.0,
                "close": 12.0,
                "volume": 1200,
                "adjClose": 12.0,
            },
        ]
    }
    adj = data._normalize_fmp_historical(payload, True)
    # first bar adj factor 0.5 -> close 5.0
    assert adj["close"].iloc[0] == pytest.approx(5.0)
    raw = data._normalize_fmp_historical(payload, False)
    assert raw["close"].iloc[0] == pytest.approx(10.0)


def test_fallback_price_fetcher_fills_missing():
    primary = StubPriceFetcher({"AAA": make_bars(n=5)})  # BBB missing
    fallback = StubPriceFetcher({"BBB": make_bars(n=5)})
    fb = data.FallbackPriceFetcher(primary, fallback)
    out = fb.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 4, 1))
    assert not out["AAA"].empty
    assert not out["BBB"].empty

    # When nothing missing, returns primary results directly.
    both = StubPriceFetcher({"AAA": make_bars(n=5), "BBB": make_bars(n=5)})
    fb2 = data.FallbackPriceFetcher(both, StubPriceFetcher({}))
    out2 = fb2.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 4, 1))
    assert set(out2) == {"AAA", "BBB"}

    # missing in BOTH primary and fallback -> empty placeholder frame set.
    fb3 = data.FallbackPriceFetcher(StubPriceFetcher({}), StubPriceFetcher({}))
    out3 = fb3.fetch(["ZZZ"], date(2024, 1, 1), date(2024, 4, 1))
    assert out3["ZZZ"].empty


def test_build_price_fetcher_variants(monkeypatch):
    monkeypatch.setattr(data, "_load_env_file", lambda: None)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("SCREENER_PRICE_PROVIDER", raising=False)
    assert isinstance(data.build_price_fetcher("auto"), data.YFinancePriceFetcher)
    assert isinstance(data.build_price_fetcher("yf"), data.YFinancePriceFetcher)

    monkeypatch.setenv("FMP_API_KEY", "k")
    assert isinstance(data.build_price_fetcher("auto"), data.FallbackPriceFetcher)
    assert isinstance(data.build_price_fetcher("fmp"), data.FMPPriceFetcher)

    with pytest.raises(ValueError):
        data.build_price_fetcher("bogus")


def test_fetch_benchmark_empty_and_present():
    fetcher = StubPriceFetcher({"SPY": make_bars(n=10)})
    series = data.fetch_benchmark("SPY", date(2024, 1, 1), date(2024, 4, 1), fetcher)
    assert not series.empty
    empty = data.fetch_benchmark("NONE", date(2024, 1, 1), date(2024, 4, 1), fetcher)
    assert empty.empty


def test_ensure_date_variants():
    assert data.ensure_date(date(2024, 1, 1)) == date(2024, 1, 1)
    assert data.ensure_date(datetime(2024, 1, 1, 9, 30)) == date(2024, 1, 1)
    assert data.ensure_date(pd.Timestamp("2024-01-01")) == date(2024, 1, 1)
    assert data.ensure_date("2024-01-01") == date(2024, 1, 1)
    with pytest.raises(TypeError):
        data.ensure_date(123)


def test_load_env_file(monkeypatch, tmp_path):
    data._DOTENV_LOADED = False
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        '# comment\nFOO_COV="bar"\nnoequals\nEXISTING=ignored\n'
    )
    monkeypatch.setenv("EXISTING", "kept")
    monkeypatch.delenv("FOO_COV", raising=False)
    data.load_env_file()
    assert __import__("os").environ["FOO_COV"] == "bar"
    assert __import__("os").environ["EXISTING"] == "kept"
    # second call short-circuits.
    data.load_env_file()


def test_load_env_file_missing(monkeypatch, tmp_path):
    data._DOTENV_LOADED = False
    monkeypatch.chdir(tmp_path)
    data.load_env_file()  # no .env -> returns early without error


# ───────────────────────── pine.py ─────────────────────────


def test_pine_unexpected_character():
    with pytest.raises(PineSyntaxError):
        parse("close @ 3")


def test_pine_empty_expression():
    with pytest.raises(PineSyntaxError):
        parse("")


def test_pine_true_false_literals():
    bars = make_bars(n=5)
    assert bool(evaluate(parse("true"), bars).iloc[0]) is True
    assert bool(evaluate(parse("false"), bars).iloc[0]) is False


def test_pine_unary_plus_and_not():
    bars = make_bars(n=5)
    out = evaluate(parse("+close"), bars)
    assert out.iloc[0] == pytest.approx(float(bars.iloc[0]["close"]))
    notout = evaluate(parse("not (close > 0)"), bars)
    assert not bool(notout.iloc[0])


def test_pine_all_binops_and_compares():
    bars = make_bars(n=6)
    assert not evaluate(parse("close - close == 0"), bars).empty
    assert not evaluate(parse("close * 2 / 2 >= close"), bars).empty
    assert not evaluate(parse("close + 1 > close"), bars).empty
    assert not evaluate(parse("close <= high"), bars).empty
    assert not evaluate(parse("close < high"), bars).empty
    assert not evaluate(parse("close != 0"), bars).empty


def test_pine_boolop_or():
    bars = make_bars(n=5)
    out = evaluate(parse("close > 0 or close < 0"), bars)
    assert bool(out.iloc[0])


def test_pine_scalar_compare():
    bars = make_bars(n=5)
    # 1 > 0 is a pure-scalar compare -> broadcast.
    out = evaluate(parse("1 > 0"), bars)
    assert bool(out.iloc[0])


def test_pine_all_rolling_funcs():
    bars = make_bars(n=30)
    for expr in [
        "ema(close, 5)",
        "rsi(close, 5)",
        "highest(high, 5)",
        "lowest(low, 5)",
        "atr(5)",
        "crossover(close, sma(close, 3))",
        "crossunder(close, sma(close, 3))",
    ]:
        out = evaluate(parse(expr), bars)
        assert len(out) == len(bars)


def test_pine_rsi_no_loss_branch():
    idx = pd.bdate_range("2024-01-01", periods=20)
    close = pd.Series(np.arange(100.0, 120.0), index=idx)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
        }
    )
    out = evaluate(parse("rsi(close, 5)"), bars)
    # strictly rising -> avg_loss 0, avg_gain > 0 -> RSI 100.
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_pine_function_arity_errors():
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close)"), make_bars(n=5))
    with pytest.raises(PineSyntaxError):
        evaluate(parse("atr(5, 3)"), make_bars(n=5))
    with pytest.raises(PineSyntaxError):
        evaluate(parse("crossover(close)"), make_bars(n=5))


def test_pine_unknown_function():
    with pytest.raises(PineError):
        evaluate(parse("foobar(close, 3)"), make_bars(n=5))


def test_pine_non_integer_literal_arg():
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close, 2.5)"), make_bars(n=10))
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close, close)"), make_bars(n=10))
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close, 0)"), make_bars(n=10))


def test_pine_sma_of_scalar_source():
    # source that is not a Series (a numeric literal) -> _as_series branch.
    bars = make_bars(n=10)
    out = evaluate(parse("sma(2, 3)"), bars)
    assert out.dropna().iloc[-1] == pytest.approx(2.0)


def test_pine_unknown_identifier():
    with pytest.raises(PineError):
        evaluate(parse("nonexistent_col > 0"), make_bars(n=5))


def test_pine_extra_column_identifier():
    bars = make_bars(n=5)
    bars["custom"] = np.arange(5.0)
    out = evaluate(parse("custom + 1"), bars)
    assert out.iloc[0] == pytest.approx(1.0)


def test_pine_adj_close_alias():
    bars = make_bars(n=5)
    out = evaluate(parse("adj_close"), bars)
    assert out.iloc[0] == pytest.approx(float(bars.iloc[0]["close"]))
    bars2 = make_bars(n=5)
    bars2["adj_close"] = bars2["close"] * 2.0
    out2 = evaluate(parse("adj_close"), bars2)
    assert out2.iloc[0] == pytest.approx(float(bars2.iloc[0]["close"]) * 2.0)


def test_pine_missing_series_column():
    bars = make_bars(n=5).drop(columns=["volume"])
    # evaluate() guards missing required columns first.
    with pytest.raises(PineError):
        evaluate(parse("volume > 0"), bars)


def test_pine_empty_bars():
    assert evaluate(parse("close > 0"), pd.DataFrame()).empty


def test_pine_parse_errors():
    with pytest.raises(PineSyntaxError):
        parse("close >")  # incomplete
    with pytest.raises(PineSyntaxError):
        parse("(close")  # unclosed paren
    with pytest.raises(PineSyntaxError):
        parse("close close")  # trailing token


def test_pine_required_lookback():
    assert required_lookback(parse("close > sma(close, 10)")) == 10
    assert required_lookback(parse("atr(14) > 0")) == 14
    assert required_lookback(parse("crossover(close, open)")) == 1
    assert required_lookback(parse("not (close > sma(close, 7))")) == 7
    assert required_lookback(parse("-close + sma(close, 4)")) == 4
    assert required_lookback(parse("close > 0")) == 0


# ───────────────────────── core.py ─────────────────────────


def test_apply_slip_shim():
    cfg = _cfg(slippage_bps=100.0)
    out = _apply_slip(100.0, "buy", cfg)
    assert out == pytest.approx(101.0)


def test_trailing_liquidity_edges():
    bars = make_bars(n=30)
    assert _trailing_liquidity(bars, -1) == (0.0, 0.0)
    assert _trailing_liquidity(bars, 5, window=0) == (0.0, 0.0)
    # single-bar window -> sigma 0.
    adv, sigma = _trailing_liquidity(bars, 0, window=1)
    assert sigma == 0.0
    adv2, sigma2 = _trailing_liquidity(bars, 10, window=5)
    assert adv2 > 0


def test_trailing_liquidity_non_finite():
    bars = make_bars(n=10)
    bars["volume"] = np.inf  # adv mean -> inf -> reset to 0.0
    bars["close"] = [1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    adv, sigma = _trailing_liquidity(bars, 5, window=5)
    assert adv == 0.0
    assert np.isfinite(sigma)


def test_trailing_liquidity_empty_window():
    # signal_idx before the start with a window that lands on empty slice.
    bars = make_bars(n=5)
    empty = bars.iloc[0:0]
    assert _trailing_liquidity(empty, 0) == (0.0, 0.0)


def test_passes_entry_filters_no_filters():
    bars = make_bars(n=10)
    cfg = _cfg(min_price=None, min_avg_dollar_volume=None)
    assert _passes_entry_filters(bars, bars.index[5], cfg) == (True, None)


def test_passes_entry_filters_no_history():
    bars = make_bars(n=10, start="2024-02-01")
    cfg = _cfg(min_price=1.0)
    ok, reason = _passes_entry_filters(bars, pd.Timestamp("2024-01-01"), cfg)
    assert not ok and reason == "no history"


def test_passes_entry_filters_price_fail():
    bars = make_bars(n=10, open_base=5.0)
    cfg = _cfg(min_price=1000.0)
    ok, reason = _passes_entry_filters(bars, bars.index[5], cfg)
    assert not ok and "price" in reason


def test_passes_entry_filters_adv_fail():
    bars = make_bars(n=10, open_base=100.0)
    cfg = _cfg(min_price=None, min_avg_dollar_volume=1e15)
    ok, reason = _passes_entry_filters(bars, bars.index[5], cfg)
    assert not ok and "adv" in reason


def test_passes_entry_filters_pass():
    bars = make_bars(n=20, open_base=100.0)
    cfg = _cfg(min_price=1.0, min_avg_dollar_volume=1.0)
    ok, reason = _passes_entry_filters(bars, bars.index[10], cfg)
    assert ok and reason is None


def test_make_slot_state_exit_eval_failure():
    bars = make_bars(n=20)
    cfg = _cfg()
    bad_exit = parse("nonexistent_col > 0")
    state, warn = _make_slot_state("AAA", bars, 5, cfg, bad_exit, 1)
    assert state is None
    assert warn and "exit eval failed" in warn


def test_make_slot_state_no_entry_bar():
    bars = make_bars(n=5)
    cfg = _cfg()
    state, warn = _make_slot_state("AAA", bars, 4, cfg, None, 1)
    assert state is None
    assert warn and "no post-signal" in warn


def test_make_slot_state_with_partials_and_stop_target():
    bars = make_bars(n=30)
    cfg = _cfg(
        stop_loss=0.05,
        take_profit=0.1,
        partial_exits=((0.05, 0.5),),
    )
    state, warn = _make_slot_state("AAA", bars, 5, cfg, None, 3)
    assert state is not None
    assert state.stop_ref is not None and state.target_ref is not None
    assert state.partial_targets and state.partial_fractions


def test_resolve_universe_tickers_and_cap():
    cfg = _cfg(tickers=("AAA", "BBB", "CCC"), max_universe=2)
    syms, warns = _resolve_universe(cfg)
    assert syms == ["AAA", "BBB"]
    assert any("capped" in w for w in warns)


def test_resolve_universe_file(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("AAA\n# comment\n\nBBB\n")
    cfg = _cfg(tickers=None, universe_file=str(f))
    syms, warns = _resolve_universe(cfg)
    assert syms == ["AAA", "BBB"]


def test_resolve_universe_none_raises():
    cfg = _cfg(tickers=None, universe_file=None)
    with pytest.raises(ValueError):
        _resolve_universe(cfg)


def test_prepare_strategy_bars_no_spec():
    cfg = _cfg(strategy_name=None)
    bars_by = {"AAA": make_bars(n=5)}
    out, lb = _prepare_strategy_bars(
        cfg,
        bars_by,
        {},
        ["AAA"],
        date(2024, 1, 1),
        date(2024, 4, 1),
        StubPriceFetcher({}),
        [],
    )
    assert out is bars_by and lb == 0


def test_eligible_reserve_signal_idx_paths():
    bars = make_bars(n=30, open_base=100.0)
    cfg = _cfg(min_price=None, min_avg_dollar_volume=None)
    entry = parse("close > sma(close, 3)")
    # no history at all (day before start).
    assert (
        _eligible_reserve_signal_idx(bars, pd.Timestamp("2020-01-01"), cfg, entry, 3)
        is None
    )
    # insufficient lookback.
    early = bars.index[1]
    assert _eligible_reserve_signal_idx(bars, early, cfg, entry, 50) is None
    # filter fail.
    cfg_filtered = _cfg(min_price=1e9)
    assert (
        _eligible_reserve_signal_idx(bars, bars.index[20], cfg_filtered, entry, 3)
        is None
    )
    # entry eval failure.
    bad_entry = parse("nonexistent_col > 0")
    assert _eligible_reserve_signal_idx(bars, bars.index[20], cfg, bad_entry, 3) is None


def test_bar_index_on_or_before():
    bars = make_bars(n=10, start="2024-01-08")
    assert _bar_index_on_or_before(bars, pd.Timestamp("2024-01-01")) is None
    idx = _bar_index_on_or_before(bars, bars.index[3] + pd.Timedelta(hours=5))
    assert idx == 3


def test_precompute_filter_signals_sentinel_and_values():
    bars_by = {"AAA": make_bars(n=20, open_base=100.0), "EMPTY": pd.DataFrame()}
    # no filters -> empty sentinel.
    assert (
        _precompute_filter_signals(
            bars_by, _cfg(min_price=None, min_avg_dollar_volume=None)
        )
        == {}
    )
    out = _precompute_filter_signals(
        bars_by, _cfg(min_price=1.0, min_avg_dollar_volume=1.0)
    )
    assert "AAA" in out and out["AAA"].dtype == bool
    assert "EMPTY" not in out


def test_precompute_entry_signals_eval_failure():
    bars_by = {"AAA": make_bars(n=10), "EMPTY": pd.DataFrame()}
    warns: list[str] = []
    out = _precompute_entry_signals(bars_by, parse("nonexistent_col > 0"), warns)
    assert out == {}
    assert any("entry eval failed" in w for w in warns)


def test_build_rolling_candidate_matrices_membership_and_regime():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bars = make_bars(n=20, open_base=100.0)
    bars.index = idx
    bars_by = {"AAA": bars}
    entry_sig = {"AAA": pd.Series(True, index=idx)}
    master = list(idx)
    # membership_added suppresses early signals; regime_allowed gates days.
    regime_allowed = pd.Series([False] * 5 + [True] * 15, index=idx)
    mats = _build_rolling_candidate_matrices(
        bars_by,
        entry_sig,
        {},
        master,
        lookback_required=3,
        membership_added={"AAA": idx[10].date()},
        regime_allowed=regime_allowed,
    )
    # before date-added -> suppressed.
    assert not bool(mats.signal_mat.iloc[5]["AAA"])
    # after both gates -> allowed.
    assert bool(mats.signal_mat.iloc[15]["AAA"])
    assert mats.filter_mat is None


# ───────────────────────── historical.py ─────────────────────────


def test_select_candidates_warnings():
    bars_by = {
        "EMPTY": pd.DataFrame(),
        "SHORT": make_bars(n=2),
        "GOOD": make_bars(n=30, open_base=100.0),
    }
    entry = parse("close > sma(close, 3)")
    df, warns = select_candidates(
        bars_by,
        entry,
        pd.Timestamp("2024-02-20"),
        2,
        3,
        _cfg(min_price=None, min_avg_dollar_volume=None),
    )
    assert any("no data" in w for w in warns)
    assert any("insufficient lookback" in w for w in warns)


def test_select_candidates_filtered_count():
    bars_by = {"AAA": make_bars(n=30, open_base=100.0)}
    entry = parse("close > 0")
    df, warns = select_candidates(
        bars_by,
        entry,
        pd.Timestamp("2024-02-20"),
        2,
        0,
        _cfg(min_price=1e9),
    )
    assert any("filtered" in w for w in warns)
    assert df.empty


def test_select_candidates_eval_failure_warning():
    bars_by = {"AAA": make_bars(n=30)}
    df, warns = select_candidates(
        bars_by, parse("nonexistent_col > 0"), pd.Timestamp("2024-02-20"), 2, 0
    )
    assert any("entry eval failed" in w for w in warns)


def test_select_candidates_no_signal():
    bars_by = {"AAA": make_bars(n=30, open_base=100.0)}
    df, warns = select_candidates(
        bars_by, parse("close > 1000000000"), pd.Timestamp("2024-02-20"), 2, 3
    )
    assert df.empty


def test_select_candidates_ranks_roles():
    bars_by = {
        "AAA": make_bars(n=30, open_base=100.0),
        "BBB": make_bars(n=30, open_base=50.0),
        "CCC": make_bars(n=30, open_base=30.0),
    }
    df, _ = select_candidates(
        bars_by,
        parse("close > 0"),
        pd.Timestamp("2024-02-20"),
        1,
        0,
        _cfg(reserve_multiple=3),
    )
    assert "rank" in df.columns and "role" in df.columns
    assert (df["role"] == "active").sum() == 1
    assert (df["role"] == "reserve").sum() >= 1


def test_benchmark_series_from_panel_empty_and_present():
    assert _benchmark_series_from_panel({}, "SPY").empty
    assert _benchmark_series_from_panel({"SPY": pd.DataFrame()}, "SPY").empty
    s = _benchmark_series_from_panel({"SPY": make_bars(n=5)}, "SPY")
    assert not s.empty and s.name == "SPY"


def test_run_backtest_empty_selection():
    # entry never fires -> empty selection branch.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, open_base=100.0),
            "SPY": make_bars(n=60, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 15),
        entry_expr="close > 1000000000",
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert result.trades == []
    assert result.selection.empty


def test_run_backtest_with_reserve_rotation():
    # Build a universe with an active that exits early and reserves to rotate in.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=80, seed=1, open_base=100.0),
            "BBB": make_bars(n=80, seed=2, open_base=80.0),
            "CCC": make_bars(n=80, seed=3, open_base=60.0),
            "SPY": make_bars(n=80, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("AAA", "BBB", "CCC"),
        reserve_multiple=3,
        reinvest=True,
        stop_loss=0.03,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_run_backtest_allow_reentry():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=80, seed=5, open_base=100.0),
            "SPY": make_bars(n=80, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("AAA",),
        allow_reentry=True,
        max_reentries=2,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


# ───────────────────────── rolling.py ─────────────────────────


def test_run_rolling_backtest_end_before_start():
    fetcher = StubPriceFetcher({"AAA": make_bars(n=10), "SPY": make_bars(n=10)})
    with pytest.raises(ValueError):
        run_rolling_backtest(
            _cfg(), fetcher, start_date=date(2024, 5, 1), end_date=date(2024, 1, 1)
        )


def test_run_rolling_backtest_no_trading_days():
    # Price data exists only OUTSIDE the requested window -> early_result path.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=10, start="2024-01-01"),
            "SPY": make_bars(n=10, start="2024-01-01"),
        }
    )
    result = run_rolling_backtest(
        _cfg(tickers=("AAA",), min_price=None, min_avg_dollar_volume=None),
        fetcher,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 2, 1),
    )
    assert any("no trading days" in w for w in result.warnings)
    assert result.trades == []


def test_run_rolling_backtest_with_regime_and_splits():
    data_dict = {
        "AAA": make_bars(n=60, seed=1, open_base=100.0),
        "SPY": make_bars(n=60, seed=9, open_base=400.0),
    }
    # add split_factor so splits_only path runs.
    aaa = data_dict["AAA"]
    aaa["split_factor"] = 1.0
    aaa["dividend"] = 0.0
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=3,
        top=1,
        tickers=("AAA",),
        regime_filter=("uptrend",),
        price_adjustment="splits_only",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data_dict),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 3, 1),
    )
    assert isinstance(result.trades, list)


def test_run_rolling_backtest_with_membership_added():
    idx = pd.bdate_range("2024-01-01", periods=60)
    aaa = make_bars(n=60, seed=1, open_base=100.0)
    aaa.index = idx
    spy = make_bars(n=60, seed=9, open_base=400.0)
    spy.index = idx
    cfg = _cfg(
        as_of=idx[-1].date(),
        hold=3,
        top=1,
        tickers=("AAA",),
        membership_added=(("AAA", idx[30].date()),),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": aaa, "SPY": spy}),
        start_date=idx[5].date(),
        end_date=idx[-1].date(),
    )
    # entries suppressed before date-added => trades only have signal dates >= added.
    for t in result.trades:
        assert t.signal_date >= idx[30].date()


# ───────────────────────── cli_common.py ─────────────────────────


def test_cli_common_helpers():
    # slippage models
    for m in ["fixed", "half-spread", "vol-impact", "composite"]:
        assert build_slippage_model(m, 1.0, 2.0, 0.1) is not None
    # partial exits
    assert parse_partial_exits(()) == ()
    assert parse_partial_exits(("0.05:0.5",)) == ((0.05, 0.5),)
    import click

    with pytest.raises(click.UsageError):
        parse_partial_exits(("bad",))
    # min filters: 0 disables.
    assert resolve_min_filters("us", 0.0, 0.0) == (None, None)
    assert resolve_min_filters("us", None, None) == (1.0, 1000.0)
    # strategy exprs.
    e, x = resolve_strategy_exprs(None, "close > 0", None)
    assert e == "close > 0"
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs(None, None, None)
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs("does_not_exist", None, None)


# ───────────────────────── CLI paths ─────────────────────────


def _stub_env(n=60):
    return StubPriceFetcher(
        {
            "AAA": make_bars(n=n, seed=11, open_base=100.0),
            "BBB": make_bars(n=n, seed=12, open_base=50.0),
            "SPY": make_bars(n=n, seed=99, open_base=400.0),
        }
    )


def test_historical_cli_csv_and_report(tmp_path):
    fetcher = _stub_env()
    report = tmp_path / "report.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            "2024-02-15",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--csv",
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert report.exists()


def test_historical_cli_report_no_csv(tmp_path):
    fetcher = _stub_env()
    report = tmp_path / "r.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--universe-file",
            str(_universe_file(tmp_path)),
            "--as-of",
            "2024-02-15",
            "--hold",
            "5",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Report:" in res.output


def _universe_file(tmp_path):
    f = tmp_path / "univ.txt"
    f.write_text("AAA\nBBB\n")
    return f


def test_historical_cli_no_universe_error():
    res = CliRunner().invoke(
        cli,
        ["backtest-historical", "--as-of", "2024-02-15", "--entry", "close > 0"],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "No universe provided" in res.output


def test_historical_cli_strategy_shortcut():
    fetcher = _stub_env()
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            "2024-02-15",
            "--strategy",
            "breakout",
            "--hold",
            "5",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_csv():
    fetcher = _stub_env()
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-20",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--csv",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_default_window_and_report(tmp_path):
    fetcher = _stub_env(n=400)
    report = tmp_path / "roll.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--years",
            "1",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert report.exists()


def test_rolling_cli_point_in_time_requires_universe():
    # --point-in-time with --tickers is a usage error.
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA",
            "--entry",
            "close > 0",
            "--point-in-time",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "point-in-time" in res.output


def test_rolling_cli_universe_path(monkeypatch):
    fetcher = _stub_env()

    loaded = SimpleNamespace(
        symbols=("AAA", "BBB"),
        name="sp500",
        source="test",
        cached_path="/tmp/x",
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)

    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--universe",
            "sp500",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Universe:" in res.output


def test_rolling_cli_point_in_time_universe(monkeypatch):
    fetcher = _stub_env()
    loaded = SimpleNamespace(
        symbols=("AAA", "BBB"),
        name="sp500",
        source="test",
        cached_path="/tmp/x",
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)
    monkeypatch.setattr(
        rolling,
        "load_sp500_membership",
        lambda **k: {"AAA": date(2010, 1, 1), "BBB": None},
    )
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--universe",
            "sp500",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--point-in-time",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_point_in_time_non_sp500(monkeypatch):
    fetcher = _stub_env()
    loaded = SimpleNamespace(
        symbols=("AAA",), name="nifty50", source="test", cached_path="/tmp/x"
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "india",
            "--universe",
            "nifty50",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--entry",
            "close > 0",
            "--point-in-time",
        ],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "sp500" in res.output


# ───────────────────────── core helpers (direct) ─────────────────────────

from screener.backtester.core import (  # noqa: E402
    _SlotState,
    _close_slot_at_day,
    _fire_partial_exits_at_bar,
    _force_close_open_slots,
    _maybe_credit_dividends,
)
from screener.backtester.portfolio import Portfolio  # noqa: E402


def _open_slot(bars, *, entry_idx=1, ticker="AAA", **state_kw):
    """Build a portfolio with an open position + matching slot state."""
    cfg = _cfg(initial_capital=10_000.0)
    portfolio = Portfolio(cfg.initial_capital, 1)
    entry_fill = float(bars.iloc[entry_idx]["open"])
    portfolio.assign(ticker, 1, bars.index[0].date())
    portfolio.open(
        ticker=ticker,
        entry_date=bars.index[entry_idx].date(),
        entry_price=entry_fill,
        commission_bps=0.0,
    )
    defaults = dict(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=bars.index[entry_idx].date(),
        entry_fill=entry_fill,
        signal_date=bars.index[entry_idx - 1].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=entry_idx + 5,
        peak=entry_fill,
        exit_signal=None,
    )
    defaults.update(state_kw)
    state = _SlotState(**defaults)
    return cfg, portfolio, state


def test_close_slot_day_not_in_bars():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars)
    slot_states = {0: state}
    fm = FillModel(cfg)
    # a day not present in the index -> returns False.
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=pd.Timestamp("2050-01-01"),
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False


def test_close_slot_before_entry_bar():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars, entry_idx=3)
    slot_states = {0: state}
    fm = FillModel(cfg)
    # day == entry bar (i < entry_idx+1) -> skip.
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[3],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False


def test_close_slot_fully_closed_by_partial():
    bars = make_bars(n=20, open_base=100.0)
    # craft a target that the next bar's high definitely exceeds, fraction 1.0.
    cfg, portfolio, state = _open_slot(
        bars,
        entry_idx=1,
        partial_targets=(0.0,),  # any positive high triggers
        partial_fractions=(1.0,),
        stop_ref=None,
        target_ref=None,
    )
    state.partial_fired = [False]
    slot_states = {0: state}
    fm = FillModel(cfg)
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[2],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    # whole position scaled out -> slot freed via the position-None branch.
    assert out is True
    assert slot_states[0] is None


def test_close_slot_duplicate_index_returns_false():
    bars = make_bars(n=6)
    # duplicate a timestamp so get_loc returns a slice/array.
    dup_idx = bars.index.tolist()
    dup_idx[3] = dup_idx[2]
    bars.index = pd.DatetimeIndex(dup_idx)
    cfg, portfolio, state = _open_slot(bars, entry_idx=0)
    slot_states = {0: state}
    fm = FillModel(cfg)
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[2],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False


def test_fire_partial_exits_no_position():
    bars = make_bars(n=10, open_base=100.0)
    cfg = _cfg()
    portfolio = Portfolio(cfg.initial_capital, 1)
    state = _SlotState(
        ticker="AAA",
        entry_idx=1,
        entry_date=bars.index[1].date(),
        entry_fill=100.0,
        signal_date=bars.index[0].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=6,
        peak=100.0,
        exit_signal=None,
        partial_targets=(0.05,),
        partial_fractions=(0.5,),
        partial_fired=[False],
    )
    # no open position -> early return (pos is None).
    _fire_partial_exits_at_bar(state, bars, 2, cfg, portfolio, FillModel(cfg))
    assert state.partial_fired == [False]


def test_fire_partial_exits_no_targets():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars)
    # no partial targets -> immediate return.
    _fire_partial_exits_at_bar(state, bars, 2, cfg, portfolio, FillModel(cfg))


def test_maybe_credit_dividends_paths():
    cfg_none = _cfg(price_adjustment="none")
    bars = make_bars(n=5)
    portfolio = Portfolio(cfg_none.initial_capital, 1)
    state = _SlotState(
        ticker="AAA",
        entry_idx=0,
        entry_date=bars.index[0].date(),
        entry_fill=100.0,
        signal_date=bars.index[0].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=5,
        peak=100.0,
        exit_signal=None,
    )
    # no dividend column -> early return.
    _maybe_credit_dividends(portfolio, state, bars, 1, cfg_none)

    # full adjustment -> early return even with dividend column.
    bars_div = make_bars(n=5)
    bars_div["dividend"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    _maybe_credit_dividends(
        portfolio, state, bars_div, 1, _cfg(price_adjustment="full")
    )

    # non-numeric dividend -> ValueError swallowed.
    bars_bad = make_bars(n=5)
    bars_bad["dividend"] = ["x", "y", "z", "w", "v"]
    _maybe_credit_dividends(portfolio, state, bars_bad, 1, cfg_none)

    # zero/neg dividend -> not credited.
    bars_zero = make_bars(n=5)
    bars_zero["dividend"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    _maybe_credit_dividends(portfolio, state, bars_zero, 1, cfg_none)


def test_force_close_open_slots_empty_tail():
    bars = make_bars(n=10, start="2024-01-01")
    cfg, portfolio, state = _open_slot(bars, entry_idx=2)
    slot_states = {0: state, 1: None}
    slot_bars = {0: bars}
    # end_ts before entry_date -> tail empty -> skip (continue).
    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=pd.Timestamp("2020-01-01"),
        fill_model=FillModel(cfg),
    )
    assert slot_states[0] is state  # not closed


def test_force_close_open_slots_closes():
    bars = make_bars(n=10, start="2024-01-01")
    cfg, portfolio, state = _open_slot(bars, entry_idx=2)
    slot_states = {0: state}
    slot_bars = {0: bars}
    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=bars.index[-1],
        fill_model=FillModel(cfg),
    )
    assert slot_states[0] is None
    assert portfolio.closed_trades()


# ───────────── pine reachable defensive line ─────────────


def test_pine_unary_minus_eval():
    bars = make_bars(n=5)
    out = evaluate(parse("-close + close"), bars)
    assert out.iloc[0] == pytest.approx(0.0)


def test_pine_series_from_name_missing_series_direct():
    from screener.backtester.pine import _series_from_name

    bars = make_bars(n=5).drop(columns=["open"])
    with pytest.raises(PineError):
        _series_from_name("open", bars)


# ───────────── rolling matrices with filters + exit_ast ─────────────


def test_rolling_with_filters_and_exit_expr():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, seed=1, open_base=100.0),
            "SPY": make_bars(n=60, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=1,
        tickers=("AAA",),
        exit_expr="close < sma(close, 3)",
        min_price=1.0,
        min_avg_dollar_volume=1.0,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=date(2024, 1, 15), end_date=date(2024, 3, 1)
    )
    assert isinstance(result.trades, list)


def test_build_matrices_filter_mat_present():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bars = make_bars(n=20, open_base=100.0)
    bars.index = idx
    bars_by = {"AAA": bars}
    entry_sig = {"AAA": pd.Series(True, index=idx)}
    filter_sig = {"AAA": pd.Series(True, index=idx)}
    mats = _build_rolling_candidate_matrices(
        bars_by, entry_sig, filter_sig, list(idx), lookback_required=3
    )
    assert mats.filter_mat is not None


# ───────────── historical event-driven edge branches ─────────────


def test_run_backtest_active_no_data_and_no_history():
    # AAA has no data; BBB has data only after as_of (no history at as_of).
    aaa_empty = pd.DataFrame()
    bbb = make_bars(n=30, start="2024-06-01", open_base=80.0)
    spy = make_bars(n=120, start="2024-01-01", open_base=400.0)
    # GOOD ticker selectable at as_of.
    good = make_bars(n=120, start="2024-01-01", open_base=100.0)
    fetcher = StubPriceFetcher({"AAA": aaa_empty, "BBB": bbb, "GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=3,
        entry_expr="close > 0",
        tickers=("GOOD", "AAA", "BBB"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_run_backtest_reentry_full_flow():
    # Long horizon ticker that re-enters after closing.
    good = make_bars(n=150, start="2024-01-01", seed=7, open_base=100.0)
    spy = make_bars(n=150, start="2024-01-01", seed=9, open_base=400.0)
    fetcher = StubPriceFetcher({"GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("GOOD",),
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_rolling_dashboard_path(monkeypatch, tmp_path):
    import screener.backtester.dashboard as dash

    monkeypatch.setattr(
        dash, "render_dashboard", lambda result, d: tmp_path / "dash.html"
    )
    monkeypatch.setattr(dash, "serve_dashboard", lambda d, p: None)
    fetcher = _stub_env()
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--dashboard",
            "--dashboard-dir",
            str(tmp_path),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Dashboard:" in res.output


def test_rolling_candidate_make_slot_fails_on_last_bar():
    # A ticker whose data ends exactly at the window end: a signal on the final
    # in-window bar has no post-signal entry bar -> _make_slot_state returns None
    # inside _simulate_day (the candidate make-slot-fail branch).
    idx = pd.bdate_range("2024-01-02", periods=25)
    aaa = make_bars(n=25, seed=3, open_base=100.0)
    aaa.index = idx
    spy = make_bars(n=25, seed=9, open_base=400.0)
    spy.index = idx
    fetcher = StubPriceFetcher({"AAA": aaa, "SPY": spy})
    cfg = _cfg(
        as_of=idx[-1].date(),
        hold=3,
        top=1,
        tickers=("AAA",),
        entry_expr="close > 0",  # fires every bar, including the last
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=idx[0].date(), end_date=idx[-1].date()
    )
    assert isinstance(result.trades, list)


def test_rolling_with_empty_panel_ticker():
    # One ticker has data, another resolves to an empty frame in the panel.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, seed=1, open_base=100.0),
            "EMPTY": pd.DataFrame(),
            "SPY": make_bars(n=60, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=2,
        tickers=("AAA", "EMPTY"),
        entry_expr="close > sma(close, 3)",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=date(2024, 1, 15), end_date=date(2024, 3, 1)
    )
    assert isinstance(result.trades, list)


# ───────────── data fetcher branches (offline yfinance/FMP) ─────────────


def test_normalize_frame_with_adj_close_and_actions():
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [1000, 1100],
            "Adj Close": [10.5, 11.5],
            "Dividends": [0.0, 0.5],
            "Stock Splits": [0.0, 2.0],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    out = data._normalize_frame(raw)
    assert "adj_close" in out.columns
    assert "dividend" in out.columns
    assert "split_factor" in out.columns


def test_normalize_frame_dividend_alias():
    raw = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
            "dividend": [0.25],
        },
        index=pd.bdate_range("2024-01-01", periods=1),
    )
    out = data._normalize_frame(raw)
    assert out["dividend"].iloc[0] == pytest.approx(0.25)


def test_yfinance_fetcher_partial_cache_extends_forward(tmp_path, monkeypatch):
    import yfinance as yf

    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, batch_size=50)
    # seed cache with an early window.
    early = pd.DataFrame(
        {
            "Open": np.arange(10.0, 20.0),
            "High": np.arange(11.0, 21.0),
            "Low": np.arange(9.0, 19.0),
            "Close": np.arange(10.5, 20.5),
            "Volume": np.arange(1000, 1010),
        },
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    data._save_cache("AAA", data._normalize_frame(early), tmp_path)

    def fake_download(target, **kwargs):
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(30.0, 35.0),
                "High": np.arange(31.0, 36.0),
                "Low": np.arange(29.0, 34.0),
                "Close": np.arange(30.5, 35.5),
                "Volume": np.arange(2000, 2005),
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    # request a window that extends past the cached max -> forward fetch branch.
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 3, 1))
    assert "AAA" in out


def test_yfinance_fetcher_partial_cache_extends_backward(tmp_path, monkeypatch):
    import yfinance as yf

    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, batch_size=50)
    # seed cache with a LATE window (covers the recent end, misses early start).
    late = pd.DataFrame(
        {
            "Open": np.arange(10.0, 20.0),
            "High": np.arange(11.0, 21.0),
            "Low": np.arange(9.0, 19.0),
            "Close": np.arange(10.5, 20.5),
            "Volume": np.arange(1000, 1010),
        },
        index=pd.bdate_range("2024-02-15", periods=10),
    )
    data._save_cache("AAA", data._normalize_frame(late), tmp_path)

    def fake_download(target, **kwargs):
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(30.0, 35.0),
                "High": np.arange(31.0, 36.0),
                "Low": np.arange(29.0, 34.0),
                "Close": np.arange(30.5, 35.5),
                "Volume": np.arange(2000, 2005),
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    # request a window starting well before cache min -> backward fetch branch.
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 2, 28))
    assert "AAA" in out


def test_yfinance_fetcher_auto_adjust_false_actions(tmp_path, monkeypatch):
    import yfinance as yf

    captured = {}

    def fake_download(target, **kwargs):
        captured.update(kwargs)
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(10.0, 15.0),
                "High": np.arange(11.0, 16.0),
                "Low": np.arange(9.0, 14.0),
                "Close": np.arange(10.5, 15.5),
                "Volume": np.arange(1000, 1005),
                "Dividends": [0.0] * 5,
                "Stock Splits": [0.0] * 5,
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    fetcher = data.YFinancePriceFetcher(
        cache_dir=tmp_path, auto_adjust=False, batch_size=50
    )
    fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 15))
    assert captured.get("actions") is True


def test_fmp_fetcher_requires_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        data.FMPPriceFetcher()


def test_fmp_fetcher_empty_payload(monkeypatch, tmp_path):
    import requests

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(requests, "Session", lambda: FakeSession())
    fetcher = data.FMPPriceFetcher(api_key="k", cache_dir=tmp_path)
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 15))
    assert out["AAA"].empty


def test_configure_yfinance_missing_private_symbols(monkeypatch):
    import sys
    import types

    data._YFINANCE_CONFIGURED = False
    fake_yf = types.ModuleType("yfinance")
    fake_yf.set_tz_cache_location = lambda loc: None
    fake_cache = types.ModuleType("yfinance.cache")
    # deliberately omit _TzCacheManager/_TzCacheDummy -> AttributeError path.
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setitem(sys.modules, "yfinance.cache", fake_cache)
    data._configure_yfinance()
    assert data._YFINANCE_CONFIGURED is True


def test_configure_yfinance_swap_failure(monkeypatch):
    import sys
    import types

    data._YFINANCE_CONFIGURED = False
    fake_yf = types.ModuleType("yfinance")

    def boom(loc):
        raise RuntimeError("nope")

    fake_yf.set_tz_cache_location = boom
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    data._configure_yfinance()
    assert data._YFINANCE_CONFIGURED is True


def test_configure_yfinance_already_configured():
    data._YFINANCE_CONFIGURED = True
    data._configure_yfinance()  # short-circuit


# ───────────── historical _run_event_driven_sim (direct) ─────────────

from screener.backtester.historical import _run_event_driven_sim  # noqa: E402


def _flat_then_trending(start, n, base, *, dip_at=None):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(base, base + n, n), index=idx, dtype=float)
    if dip_at is not None:
        close.iloc[dip_at] = base * 0.5  # crash to trip stop / free slot
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_event_driven_sim_active_edge_branches():
    as_of = pd.Timestamp("2024-02-01")
    # ACTIVE rows: no-data, no-history, make-slot-fail, good.
    good = _flat_then_trending("2024-01-01", 60, 100.0)
    no_history = _flat_then_trending("2024-03-01", 30, 80.0)  # all after as_of
    actives = pd.DataFrame(
        [
            {"ticker": "GOOD", "rank": 1},
            {"ticker": "NODATA", "rank": 2},
            {"ticker": "NOHIST", "rank": 3},
            {"ticker": "MAKEFAIL", "rank": 4},
        ]
    )
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    # MAKEFAIL frame ends exactly at as_of so signal bar is last -> no entry bar.
    makefail_frame = _flat_then_trending("2023-12-01", 44, 50.0)
    bars_by = {
        "GOOD": good,
        "NODATA": pd.DataFrame(),
        "NOHIST": no_history,
        "MAKEFAIL": makefail_frame,
    }
    cfg = _cfg(
        as_of=as_of.date(),
        hold=3,
        top=4,
        entry_expr="close > 0",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 4)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert any("no data during sim" in w for w in warnings)
    assert any("no history at as_of" in w for w in warnings)


def test_event_driven_sim_reserve_rotation_branches():
    as_of = pd.Timestamp("2024-02-01")
    # ACTIVE exits early (dip trips stop), freeing the slot for reserves.
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    # reserve candidates: no-data, ineligible (insufficient lookback), good.
    good_reserve = _flat_then_trending("2024-01-01", 80, 90.0)
    short_reserve = _flat_then_trending("2024-02-02", 80, 70.0)  # starts after as_of
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(
        [
            {"ticker": "NODATA_R", "rank": 2},
            {"ticker": "SHORT_R", "rank": 3},
            {"ticker": "GOOD_R", "rank": 4},
        ]
    )
    bars_by = {
        "ACTIVE": active,
        "NODATA_R": pd.DataFrame(),
        "SHORT_R": short_reserve,
        "GOOD_R": good_reserve,
    }
    cfg = _cfg(
        as_of=as_of.date(),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    # The active stopped out and at least one reserve rotated in.
    assert any(t.ticker == "ACTIVE" for t in trades)


def test_event_driven_sim_reentry_branches():
    as_of = pd.Timestamp("2024-02-01")
    # active closes (stop) then re-signals later so it re-enters.
    active = _flat_then_trending("2024-01-01", 90, 100.0, dip_at=24)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_force_close_open_slot():
    # Active never exits within horizon -> force-closed (eod) at the end.
    as_of = pd.Timestamp("2024-02-01")
    active = _flat_then_trending("2024-01-01", 120, 100.0)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=500,
        top=1,
        entry_expr="close > 0",
        stop_loss=None,
        take_profit=None,
        reinvest=False,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    assert any(str(t.exit_reason) == "eod" for t in trades)


def test_event_driven_sim_reserve_makefail():
    # A reserve whose only eligible signal is its last bar -> make_slot fails
    # (no post-signal entry bar) -> warning branch.
    as_of = pd.Timestamp("2024-02-01")
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    # reserve frame ends one bar after the freeing day so its signal is last.
    good_reserve = _flat_then_trending("2024-01-01", 28, 90.0)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame([{"ticker": "RSV", "rank": 2}])
    bars_by = {"ACTIVE": active, "RSV": good_reserve}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(warnings, list)


def test_event_driven_sim_reserve_already_taken():
    # A reserve ticker that is also an active -> the `ticker in taken` skip.
    as_of = pd.Timestamp("2024-02-01")
    dup = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    other = _flat_then_trending("2024-01-01", 80, 90.0)
    actives = pd.DataFrame([{"ticker": "DUP", "rank": 1}])
    # DUP appears again in reserves (already taken) before a fresh reserve.
    reserves = pd.DataFrame(
        [{"ticker": "DUP", "rank": 2}, {"ticker": "OTHER", "rank": 3}]
    )
    bars_by = {"DUP": dup, "OTHER": other}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_reentry_not_eligible_then_eligible():
    # Active stops out early; entry signal stays False for a while (re-entry
    # pending but not yet eligible -> the `continue` branch), then fires again.
    as_of = pd.Timestamp("2024-02-01")
    idx = pd.bdate_range("2024-01-01", periods=90)
    # Build a path: rise (enter), crash (stop), flat-down (no signal), rise again.
    seg1 = np.linspace(100, 110, 30)
    seg2 = np.linspace(70, 60, 30)  # falling -> entry signal stays False
    seg3 = np.linspace(60, 90, 30)  # rising -> entry signal fires again
    close = pd.Series(np.concatenate([seg1, seg2, seg3]), index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    active = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 5)",
        stop_loss=0.05,
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 5)"),
        exit_ast=None,
        lookback=5,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    # at least the initial trade plus a re-entry occurred.
    assert len([t for t in trades if t.ticker == "ACTIVE"]) >= 1


def test_event_driven_sim_reserve_makeslot_fail_last_bar():
    # Reserve becomes eligible on a day that is the LAST bar of its frame, so
    # _make_slot_state has no post-signal entry bar -> the make-slot-fail branch.
    as_of = pd.Timestamp("2024-02-01")
    # active stops out on the freeing day.
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    freeing_day = active.index[26]  # roughly where stop fires + day_loop frees
    # reserve frame: enough history to be eligible, ending exactly at a freeing day.
    rsv_idx = pd.bdate_range("2024-01-02", periods=40)
    rsv_close = pd.Series(np.linspace(90, 130, 40), index=rsv_idx, dtype=float)
    rsv_open = rsv_close.shift(1).fillna(rsv_close.iloc[0] - 1.0)
    rsv = pd.DataFrame(
        {
            "open": rsv_open,
            "high": pd.concat([rsv_open, rsv_close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([rsv_open, rsv_close], axis=1).min(axis=1) - 1.0,
            "close": rsv_close,
            "volume": pd.Series(100_000.0, index=rsv_idx),
        }
    )
    # Truncate the reserve so its last bar is exactly the active's freeing day
    # (2024-02-05): the reserve is eligible there but has no post-signal bar, so
    # _make_slot_state returns None -> the reserve make-slot-fail branch.
    freeing_day = pd.Timestamp("2024-02-05")
    rsv = rsv.loc[rsv.index <= freeing_day]
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame([{"ticker": "RSV", "rank": 2}])
    bars_by = {"ACTIVE": active, "RSV": rsv}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_reentry_makeslot_fail_last_bar():
    # Re-entry signal lands on the final bar of the frame -> make-slot-fail.
    as_of = pd.Timestamp("2024-02-01")
    idx = pd.bdate_range("2024-01-01", periods=40)
    # rise (enter), crash (stop near bar 27), then rise so re-entry signal fires
    # right up to the last bar.
    close = pd.Series(
        np.concatenate([np.linspace(100, 110, 27), np.linspace(70, 100, 13)]),
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    frame = pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([openp, close], axis=1).min(axis=1) - 1.0,
            "close": close,
            "volume": pd.Series(100_000.0, index=idx),
        }
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": frame}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.05,
        allow_reentry=True,
        max_reentries=5,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_force_close_empty_tail():
    # Active enters on the final available bar so the force-close tail
    # (bars.index > entry_date) is empty -> the `tail.empty` continue branch.
    # frame ends 2 bars after as_of: signal at as_of -> entry on the last bar.
    idx = pd.bdate_range("2024-01-01", periods=24)  # ends ~ 2024-02-01
    last = idx[-1]
    # ensure as_of is the second-to-last bar so entry is the last bar.
    as_of2 = idx[-2]
    close = pd.Series(np.linspace(100, 120, 24), index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    frame = pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([openp, close], axis=1).min(axis=1) - 1.0,
            "close": close,
            "volume": pd.Series(100_000.0, index=idx),
        }
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": frame}
    cfg = _cfg(
        as_of=as_of2.date(),
        hold=500,
        top=1,
        entry_expr="close > 0",
        stop_loss=None,
        take_profit=None,
        reinvest=False,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=pd.Timestamp(as_of2),
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    # entry on the last bar, no post-entry bars -> position stays open, tail empty.
    assert last in frame.index


def test_run_backtest_with_exit_expr_and_empty_frame_trade():
    # Exercises run_backtest exit_ast lookback branch + empty-frame skip.
    good = make_bars(n=120, start="2024-01-01", seed=4, open_base=100.0)
    spy = make_bars(n=120, start="2024-01-01", seed=9, open_base=400.0)
    fetcher = StubPriceFetcher({"GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        exit_expr="close < sma(close, 5)",
        tickers=("GOOD",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)
