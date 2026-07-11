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


import numpy as np


import pandas as pd


import pytest


from screener.backtester import data


from screener.backtester.core import (
    _bar_index_on_or_before,
    _eligible_reserve_signal_idx,
    _make_slot_state,
    _passes_entry_filters,
    _precompute_filter_signals,
    prepare_strategy_bars,
    _resolve_universe,
    _trailing_liquidity,
)


from screener.backtester.models import BacktestConfig


from screener.backtester.pine import (
    PineError,
    PineSyntaxError,
    evaluate,
    parse,
    required_lookback,
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


def _stub_env(n=60):
    return StubPriceFetcher(
        {
            "AAA": make_bars(n=n, seed=11, open_base=100.0),
            "BBB": make_bars(n=n, seed=12, open_base=50.0),
            "SPY": make_bars(n=n, seed=99, open_base=400.0),
        }
    )


def _universe_file(tmp_path):
    f = tmp_path / "univ.txt"
    f.write_text("AAA\nBBB\n")
    return f


from screener.backtester.core import (  # noqa: E402
    _SlotState,
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
    out, lb = prepare_strategy_bars(
        cfg.strategy_name,
        bars_by,
        {},
        ["AAA"],
        date(2024, 1, 1),
        date(2024, 4, 1),
        StubPriceFetcher({}),
        [],
        market=cfg.market,
        benchmark=cfg.benchmark,
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
