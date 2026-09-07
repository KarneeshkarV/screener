from __future__ import annotations

import copy
import pickle
import threading
from collections.abc import Iterable
from datetime import date

import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from screener.backtester import fundamentals
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.cli import cli
from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=3,
        top=1,
        entry_expr="roe_ttm > 15 and close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _sample_payload() -> dict[str, object]:
    return {
        "income": [
            {
                "date": "2024-01-31",
                "acceptedDate": "2024-02-05 16:30:00",
                "revenue": 120.0,
                "eps": 1.2,
            },
            {
                "date": "2023-01-31",
                "acceptedDate": "2023-02-04 16:30:00",
                "revenue": 100.0,
                "eps": 1.0,
            },
            {
                "date": "2022-10-31",
                "acceptedDate": "2022-11-04 16:30:00",
                "revenue": 90.0,
                "eps": 0.9,
            },
        ],
        "ratios": [
            {
                "date": "2024-01-31",
                "priceEarningsRatio": 20.0,
                "priceToBookRatio": 3.0,
                "returnOnEquity": 0.18,
                "debtEquityRatio": 0.4,
            }
        ],
        "key_metrics": [{"date": "2024-01-31", "peRatioTTM": 19.0}],
        "balance": [{"date": "2024-01-31", "totalDebt": 40.0}],
        "enterprise_values": [
            {"date": "2024-01-31", "marketCapitalization": 5_000_000_000.0}
        ],
    }


def test_fmp_payload_normalizes_effective_dates_and_fields():
    frame = fundamentals._normalize_fmp_payload(
        _sample_payload(),
        fields=fundamentals.DEFAULT_FUNDAMENTAL_FIELDS,
        lag_days=1,
    )

    assert pd.Timestamp("2024-02-06") in frame.index
    row = frame.loc[pd.Timestamp("2024-02-06")]
    assert row["pe_ttm"] == 19.0
    assert row["pb_ttm"] == 3.0
    assert row["roe_ttm"] == 18.0
    assert row["revenue_growth_yoy"] == pytest.approx(20.0)
    assert row["eps_growth_yoy"] == pytest.approx(20.0)
    assert row["revenue_up_3q"] == 1.0
    assert row["market_cap"] == 5_000_000_000.0


def test_openscreener_payload_normalizes_revenue_up_3q_with_india_lag():
    frame = fundamentals._normalize_openscreener_payload(
        {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
                {"date": "Mar 2024", "sales": 110.0},
            ]
        },
        fields=("revenue_up_3q",),
        lag_days=60,
    )

    assert pd.Timestamp("2025-03-01") in frame.index
    assert frame.loc[pd.Timestamp("2025-03-01"), "revenue_up_3q"] == 1.0
    assert frame.loc[pd.Timestamp("2024-11-29"), "revenue_up_3q"] == 0.0


def test_yfinance_fetcher_fetches_quarterly_revenue(monkeypatch, fake_provider):
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", fake_provider()
    )
    monkeypatch.setattr(
        fundamentals,
        "_fetch_yfinance_quarterly_revenue",
        lambda ticker: {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
            ]
        },
    )

    fetcher = fundamentals.YFinanceFundamentalFetcher(
        fields=("revenue_up_3q",), lag_days=60
    )
    out = fetcher.fetch(
        ["RELIANCE.NS"],
        date(2024, 1, 1),
        date(2025, 12, 31),
    )

    assert out["RELIANCE.NS"].loc[pd.Timestamp("2025-03-01"), "revenue_up_3q"] == 1.0


def test_merge_fundamentals_forward_fills_only_after_effective_date():
    bars = make_bars(start="2024-02-01", n=8)
    fundamentals_frame = pd.DataFrame(
        {"roe_ttm": [18.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-02-06")]),
    )

    merged = fundamentals.merge_fundamentals_into_bars(
        {"AAA": bars},
        {"AAA": fundamentals_frame},
        {"AAA": "AAA"},
        filing_lag_days=fundamentals.INDIA_FUNDAMENTAL_FILING_LAG_DAYS,
    )["AAA"]

    assert pd.isna(merged.loc[pd.Timestamp("2024-02-05"), "roe_ttm"])
    assert merged.loc[pd.Timestamp("2024-02-06"), "roe_ttm"] == 18.0
    assert merged.loc[pd.Timestamp("2024-02-08"), "roe_ttm"] == 18.0


class _StubFundamentalFetcher:
    lag_days = 0

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.frame = frame

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        return {
            ticker: self.frame.copy() if self.frame is not None else pd.DataFrame()
            for ticker in tickers
        }


def test_rolling_backtest_uses_fundamental_columns_in_entry():
    idx = pd.bdate_range("2024-01-01", periods=50)
    aaa = make_bars(n=50, start="2024-01-01", open_base=100.0)
    aaa.index = idx
    aaa["volume"] = 100_000.0
    spy = make_bars(n=50, start="2024-01-01", open_base=400.0)
    spy.index = idx
    fundamental_frame = pd.DataFrame(
        {"roe_ttm": [20.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-22")]),
    )

    result = run_rolling_backtest(
        _cfg(fundamentals_provider="fmp", fundamental_fields=("roe_ttm",)),
        StubPriceFetcher({"AAA": aaa, "SPY": spy}),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 29),
        fundamental_fetcher=_StubFundamentalFetcher(fundamental_frame),
    )

    assert not result.selection.empty
    assert result.selection["signal_date"].min() >= date(2024, 1, 22)


def test_rolling_backtest_missing_fundamentals_does_not_break_price_only_entry():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=40, start="2024-01-01", open_base=100.0),
            "SPY": make_bars(n=40, start="2024-01-01", open_base=400.0),
        }
    )
    result = run_rolling_backtest(
        _cfg(entry_expr="close > 0", fundamentals_provider="fmp"),
        fetcher,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 2, 20),
        fundamental_fetcher=_StubFundamentalFetcher(),
    )

    assert isinstance(result.trades, list)


def test_fmp_provider_accepts_india_market(monkeypatch):
    from screener.backtester.workflow import resolve_backtest_run

    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")
    run = resolve_backtest_run(
        _rolling_request(market="india", fundamentals_provider="fmp")
    )

    assert run.config.fundamentals_provider == "fmp"
    assert isinstance(run.fundamental_fetcher, fundamentals.FMPFundamentalFetcher)
    assert run.fundamental_fetcher.market == "india"


def _record_fetch_sessions(monkeypatch, fetcher, tickers):
    """Run ``fetcher.fetch`` and return the sessions ``_fetch_fmp_sections`` saw."""
    seen: list[requests.Session] = []
    lock = threading.Lock()

    def fake_sections(symbol, *, api_key, session, limit, fields):
        with lock:
            seen.append(session)
        return {"income": []}

    monkeypatch.setattr(fundamentals, "_fetch_fmp_sections", fake_sections)
    monkeypatch.setattr(
        fundamentals._FMP_PROVIDER,
        "fetch",
        lambda key, loader, **kwargs: loader(),
    )
    fetcher.fetch(tickers, date(2024, 1, 1), date(2024, 12, 31))
    return seen


def test_fetch_reuses_one_session_per_worker_thread(monkeypatch):
    """One session per worker thread, so HTTP keep-alive survives across tickers.

    The old code built a fresh ``requests.Session`` per ticker, which meant a
    new TCP connect plus TLS handshake for each of that ticker's five section
    requests. Assert on what ``_fetch_fmp_sections`` actually receives: a test
    that only exercises the accessor stays green if the per-ticker session is
    reintroduced at the call site.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=2)
    tickers = [f"T{i}" for i in range(12)]

    seen = _record_fetch_sessions(monkeypatch, fetcher, tickers)

    assert len(seen) == len(tickers)
    assert 0 < len({id(session) for session in seen}) <= fetcher.max_workers


def test_fetch_routes_every_call_through_an_injected_session(monkeypatch):
    """An injected session is an override, not a hint the pool may ignore.

    ``max_workers`` defaults to 8, so routing on ``max_workers == 1`` alone
    silently dropped a caller-supplied session - a test double, a proxy, or a
    mounted retry adapter - and used a pooled one instead.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    injected = requests.Session()
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", session=injected)

    seen = _record_fetch_sessions(monkeypatch, fetcher, ["AAA", "BBB", "CCC"])

    assert seen and all(session is injected for session in seen)


def test_fetch_closes_the_sessions_it_pooled(monkeypatch):
    """Pooled sessions are closed on the way out, not left to the collector."""
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=2)
    closed: list[object] = []
    real_close = requests.Session.close
    monkeypatch.setattr(
        requests.Session,
        "close",
        lambda self: (closed.append(self), real_close(self))[1],
    )

    seen = _record_fetch_sessions(monkeypatch, fetcher, ["AAA", "BBB", "CCC", "DDD"])

    assert {id(session) for session in seen} <= {id(session) for session in closed}


def test_fetcher_survives_a_process_boundary(monkeypatch):
    """The fetcher must stay picklable so it can travel to a worker process.

    Nothing pickles this fetcher today - ``optimization/grid.py`` is the only
    ``ProcessPoolExecutor`` in the tree and it ships a ``PriceFetcher``. But
    ``BacktestRun`` holds a ``fundamental_fetcher`` in the same frozen
    dataclass that carries the price fetcher, so the day one is submitted the
    failure lands at submit time with a ``TypeError`` about ``_thread._local``
    rather than anywhere near the cause. Holding the line is a one-line cost.

    ``load_env_file`` is patched out for the same reason as its three
    siblings: unpatched it reads the repo's real ``.env`` and, because
    ``screener.config`` latches ``_DOTENV_LOADED`` for the process, leaves
    ``FMP_API_KEY`` in ``os.environ`` for every test that runs after it.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    revived = pickle.loads(pickle.dumps(fetcher))

    assert revived.market == fetcher.market
    assert revived.max_workers == fetcher.max_workers
    assert copy.deepcopy(fetcher).fields == fetcher.fields


def test_normalize_survives_a_row_whose_date_pandas_rejects(monkeypatch):
    """A good ``acceptedDate`` with a junk ``date`` must not abort the payload.

    Making ``_effective_date`` total moved the failure rather than removing it:
    such a row clears the ``continue`` guard and then reaches the prior-year
    lookup, where ``pd.Timestamp("0000-00-00")`` raised ``DateParseError``. On
    the serial path there is no ``except`` above it, so one bad row killed
    ``screener backtest --tickers X`` with a traceback.
    """
    payload = {
        "income": [
            {"date": "0000-00-00", "acceptedDate": "2024-05-15", "revenue": 100.0},
            {"date": "2024-03-31", "acceptedDate": "2024-05-15", "revenue": 100.0},
        ]
    }

    frame = fundamentals._normalize_fmp_payload(
        payload, fields=("revenue_growth_yoy",), lag_days=0
    )

    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2024-05-15")


def test_prior_year_key_is_none_for_a_date_pandas_rejects():
    assert fundamentals._prior_year_key("2024-06-30") == "2023-06-30"
    assert fundamentals._prior_year_key("0000-00-00") is None
    assert fundamentals._prior_year_key("not-a-date") is None


def test_session_assigned_after_construction_is_routed_through(monkeypatch):
    """``fetcher.session = double`` must win, not be silently ignored.

    The injected-or-not decision used to live in a second attribute written
    only in ``__init__``, so a session installed afterwards left the flag
    false and every worker built a real one and hit the network.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)
    double = requests.Session()
    fetcher.session = double

    seen = _record_fetch_sessions(monkeypatch, fetcher, ["AAA", "BBB", "CCC"])

    assert seen and all(session is double for session in seen)


def test_injected_session_pool_is_sized_for_the_fan_out(monkeypatch):
    """A shared session needs a pool as wide as the fan-out sharing it.

    The default ``HTTPAdapter`` holds 10 connections, so handing an unsized
    session to 16 workers makes urllib3 discard connections and throws away
    the keep-alive this fetcher exists to gain.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    injected = requests.Session()

    fundamentals.FMPFundamentalFetcher(api_key="x", session=injected, max_workers=16)

    adapter = injected.get_adapter("https://financialmodelingprep.com")
    assert adapter._pool_connections == 16
    assert adapter._pool_maxsize == 16


def test_no_session_is_allocated_when_the_pool_serves_the_fan_out(monkeypatch):
    """Without an injected session the instance must not build a dead one.

    ``shared`` was decided by ``max_workers`` while the execution path was
    decided by the ticker count, so the default 8-worker fetcher allocated a
    ``requests.Session`` in ``__init__`` that no path ever used or closed.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    assert fetcher._session is None

    _record_fetch_sessions(monkeypatch, fetcher, ["AAA", "BBB", "CCC"])

    assert fetcher._session is None


def test_a_failed_ticker_is_logged_and_returns_a_dated_empty_frame(monkeypatch, caplog):
    """A swallowed per-ticker failure must leave a trace and a usable frame.

    Silent, a systematic parse failure across every ticker reads downstream as
    a flat strategy with zero trades rather than as the data outage it is. The
    fallback frame also has to carry the same ``DatetimeIndex`` as every other
    return path, so a caller can slice it by date before checking ``.empty``.
    """
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=2)

    def boom(symbol, *, api_key, session, limit, fields):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr(fundamentals, "_fetch_fmp_sections", boom)
    monkeypatch.setattr(
        fundamentals._FMP_PROVIDER,
        "fetch",
        lambda key, loader, **kwargs: loader(),
    )

    with caplog.at_level("WARNING", logger=fundamentals.LOG.name):
        out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31))

    assert set(out) == {"AAA", "BBB"}
    for frame in out.values():
        assert frame.empty
        assert isinstance(frame.index, pd.DatetimeIndex)
    assert "AAA" in caplog.text and "BBB" in caplog.text


def test_openscreener_provider_rejects_non_india_market():
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--entry",
            "close > 0",
            "--fundamentals-provider",
            "openscreener",
        ],
        obj=StubPriceFetcher({}),
    )

    assert res.exit_code != 0
    assert "supports only -m india" in res.output


def test_referenced_fundamental_fields_detects_known_fields():
    from screener.backtester.cli_common import referenced_fundamental_fields

    assert referenced_fundamental_fields("revenue_up_3q > 0 and close > 0", None) == {
        "revenue_up_3q"
    }
    # Pure-price expressions reference no fundamentals.
    assert referenced_fundamental_fields("ema(close, 150) > ema(close, 200)", None) == (
        set()
    )
    # Exit expressions are inspected too.
    assert referenced_fundamental_fields("close > 0", "pe_ttm > 30") == {"pe_ttm"}


def _rolling_request(**overrides):
    from screener.backtester.workflow import BacktestRequest

    values = dict(
        mode="rolling",
        context_obj=StubPriceFetcher({}),
        market="us",
        hold=20,
        top=10,
        entry_expr="close > 0",
        exit_expr=None,
        strategy_name=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        cost_model="flat",
        initial_capital=100_000.0,
        benchmark=None,
        tickers="AAPL",
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        adv_window=20,
        slippage_model="fixed",
        half_spread_bps=0.0,
        vol_impact_k=0.1,
        no_gap_fills=False,
        entry_order="moo",
        entry_limit_bps=None,
        partial_exit_args=(),
        price_adjustment="full",
        interval="1d",
        output_csv=False,
        report_path=None,
        open_report=False,
        sizing_rule="equal_slot",
        sizing_risk_pct=0.01,
        sizing_position_pct=0.1,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
        sizing_vol_window=20,
        intraday_only=False,
    )
    values.update(overrides)
    return BacktestRequest(**values)


def test_rolling_auto_enables_fundamentals_for_fundamental_expr(monkeypatch):
    from screener.backtester.workflow import resolve_backtest_run

    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")
    run = resolve_backtest_run(
        _rolling_request(strategy_name="ema150_200_revenue_up_3q", entry_expr=None)
    )

    assert run.config.fundamentals_provider == "fmp"
    assert isinstance(run.fundamental_fetcher, fundamentals.FMPFundamentalFetcher)


def test_rolling_does_not_enable_fundamentals_for_price_only_expr():
    from screener.backtester.workflow import resolve_backtest_run

    run = resolve_backtest_run(
        _rolling_request(entry_expr="ema(close, 150) > ema(close, 200)")
    )

    assert run.config.fundamentals_provider is None
    assert run.fundamental_fetcher is None


def test_rolling_unions_referenced_field_into_explicit_field_list(monkeypatch):
    from screener.backtester.workflow import resolve_backtest_run

    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")
    run = resolve_backtest_run(
        _rolling_request(
            entry_expr="revenue_up_3q > 0",
            fundamental_field_args=("roe_ttm",),
        )
    )

    assert run.config.fundamentals_provider == "fmp"
    assert set(run.config.fundamental_fields) == {"roe_ttm", "revenue_up_3q"}


# --------------------------------------------------------------------------- #
# Unit coverage for the fundamentals adapter helpers and fetcher orchestration
# --------------------------------------------------------------------------- #


class _RaisingProvider:
    """Provider seam whose ``fetch`` always raises (drives thread except-branch)."""

    def fetch(self, *args, **kwargs):
        raise RuntimeError("provider boom")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, *, headers, timeout):
        return _FakeResponse(self._payload)


def test_num_handles_unparseable_and_nan():
    from screener.financials import to_number as _num

    assert _num("not-a-number") is None
    assert _num(float("nan")) is None
    assert _num("N/A") is None
    assert _num(None) is None
    assert _num("1,234.5%") == 1234.5


def test_increased_last_n_quarters_none_when_value_missing():
    rows = [{"revenue": 100.0}, {"revenue": None}, {"revenue": 90.0}]
    assert fundamentals._increased_last_n_quarters(rows, 0, "revenue", 3) is None


def test_increased_last_n_revenues_none_when_value_missing():
    rows = [{"sales": 100.0}, {"sales": None}, {"sales": 90.0}]
    assert fundamentals._increased_last_n_revenues(rows, 0, 3) is None


def test_effective_date_none_when_unparseable():
    assert fundamentals._effective_date({"date": "not-a-date"}, 1) is None
    assert fundamentals._effective_date({}, 1) is None


@pytest.mark.parametrize(
    "raw",
    [
        "2026-06-30",
        "2026-06-30 00:00:00",
        "2026-06-30 17:30:12",
        "2026-06-30T17:30:12",
        "2026-06-30T17:30:12+05:30",
        "2026-01-01",
        "1999-12-31 23:59:59",
        "not-a-date",
        "Q2 2026",
        # ISO week and ordinal dates: ``datetime.fromisoformat`` accepts
        # these, ``pd.to_datetime`` coerces them to NaT. An earlier cut of
        # ``_filing_timestamp`` used ``fromisoformat`` and turned a row the
        # caller drops into a live effective date, which is look-ahead.
        "2026-W27",
        "2026-W27-1",
        "2026-181",
    ],
)
def test_filing_timestamp_matches_the_pandas_parse_it_replaced(raw):
    """The fast path must never disagree with ``pd.to_datetime``.

    ``_filing_timestamp`` calls ``pd.Timestamp`` directly because the scalar
    ``pd.to_datetime`` costs 151us against 3.4us. That is only safe while both
    produce the same stamp, so pin the equivalence on every shape FMP serves,
    the junk the parse has to reject, and the ISO spellings that a
    stdlib-parser shortcut would wrongly admit.
    """
    parsed = pd.to_datetime(raw, errors="coerce")
    expected = None if pd.isna(parsed) else parsed.tz_localize(None).normalize()
    assert fundamentals._filing_timestamp(raw) == expected


def test_filing_timestamp_falls_back_for_non_string_input():
    stamp = pd.Timestamp("2026-06-30 08:15:00")
    assert fundamentals._filing_timestamp(stamp) == pd.Timestamp("2026-06-30")
    assert fundamentals._filing_timestamp(None) is None


def test_parse_india_period_end_handles_empty_iso_and_garbage():
    assert fundamentals._parse_india_period_end("") is None
    assert fundamentals._parse_india_period_end(None) is None

    iso = fundamentals._parse_india_period_end("2024-03-31")
    assert iso == pd.Timestamp("2024-03-31")

    assert fundamentals._parse_india_period_end("nonsense-label") is None


def test_fmp_get_returns_parsed_json():
    out = fundamentals._fmp_get(
        _FakeSession({"symbol": "AAPL", "revenue": 1}),
        "income-statement/AAPL",
        {"period": "quarter", "limit": 120},
        "test-key",
    )
    assert out == {"symbol": "AAPL", "revenue": 1}


def test_fmp_fetcher_requires_api_key(monkeypatch):
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        fundamentals.FMPFundamentalFetcher()


def test_fmp_fetcher_init_normalizes_config():
    fetcher = fundamentals.FMPFundamentalFetcher(
        api_key="x",
        fields=["roe_ttm", "roe_ttm", "pe_ttm"],
        lag_days=-1,
        limit=0,
        max_workers=0,
    )
    assert fetcher.api_key == "x"
    assert fetcher.fields == ("roe_ttm", "pe_ttm")
    assert fetcher.lag_days == 0
    assert fetcher.limit == 1
    assert fetcher.max_workers == 1
    assert fetcher.refresh is False


def test_fundamental_fetchers_declare_supported_markets():
    assert fundamentals.FMPFundamentalFetcher.markets == frozenset({"us", "india"})
    assert fundamentals.OpenScreenerFundamentalFetcher.markets == frozenset({"india"})
    assert fundamentals.YFinanceFundamentalFetcher.markets == frozenset({"india"})


def test_fmp_fetcher_keeps_india_suffix_but_strips_us(monkeypatch):
    captured_keys: list[tuple] = []

    def fake_provider_fetch(key, _fetch_payload, **_kwargs):
        captured_keys.append(key)
        return {}

    monkeypatch.setattr(fundamentals._FMP_PROVIDER, "fetch", fake_provider_fetch)
    india_fetcher = fundamentals.FMPFundamentalFetcher(
        api_key="test", market="india", max_workers=1
    )
    india_fetcher.fetch(["RELIANCE.NS"], date(2020, 1, 1), date(2024, 1, 1))
    assert captured_keys[-1][:2] == ("india", "RELIANCE.NS")

    us_fetcher = fundamentals.FMPFundamentalFetcher(api_key="test", max_workers=1)
    us_fetcher.fetch(["AAPL.NASDAQ"], date(2020, 1, 1), date(2024, 1, 1))
    assert captured_keys[-1][:2] == ("us", "AAPL")

    with pytest.raises(ValueError, match="US and India"):
        fundamentals.FMPFundamentalFetcher(api_key="test", market="japan")


def test_fmp_fetcher_fetch_single_ticker(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_fetch_fmp_sections", lambda symbol, **k: _sample_payload()
    )
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=1)

    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 12, 31))

    assert "AAA" in out
    assert pd.Timestamp("2024-02-06") in out["AAA"].index


def test_fmp_fetcher_fetch_threaded(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_fetch_fmp_sections", lambda symbol, **k: _sample_payload()
    )
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31))

    assert set(out) == {"AAA", "BBB"}
    assert not out["AAA"].empty


def test_fmp_fetcher_fetch_threaded_handles_provider_failures(monkeypatch):
    monkeypatch.setattr(fundamentals, "_FMP_PROVIDER", _RaisingProvider())
    fetcher = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=4)

    out = fetcher.fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31))

    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].empty
    assert out["BBB"].empty


def test_fundamental_fetcher_protocol_has_no_market_argument():
    assert "market" not in fundamentals.FundamentalFetcher.fetch.__annotations__


def test_openscreener_fetcher_fetch_threaded(monkeypatch, fake_provider):
    monkeypatch.setattr(fundamentals, "_OPENSCREENER_PROVIDER", fake_provider())
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", fake_provider()
    )
    monkeypatch.setattr(
        fundamentals,
        "_fetch_openscreener_quarterly",
        lambda symbol: {
            "quarterly_results": [
                {"date": "Dec 2024", "sales": 130.0},
                {"date": "Sep 2024", "sales": 120.0},
                {"date": "Jun 2024", "sales": 100.0},
            ]
        },
    )
    fetcher = fundamentals.OpenScreenerFundamentalFetcher(max_workers=4)

    out = fetcher.fetch(["RELIANCE.NS", "TCS.NS"], date(2024, 1, 1), date(2025, 12, 31))

    assert set(out) == {"RELIANCE.NS", "TCS.NS"}
    assert pd.Timestamp("2025-03-01") in out["RELIANCE.NS"].index


def test_openscreener_fetcher_fetch_threaded_handles_failures(monkeypatch):
    monkeypatch.setattr(fundamentals, "_OPENSCREENER_PROVIDER", _RaisingProvider())
    monkeypatch.setattr(
        fundamentals, "_YFINANCE_FUNDAMENTALS_PROVIDER", _RaisingProvider()
    )
    fetcher = fundamentals.OpenScreenerFundamentalFetcher(max_workers=4)

    out = fetcher.fetch(["RELIANCE.NS", "TCS.NS"], date(2024, 1, 1), date(2025, 12, 31))

    assert set(out) == {"RELIANCE.NS", "TCS.NS"}
    assert out["RELIANCE.NS"].empty
    assert out["TCS.NS"].empty


def test_build_fundamental_fetcher_resolves_providers(monkeypatch):
    monkeypatch.setattr(fundamentals, "load_env_file", lambda: None)
    monkeypatch.setenv("FMP_API_KEY", "x")

    assert fundamentals.build_fundamental_fetcher(None, market="us") is None
    assert fundamentals.build_fundamental_fetcher("   ", market="us") is None

    assert isinstance(
        fundamentals.build_fundamental_fetcher("fmp", market="us"),
        fundamentals.FMPFundamentalFetcher,
    )
    assert isinstance(
        fundamentals.build_fundamental_fetcher("FMP", market="us"),
        fundamentals.FMPFundamentalFetcher,
    )
    india_fmp = fundamentals.build_fundamental_fetcher("fmp", market="india")
    assert isinstance(india_fmp, fundamentals.FMPFundamentalFetcher)
    assert india_fmp.market == "india"
    assert isinstance(
        fundamentals.build_fundamental_fetcher("openscreener", market="india"),
        fundamentals.OpenScreenerFundamentalFetcher,
    )
    assert isinstance(
        fundamentals.build_fundamental_fetcher("open-screener", market="india"),
        fundamentals.OpenScreenerFundamentalFetcher,
    )

    assert isinstance(
        fundamentals.build_fundamental_fetcher("yfinance", market="india"),
        fundamentals.YFinanceFundamentalFetcher,
    )
    with pytest.raises(ValueError):
        fundamentals.build_fundamental_fetcher("garbage", market="us")


def test_merge_fundamentals_skips_empty_or_none_bars():
    out = fundamentals.merge_fundamentals_into_bars(
        {"AAA": pd.DataFrame(), "BBB": None},
        {},
        {},
        filing_lag_days=0,
    )
    assert out["AAA"].empty
    assert out["BBB"] is None
