"""Coverage for options-layer edge branches — offline, no network."""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from screener import cache
from screener.earnings_backtest import sentiment
from screener.operator import process as operator_process
from screener.options import (
    cli as options_cli,
    greeks,
    nse_bhavcopy,
    nse_live,
    participant,
    regime,
)
from screener.options.backtest import merge_options_into_bars
from screener.options.cboe import _contract_parts, parse_cboe_chain
from screener.options.criteria import OptionsCriterionResult, _numeric
from screener.options.metrics import (
    _implied_move_pct,
    _iv_skew,
    _near_spot_writing,
    safe_ratio,
)
from screener.options.models import ChainMetrics, OptionChain, OptionContract
from screener.options.nse_live import _contracts_from_records, _filtered_contracts
from screener.options.panels import (
    _history_metrics,
    append_metrics_rows,
    build_india_panel,
    enrich_panel_history,
    show_symbol,
    snapshot_us,
)
from screener.options.provider import FallbackOptionsProvider, default_us_provider
from screener.options.regime import (
    _vol_regime,
    fetch_india_vix_archive,
    parse_india_vix_archive,
    parse_india_vix_live,
)
from screener.options.yf_chain import (
    YFinanceOptionsProvider,
    _configure,
    _spot_from_ticker,
    chain_from_yfinance_ticker,
)
from screener.unusual_volume.option_chain import compute_oc_iv_volume

FIXTURES = Path(__file__).parent / "fixtures"


def _contract(**overrides) -> OptionContract:
    values = {
        "symbol": "ABC260731C00100000",
        "underlying": "ABC",
        "expiry": date(2026, 7, 31),
        "strike": 100.0,
        "right": "call",
        "oi": 100.0,
        "oi_change": 10.0,
        "volume": 20.0,
        "iv": 0.25,
        "bid": 4.0,
        "ask": 6.0,
        "last": 5.0,
        "previous_close": 4.0,
        "delta": 0.25,
        "lot_size": 10.0,
        "as_of": datetime(2026, 7, 10, tzinfo=timezone.utc),
        "source": "fixture",
    }
    values.update(overrides)
    return OptionContract(**values)


@pytest.fixture
def panel_root(tmp_path: Path):
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "panels"
    finally:
        cache.reset_cache_area_paths()


# ───────────────────────── metrics ─────────────────────────


def test_safe_ratio_rejects_non_finite():
    assert safe_ratio(float("inf"), 1.0) is None


def test_iv_skew_nearest_otm_fallback_and_missing_iv():
    call = _contract(strike=105.0, right="call", iv=0.30)
    put = _contract(strike=95.0, right="put", iv=0.40)
    skew = _iv_skew([call, put], 100.0)
    assert skew == pytest.approx(0.10)
    no_iv_put = _contract(strike=95.0, right="put", iv=None)
    assert _iv_skew([call, no_iv_put], 100.0) is None


def test_implied_move_no_strikes():
    assert _implied_move_pct([], 100.0) is None


def test_near_spot_writing_filters_and_put_side():
    far = _contract(strike=500.0, oi_change=5.0)
    no_change = _contract(strike=100.0, oi_change=None)
    put = _contract(strike=100.0, right="put", oi_change=7.0, last=3.0)
    calls, puts = _near_spot_writing([far, no_change, put], 100.0)
    assert calls == 0.0
    assert puts == 7.0


# ───────────────────────── models ─────────────────────────


def test_as_utc_datetime_string_forms():
    contract = _contract(as_of="2026-07-31")
    assert contract.as_of == datetime(2026, 7, 31, tzinfo=timezone.utc)
    naive = _contract(as_of=datetime(2026, 7, 10, 12, 0))
    assert naive.as_of.tzinfo is timezone.utc


def test_nonempty_validators_reject_blank():
    with pytest.raises(ValidationError, match="must not be empty"):
        _contract(symbol="   ")
    with pytest.raises(ValidationError, match="must not be empty"):
        OptionChain(
            underlying="ABC",
            market="us",
            spot=100.0,
            as_of=date(2026, 7, 10),
            source="  ",
            contracts=(),
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        ChainMetrics(underlying="  ", as_of=date(2026, 7, 10), source="x")


# ───────────────────────── cboe ─────────────────────────


def _cboe_fixture() -> dict:
    return json.loads((FIXTURES / "cboe_aapl_delayed_sample.json").read_text())


def test_contract_parts_bad_expiry():
    assert _contract_parts("AAPL999999C00100000") is None


def test_parse_cboe_chain_zero_spot_bad_row_and_empty():
    raw = _cboe_fixture()
    raw["data"]["current_price"] = 0
    raw["data"]["options"][0]["bid"] = 99999.0  # bid > ask → row skipped
    chain = parse_cboe_chain(raw, requested_symbol="AAPL")
    assert chain is not None
    assert chain.spot is None
    empty = copy.deepcopy(raw)
    empty["data"]["options"] = []
    assert parse_cboe_chain(empty, requested_symbol="AAPL") is None


# ───────────────────────── nse_live ─────────────────────────


def test_contracts_from_records_skips_malformed(monkeypatch):
    as_of = datetime(2026, 7, 10, tzinfo=timezone.utc)
    rows = [
        "not-a-dict",
        {"strikePrice": 100, "CE": "not-a-dict"},
        {
            "strikePrice": 100,
            "expiryDate": "31-Jul-2026",
            "CE": {"bidprice": 4.0, "askPrice": 6.0, "openInterest": 5},
        },
    ]

    # quote_pair sanitizes every crossed input, so the contract-validation
    # guard is defensive; force it to prove one bad leg cannot abort the batch.
    def raising_contract(**kwargs):
        raise ValueError("synthetic contract rejection")

    monkeypatch.setattr(nse_live, "OptionContract", raising_contract)
    assert (
        _contracts_from_records(
            rows, underlying="ABC", as_of=as_of, default_expiry=date(2026, 7, 31)
        )
        == []
    )


def test_filtered_contracts_skips_bad_legs():
    as_of = datetime(2026, 7, 10, tzinfo=timezone.utc)
    assert _filtered_contracts({"filtered": {"CE": "x"}}, "ABC", as_of) == []
    assert (
        _filtered_contracts({"filtered": {"CE": {"totOI": None}}}, "ABC", as_of) == []
    )


def test_nse_default_fetcher_routes_through_uv_seam(monkeypatch):
    sentinel = {"records": {}}
    monkeypatch.setattr(
        "screener.unusual_volume.option_chain.fetch_option_chain",
        lambda symbol, refresh=False: sentinel,
    )
    assert nse_live._default_fetcher("RELIANCE") is sentinel


# ───────────────────────── greeks ─────────────────────────


def test_implied_volatility_estimate_none(monkeypatch):
    monkeypatch.setattr(greeks, "black_scholes_price", lambda *a, **kw: None)
    assert greeks.implied_volatility(5.0, 100.0, 100.0, 0.5, 0.02, "call") is None


def test_implied_volatility_iteration_cap():
    # One iteration cannot converge to a tiny tolerance → midpoint returned.
    result = greeks.implied_volatility(
        5.0, 100.0, 100.0, 0.5, 0.02, "call", tolerance=1e-12, max_iterations=1
    )
    assert result is not None


# ───────────────────────── panels ─────────────────────────


def test_history_metrics_missing_columns():
    group = pd.DataFrame(
        {"as_of": ["2026-07-09", "2026-07-10"], "SYMBOL": ["AAA", "AAA"]}
    )
    out = _history_metrics(group)
    assert out["options_volume_avg_20"].isna().all()
    assert out["oi_chg_ratio"].isna().all()


def test_enrich_panel_history_empty():
    assert enrich_panel_history(pd.DataFrame()).empty


def test_append_metrics_rows_empty(panel_root):
    assert append_metrics_rows("us_options_metrics", pd.DataFrame()).empty


def test_show_symbol_empty_panel(panel_root):
    assert show_symbol("us", "AAA").empty


def test_build_india_panel_skips_non_trading_days(panel_root):
    panel = build_india_panel(
        date(2026, 7, 4),
        date(2026, 7, 5),
        trading_day=lambda d: False,
    )
    assert panel.empty


def test_snapshot_us_provider_failure(panel_root):
    class ExplodingProvider:
        def fetch_chain(self, symbol, market, refresh=False):
            raise RuntimeError("boom")

    result = snapshot_us(["AAA"], provider=ExplodingProvider())
    assert result.missing == ("AAA",)


# ───────────────────────── participant / regime ─────────────────────────


class _PassthroughCacheProvider:
    def fetch(self, key, load, *, refresh, fallback, operation):
        try:
            return load()
        except RuntimeError:
            return fallback


def test_participant_net_missing_columns():
    text = "Participant wise OI\nClient Type,Future Index Long\nClient,10\n"
    frame = participant.parse_participant_oi_csv(text, as_of=date(2026, 7, 10))
    assert frame["index_futures_net"].isna().all()


def test_participant_default_text_fetcher(monkeypatch):
    monkeypatch.setattr(participant, "fetch_nse_text", lambda url, op, timeout: "csv")
    assert participant._default_text_fetcher("http://x", "op") == "csv"


def test_participant_fetchers_unavailable_text():
    frame = participant.fetch_participant_oi(
        date(2026, 7, 10),
        text_fetcher=lambda url, op: None,
        cache_provider=_PassthroughCacheProvider(),
    )
    assert frame.empty
    lots = participant.fetch_market_lots(
        text_fetcher=lambda url, op: None,
        cache_provider=_PassthroughCacheProvider(),
    )
    assert lots == {}


def test_vol_regime_none():
    assert _vol_regime(None) is None


def test_parse_india_vix_archive_unparseable_value():
    text = "Index Name,Index Date,Closing Index Value\nIndia VIX,10-Jul-2026,NA\n"
    assert parse_india_vix_archive(text, requested_date=date(2026, 7, 10)).empty


def test_regime_default_archive_text(monkeypatch):
    monkeypatch.setattr(regime, "fetch_nse_text", lambda url, op, timeout: "csv")
    assert regime._archive_text("http://x", "op") == "csv"


def test_fetch_india_vix_archive_unavailable_text():
    frame = fetch_india_vix_archive(
        date(2026, 7, 10),
        text_fetcher=lambda url, op: None,
        cache_provider=_PassthroughCacheProvider(),
    )
    assert frame.empty


def test_parse_india_vix_live_malformed_payloads():
    as_of = date(2026, 7, 10)
    assert parse_india_vix_live({"data": "x"}, as_of=as_of).empty
    assert parse_india_vix_live({"data": ["str"]}, as_of=as_of).empty
    assert parse_india_vix_live({"data": [{"index": "OTHER"}]}, as_of=as_of).empty
    bad_value = {"data": [{"index": "India VIX", "last": "abc"}]}
    assert parse_india_vix_live(bad_value, as_of=as_of).empty


# ───────────────────────── provider / yf_chain ─────────────────────────


def test_default_us_provider_builds_fallback_stack():
    assert isinstance(default_us_provider(), FallbackOptionsProvider)


def test_yf_configure_and_spot_from_namespace():
    _configure()
    ticker = SimpleNamespace(fast_info=SimpleNamespace(last_price=12.0))
    assert _spot_from_ticker(ticker) == 12.0


def test_chain_from_yfinance_ticker_naive_now():
    ticker = SimpleNamespace(fast_info={})
    chain = chain_from_yfinance_ticker(
        ticker, "AAA", [], now=datetime(2026, 7, 10, 12, 0)
    )
    assert chain is None


def test_yf_build_chain_unparseable_expiries():
    ticker = SimpleNamespace(fast_info={}, options=["not-a-date"])
    provider = YFinanceOptionsProvider(
        ticker_factory=lambda symbol: ticker,
        configure=lambda: None,
        now=lambda: datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert provider._build_chain("AAA", None, provider.now()) is None


# ───────────────────────── backtest merge ─────────────────────────


def test_merge_options_empty_bars_and_tz_mismatch():
    idx = pd.date_range("2026-07-06", periods=3, freq="B")
    bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    panel = pd.DataFrame(
        {
            "as_of": ["2026-07-06T00:00:00+00:00"],
            "SYMBOL": ["AAA"],
            "median_iv": [0.3],
        }
    )
    result = merge_options_into_bars(
        {"AAA": bars, "EMPTY": pd.DataFrame()},
        market="us",
        fields={"median_iv"},
        panel=panel,
    )
    assert result.bars_by_tv["EMPTY"].empty
    assert result.bars_by_tv["AAA"]["median_iv"].notna().any()


# ───────────────────────── criteria / operator / uv chain ─────────────────────────


def test_numeric_missing_column():
    frame = pd.DataFrame({"a": [1.0]})
    assert _numeric(frame, "b").isna().all()


def test_run_options_criterion_empty(monkeypatch, capsys):
    from screener.options import criteria as criteria_mod

    monkeypatch.setattr(
        criteria_mod,
        "screen_options_criterion",
        lambda name, market, limit: OptionsCriterionResult(
            pd.DataFrame(), "no rows yet"
        ),
    )
    criteria_mod.run_options_criterion(
        "iv_rank", market="us", limit=5, output_csv=False
    )
    assert "no rows yet" in capsys.readouterr().out


def test_options_oi_confirmation_branches(monkeypatch):
    metrics_by_symbol = {
        "MISSING": SimpleNamespace(
            call_writing_near_spot=None, put_writing_near_spot=None
        ),
        "BULL": SimpleNamespace(call_writing_near_spot=1.0, put_writing_near_spot=5.0),
        "FLAT": SimpleNamespace(call_writing_near_spot=0.0, put_writing_near_spot=0.0),
    }
    chains = {symbol: object() for symbol in metrics_by_symbol}

    def fake_metrics(chain):
        symbol = next(s for s, c in chains.items() if c is chain)
        return metrics_by_symbol[symbol]

    monkeypatch.setattr(operator_process, "compute_chain_metrics", fake_metrics)
    frame = operator_process._options_oi_confirmation(chains)
    by_symbol = dict(zip(frame["SYMBOL"], frame["Options_OI_Confirmation"]))
    assert by_symbol["MISSING"] is None
    assert by_symbol["BULL"] == "Bullish: put writing"
    assert by_symbol["FLAT"] == "Neutral"


def test_compute_oc_iv_volume_aggregates():
    raw = {
        "records": {
            "data": [
                {
                    "CE": {"totalTradedVolume": 10, "impliedVolatility": 20},
                    "PE": {"totalTradedVolume": 4, "impliedVolatility": 30},
                }
            ]
        }
    }
    out = compute_oc_iv_volume(raw)
    assert out["total_call_volume"] == 10.0
    assert out["total_put_volume"] == 4.0
    assert out["median_iv"] == 25.0


def test_fetch_iv_sentiment_nse_unparseable_chain(monkeypatch):
    monkeypatch.setattr(sentiment, "cached_json_call", lambda *a, **kw: kw["fetch"]())
    monkeypatch.setattr(
        sentiment,
        "fetch_option_chain",
        lambda symbol: {"records": {"data": [{"CE": {}}]}},
    )
    monkeypatch.setattr(sentiment, "parse_nse_chain", lambda raw, symbol: None)
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") is None


# ───────────────────────── nse_bhavcopy ─────────────────────────


def test_load_bhavcopy_chains_refresh_unlinks_cache(monkeypatch):
    monkeypatch.setattr(
        nse_bhavcopy,
        "_read_raw",
        lambda d: pd.DataFrame(columns=sorted(nse_bhavcopy.REQUIRED_COLUMNS)),
    )
    chains = nse_bhavcopy.load_bhavcopy_chains(date(2026, 7, 10), refresh=True)
    assert chains == {}


# ───────────────────────── options CLI ─────────────────────────


def _cli(args, monkeypatch_map=None):
    return CliRunner().invoke(options_cli.options, args)


def test_cli_build_panel_paths(monkeypatch):
    def raising(start, end, **kwargs):
        raise ValueError("bad range")

    monkeypatch.setattr(options_cli, "build_india_panel", raising)
    res = _cli(["build-panel", "--start", "2026-01-01"])
    assert res.exit_code == 2
    assert "bad range" in res.output

    monkeypatch.setattr(
        options_cli, "build_india_panel", lambda start, end, **kw: pd.DataFrame()
    )
    res = CliRunner().invoke(
        options_cli.options, ["build-panel", "--start", "2026-01-01"]
    )
    assert res.exit_code == 0
    assert "No India options rows" in res.output

    def with_rows(start, end, **kwargs):
        kwargs["on_error"](start, RuntimeError("gap"))
        return pd.DataFrame(
            {"as_of": ["2026-01-02"], "SYMBOL": ["RELIANCE"], "source": ["udiff"]}
        )

    monkeypatch.setattr(options_cli, "build_india_panel", with_rows)
    res = CliRunner().invoke(
        options_cli.options,
        [
            "build-panel",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--tickers",
            "RELIANCE",
        ],
    )
    assert res.exit_code == 0
    assert "Skipped 1 unavailable trading date(s)" in res.output


def test_cli_participants_paths(monkeypatch):
    def raising(start, end, refresh=False):
        raise ValueError("no archive")

    monkeypatch.setattr(options_cli, "build_participant_panel", raising)
    res = CliRunner().invoke(options_cli.options, ["participants"])
    assert res.exit_code == 2
    assert "no archive" in res.output

    monkeypatch.setattr(
        options_cli,
        "build_participant_panel",
        lambda s, e, refresh=False: pd.DataFrame(),
    )
    res = CliRunner().invoke(options_cli.options, ["participants"])
    assert res.exit_code == 0
    assert "No participant OI rows" in res.output

    stale = pd.DataFrame({"as_of": ["2020-01-01"], "participant": ["Client"]})
    monkeypatch.setattr(
        options_cli, "build_participant_panel", lambda s, e, refresh=False: stale
    )
    res = CliRunner().invoke(options_cli.options, ["participants"])
    assert res.exit_code == 0
    assert "No participant OI rows" in res.output


def test_cli_regime_paths(monkeypatch):
    today = date.today().isoformat()
    live = pd.DataFrame({"as_of": [today], "india_vix": [14.0], "source": ["nse"]})

    monkeypatch.setattr(
        options_cli, "build_india_vix_panel", lambda s, e, refresh=False: live
    )
    monkeypatch.setattr(
        options_cli, "fetch_india_vix_live", lambda as_of, refresh=False: live
    )
    monkeypatch.setattr(
        options_cli, "append_panel_snapshot", lambda name, frame, dedupe_keys: live
    )
    res = CliRunner().invoke(options_cli.options, ["regime", "--market", "india"])
    assert res.exit_code == 0, res.output

    res = CliRunner().invoke(
        options_cli.options, ["regime", "--market", "india", "--csv"]
    )
    assert res.exit_code == 0
    assert "india_vix" in res.output

    def raising(s, e, refresh=False):
        raise ValueError("vix down")

    monkeypatch.setattr(options_cli, "build_india_vix_panel", raising)
    res = CliRunner().invoke(options_cli.options, ["regime", "--market", "india"])
    assert res.exit_code == 2
    assert "vix down" in res.output

    monkeypatch.setattr(
        options_cli, "build_india_vix_panel", lambda s, e, refresh=False: pd.DataFrame()
    )
    monkeypatch.setattr(
        options_cli, "fetch_india_vix_live", lambda as_of, refresh=False: pd.DataFrame()
    )
    res = CliRunner().invoke(options_cli.options, ["regime", "--market", "india"])
    assert res.exit_code == 0
    assert "No INDIA options regime rows" in res.output
