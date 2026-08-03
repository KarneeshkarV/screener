from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import cache
from screener.cli import cli as _root_cli
from screener.options import cli as options_cli
from screener.options.cboe import (
    CboeOptionsProvider,
    cboe_symbol,
    parse_cboe_chain,
)
from screener.options.greeks import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_live import NSELiveOptionsProvider, parse_nse_chain
from screener.options.panels import SnapshotResult, snapshot_us
from screener.options.provider import FallbackOptionsProvider
from screener.options.yf_chain import (
    YFinanceOptionsProvider,
    chain_from_yfinance_ticker,
)
from screener.providers import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"


def _json_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def cboe_raw():
    return _json_fixture("cboe_aapl_delayed_sample.json")


@pytest.fixture
def yf_raw():
    return _json_fixture("yfinance_option_chain_sample.json")


@pytest.fixture
def nse_raw():
    return _json_fixture("nse_live_option_chain_sample.json")


@pytest.fixture
def panel_root(tmp_path: Path):
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "panels"
    finally:
        cache.reset_cache_area_paths()


def test_black_scholes_price_greeks_and_iv_inversion():
    call = black_scholes_price(100, 100, 1, 0.05, 0.2, "call")
    put = black_scholes_price(100, 100, 1, 0.05, 0.2, "put")
    assert call == pytest.approx(10.4506, rel=1e-4)
    assert put == pytest.approx(5.5735, rel=1e-4)
    greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
    assert greeks is not None
    assert greeks["delta"] == pytest.approx(0.6368, rel=1e-4)
    assert greeks["gamma"] > 0
    recovered = implied_volatility(call, 100, 100, 1, 0.05, "call")
    assert recovered == pytest.approx(0.2, abs=1e-5)
    assert black_scholes_price(0, 100, 1, 0.05, 0.2, "call") is None
    assert black_scholes_greeks(100, 100, 0, 0.05, 0.2, "put") is None
    assert implied_volatility(0, 100, 100, 1, 0.05, "call") is None
    assert implied_volatility(500, 100, 100, 1, 0.05, "call") is None


@pytest.mark.parametrize(
    ("strike", "days", "volatility", "right"),
    [
        (130, 1, 0.8, "call"),
        (130, 5, 0.5, "call"),
        (70, 5, 0.5, "put"),
        (110, 5, 0.5, "call"),
        (90, 5, 0.5, "put"),
    ],
)
def test_implied_volatility_low_vega_round_trip(strike, days, volatility, right):
    time_years = days / 365
    price = black_scholes_price(100, strike, time_years, 0.0, volatility, right)
    assert price is not None

    recovered = implied_volatility(price, 100, strike, time_years, 0.0, right)

    assert recovered == pytest.approx(volatility, abs=1e-6)


def test_implied_volatility_returns_none_when_price_contains_no_volatility_signal():
    price = black_scholes_price(100, 70, 1 / 365, 0.0, 0.2, "call")
    assert price == 30.0
    assert implied_volatility(price, 100, 70, 1 / 365, 0.0, "call") is None


def test_implied_volatility_expands_initial_bracket():
    price = black_scholes_price(100, 100, 1.0, 0.0, 6.0, "call")
    assert price is not None

    recovered = implied_volatility(price, 100, 100, 1.0, 0.0, "call")

    assert recovered == pytest.approx(6.0, abs=1e-6)


def test_parse_verified_cboe_shape(cboe_raw):
    chain = parse_cboe_chain(cboe_raw, requested_symbol="AAPL")
    assert chain is not None
    assert chain.underlying == "AAPL"
    assert chain.spot == pytest.approx(315.43)
    assert len(chain.contracts) == 4
    assert chain.expiries == (date(2026, 7, 10), date(2026, 7, 17))
    assert chain.contracts[0].iv == pytest.approx(0.2459)
    assert chain.contracts[0].gamma == pytest.approx(0.0907)
    assert chain.as_of.tzinfo is not None

    filtered = parse_cboe_chain(
        cboe_raw, requested_symbol="AAPL", expiry=date(2026, 7, 17)
    )
    assert filtered is not None and len(filtered.contracts) == 2


def test_cboe_parser_bad_rows_quotes_and_symbols(cboe_raw):
    raw = json.loads(json.dumps(cboe_raw))
    raw["timestamp"] = "bad"
    raw["data"]["options"][0]["bid"] = 10
    raw["data"]["options"][0]["ask"] = 1
    raw["data"]["options"].extend([{"option": "bad"}, "not-an-object"])
    now = datetime(2026, 7, 11, tzinfo=UTC)
    chain = parse_cboe_chain(raw, requested_symbol="AAPL", now=now)
    assert chain is not None
    assert chain.as_of == now
    assert chain.contracts[0].bid is None and chain.contracts[0].ask is None
    assert parse_cboe_chain({}, requested_symbol="AAPL") is None
    assert (
        parse_cboe_chain({"data": {"options": "bad"}}, requested_symbol="AAPL") is None
    )
    assert cboe_symbol("SPX") == "_SPX"
    assert cboe_symbol("_VIX") == "_VIX"
    assert cboe_symbol("aapl") == "AAPL"


class _Response:
    def __init__(self, payload, status: int = 200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.urls: list[str] = []

    def get(self, url, timeout):
        self.urls.append(url)
        assert timeout == 20
        return _Response(self.payload)


def test_cboe_provider_cache_seam_and_validation(cboe_raw):
    session = _Session(cboe_raw)
    fake_cache = FakeProvider()
    provider = CboeOptionsProvider(
        session=session,
        cache_provider=fake_cache,  # type: ignore[arg-type]
        now=lambda: datetime(2026, 7, 10, tzinfo=UTC),
    )
    chain = provider.fetch_chain("aapl", "us", refresh=True)
    assert chain is not None and len(chain.contracts) == 4
    assert session.urls[0].endswith("/AAPL.json")
    assert fake_cache.calls[0][1] is True
    with pytest.raises(ValueError, match="only the US"):
        provider.fetch_chain("AAPL", "india")
    with pytest.raises(ValueError, match="empty"):
        provider.fetch_chain(" ", "us")

    bad = CboeOptionsProvider(
        session=_Session([]),
        cache_provider=FakeProvider(),  # type: ignore[arg-type]
    )
    assert bad.fetch_chain("AAPL", "us") is None


class _YFTicker:
    def __init__(self, payload, *, options=None):
        self.payload = payload
        self.options = options if options is not None else [payload["expiry"]]
        self.fast_info = {"last_price": payload["spot"]}

    def option_chain(self, expiry):
        assert expiry == self.payload["expiry"]
        return SimpleNamespace(
            calls=pd.DataFrame(self.payload["calls"]),
            puts=pd.DataFrame(self.payload["puts"]),
        )


def test_yfinance_fixture_normalization_and_computed_greeks(yf_raw):
    ticker = _YFTicker(yf_raw)
    chain = chain_from_yfinance_ticker(
        ticker,
        "aapl",
        [yf_raw["expiry"]],
        now=datetime(2026, 7, 10, tzinfo=UTC),
    )
    assert chain is not None
    assert chain.spot == 200
    assert len(chain.contracts) == 2
    call, put = chain.contracts
    assert call.previous_close == pytest.approx(7.6)
    assert call.delta is not None and call.gamma is not None
    assert put.right == "put" and put.iv == pytest.approx(0.3)

    no_volume = pd.DataFrame([{"openInterest": 1, "impliedVolatility": 0.2}])
    sparse = SimpleNamespace(
        fast_info={},
        option_chain=lambda _expiry: SimpleNamespace(calls=no_volume, puts=no_volume),
    )
    counted = chain_from_yfinance_ticker(
        sparse,
        "ABC",
        [yf_raw["expiry"]],
        missing_volume_as_count=True,
    )
    assert counted is not None
    assert sum(contract.volume for contract in counted.contracts) == 2
    assert chain_from_yfinance_ticker(sparse, "ABC", ["bad-date"]) is None


def test_yfinance_provider_fetch_and_validation(yf_raw):
    ticker = _YFTicker(yf_raw)
    provider = YFinanceOptionsProvider(
        ticker_factory=lambda _symbol: ticker,
        configure=lambda: None,
        cache_provider=FakeProvider(),
        now=lambda: datetime(2026, 7, 10, tzinfo=UTC),
    )
    chain = provider.fetch_chain("AAPL", "us", refresh=True)
    assert chain is not None and len(chain.contracts) == 2
    explicit = provider.fetch_chain("AAPL", "us", date(2026, 8, 21))
    assert explicit is not None
    with pytest.raises(ValueError, match="only the US"):
        provider.fetch_chain("AAPL", "india")
    with pytest.raises(ValueError, match="empty"):
        provider.fetch_chain(" ", "us")

    empty = YFinanceOptionsProvider(
        ticker_factory=lambda _symbol: _YFTicker(yf_raw, options=[]),
        configure=lambda: None,
        cache_provider=FakeProvider(),
    )
    assert empty.fetch_chain("AAPL", "us") is None


def test_parse_nse_live_and_filtered_fallback(nse_raw):
    chain = parse_nse_chain(nse_raw, symbol="reliance")
    assert chain is not None
    assert chain.spot == pytest.approx(1275.9)
    assert len(chain.contracts) == 4
    assert chain.contracts[0].iv == pytest.approx(0.225)
    assert chain.contracts[0].oi_change == 100
    filtered_expiry = parse_nse_chain(
        nse_raw, symbol="RELIANCE", expiry=date(2026, 8, 25)
    )
    assert filtered_expiry is not None and len(filtered_expiry.contracts) == 2

    totals = parse_nse_chain(
        {"filtered": {"CE": {"totOI": 100}, "PE": {"totOI": 200}}},
        symbol="NIFTY",
    )
    assert totals is not None
    assert [contract.oi for contract in totals.contracts] == [100, 200]
    assert parse_nse_chain({}, symbol="TCS") is None


def test_nse_provider_forwards_refresh(nse_raw):
    calls = []

    def fetch(symbol, refresh=False):
        calls.append((symbol, refresh))
        return nse_raw

    provider = NSELiveOptionsProvider(raw_fetcher=fetch)
    assert provider.fetch_chain("reliance", "india", refresh=True) is not None
    assert calls == [("RELIANCE", True)]

    with pytest.raises(ValueError, match="only India"):
        provider.fetch_chain("RELIANCE", "us")
    with pytest.raises(ValueError, match="empty"):
        provider.fetch_chain(" ", "india")


def _simple_chain(symbol="AAPL"):
    contract = OptionContract(
        symbol=f"{symbol}260821C00100000",
        underlying=symbol,
        expiry=date(2026, 8, 21),
        strike=100,
        right="call",
        oi=10,
        volume=5,
        as_of=datetime(2026, 7, 10, tzinfo=UTC),
        source="stub",
    )
    return OptionChain(
        underlying=symbol,
        market="us",
        spot=100,
        as_of=contract.as_of,
        source="stub",
        contracts=(contract,),
    )


class _Provider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
        self.calls.append((symbol, market, expiry, refresh))
        if self.error:
            raise self.error
        return self.result


def test_fallback_provider_and_snapshot_batch(panel_root: Path):
    first = _Provider(error=RuntimeError("down"))
    second = _Provider(result=_simple_chain())
    fallback = FallbackOptionsProvider(first, second)
    assert fallback.fetch_chain("AAPL", "us", refresh=True) is not None
    assert second.calls[0][-1] is True
    assert FallbackOptionsProvider(_Provider()).fetch_chain("AAPL", "us") is None

    class BatchProvider:
        def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
            if symbol == "BAD":
                return None
            return _simple_chain(symbol)

    result = snapshot_us(
        ["aapl", "BAD", "aapl"], provider=BatchProvider(), max_workers=2
    )
    assert result.requested == 2
    assert len(result.chains) == 1
    assert result.missing == ("BAD",)
    assert len(result.panel) == 1
    empty = snapshot_us([], provider=BatchProvider())
    assert empty.requested == 0 and len(empty.panel) == 1


def test_snapshot_cli_paths(monkeypatch, panel_root: Path):
    assert _root_cli is not None
    runner = CliRunner()
    summary = SnapshotResult(
        panel=pd.DataFrame(),
        chains=(_simple_chain(),),
        requested=2,
        missing=("BAD",),
    )
    monkeypatch.setattr(options_cli, "snapshot_us", lambda *a, **k: summary)
    result = runner.invoke(
        options_cli.options,
        ["snapshot", "-m", "us", "--tickers", "AAPL,BAD"],
    )
    assert result.exit_code == 0
    assert "AAPL: as_of=" in result.output
    assert "1/2 symbols" in result.output
    assert "Unavailable: BAD" in result.output

    neither = runner.invoke(options_cli.options, ["snapshot", "-m", "us"])
    assert neither.exit_code == 2
    both = runner.invoke(
        options_cli.options,
        [
            "snapshot",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--universe-size",
            "1",
        ],
    )
    assert both.exit_code == 2

    from screener import universes

    monkeypatch.setattr(
        universes,
        "load_current_universe",
        lambda _name: SimpleNamespace(symbols=("AAPL", "MSFT")),
    )
    universe = runner.invoke(
        options_cli.options,
        ["snapshot", "-m", "us", "--universe-size", "1"],
    )
    assert universe.exit_code == 0
