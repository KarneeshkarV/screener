from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from click.testing import CliRunner
import pandas as pd
import pytest
from pydantic import ValidationError

from screener import cache
from screener.cli import cli as _root_cli  # import-order guard used by CLI modules
from screener.options import cli as options_cli
from screener.options.metrics import (
    classify_oi_changes,
    compute_chain_metrics,
    safe_ratio,
)
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_bhavcopy import (
    load_bhavcopy_chains,
    normalize_bhavcopy_options,
)
from screener.options.panels import (
    append_chains,
    build_india_panel,
    enrich_panel_history,
    metrics_row,
    read_options_panel,
    show_symbol,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nse_fo_bhavcopy_options_sample.csv"


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture
def panel_root(tmp_path: Path):
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "panels"
    finally:
        cache.reset_cache_area_paths()


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


def _chain(*contracts: OptionContract, spot: float = 100.0) -> OptionChain:
    return OptionChain(
        underlying="abc",
        market="us",
        spot=spot,
        as_of=date(2026, 7, 10),
        source="fixture",
        contracts=tuple(contracts),
    )


def test_models_are_normalized_frozen_and_validate_quotes():
    contract = _contract(underlying="abc")
    assert contract.underlying == "ABC"
    assert contract.as_of.tzinfo is timezone.utc
    with pytest.raises(ValidationError):
        _contract(bid=8.0, ask=7.0)
    with pytest.raises(ValidationError):
        contract.oi = 2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="underlying mismatch"):
        OptionChain(
            underlying="XYZ",
            market="us",
            as_of=date(2026, 7, 10),
            source="fixture",
            contracts=(contract,),
        )


def test_normalize_real_truncated_bhavcopy(sample_frame: pd.DataFrame):
    chains = normalize_bhavcopy_options(sample_frame, as_of=date(2026, 7, 8))
    assert list(chains) == ["RELIANCE"]
    chain = chains["RELIANCE"]
    assert chain.market == "india"
    assert chain.spot == pytest.approx(1275.9)
    assert chain.expiries == (date(2026, 7, 28), date(2026, 8, 25))
    assert len(chain.contracts) == 12
    assert all(contract.lot_size == 500 for contract in chain.contracts)
    assert all(contract.source == "nse_bhavcopy" for contract in chain.contracts)
    assert chain.contracts[0].previous_close is not None


def test_normalize_bhavcopy_filters_symbols_and_validates_columns(
    sample_frame: pd.DataFrame,
):
    assert (
        normalize_bhavcopy_options(
            sample_frame, as_of=date(2026, 7, 8), symbols={"TCS"}
        )
        == {}
    )
    assert (
        normalize_bhavcopy_options(sample_frame.iloc[0:0], as_of=date(2026, 7, 8)) == {}
    )
    with pytest.raises(ValueError, match="missing columns"):
        normalize_bhavcopy_options(
            pd.DataFrame({"FinInstrmTp": ["STO"]}), as_of=date(2026, 7, 8)
        )


def test_normalize_bhavcopy_fallback_fields_and_bad_rows(sample_frame: pd.DataFrame):
    row = sample_frame.iloc[[0]].copy()
    row["FinInstrmNm"] = ""
    row["TradDt"] = "bad"
    row["NewBrdLotQty"] = float("nan")
    row["ClsPric"] = 0
    row["LastPric"] = 0
    row["StrkPric"] = 1250
    chain = normalize_bhavcopy_options(
        row,
        as_of=date(2026, 7, 9),
        lot_sizes={"RELIANCE": 250},
    )["RELIANCE"]
    contract = chain.contracts[0]
    assert contract.symbol == "RELIANCE-2026-07-28-1250-CE"
    assert contract.as_of.date() == date(2026, 7, 9)
    assert contract.last == pytest.approx(45.25)
    assert contract.lot_size == 250

    invalid = pd.concat([row, row], ignore_index=True)
    invalid.loc[0, "OptnTp"] = "XX"
    invalid.loc[1, "StrkPric"] = -1
    assert normalize_bhavcopy_options(invalid, as_of=date(2026, 7, 9)) == {}


def test_load_bhavcopy_chains_uses_injected_fetcher(sample_frame: pd.DataFrame):
    seen: list[date] = []

    def fetch(day: date) -> pd.DataFrame:
        seen.append(day)
        return sample_frame

    chains = load_bhavcopy_chains(
        date(2026, 7, 8), symbols={"RELIANCE"}, refresh=True, fetcher=fetch
    )
    assert seen == [date(2026, 7, 8)]
    assert "RELIANCE" in chains


def test_core_metrics_from_real_bhavcopy(sample_frame: pd.DataFrame):
    chain = normalize_bhavcopy_options(sample_frame, as_of=date(2026, 7, 8))["RELIANCE"]
    metrics = compute_chain_metrics(chain)
    assert metrics.call_oi == 9_965_000
    assert metrics.put_oi == 9_516_500
    assert metrics.pcr == pytest.approx(9_516_500 / 9_965_000)
    assert metrics.call_put_oi_ratio == pytest.approx(9_965_000 / 9_516_500)
    assert metrics.max_pain_strike in {1250.0, 1270.0, 1300.0}
    assert metrics.support_strikes[0] in {1250.0, 1270.0}
    assert metrics.resistance_strikes[0] == 1300.0
    assert metrics.call_oi_change == 3_150_000
    assert metrics.put_oi_change == 1_144_500
    assert metrics.notional_oi == pytest.approx(
        (metrics.call_oi + metrics.put_oi) * 500 * 1275.9
    )
    assert metrics.median_iv is None

    labels = {row["classification"] for row in classify_oi_changes(chain)}
    assert labels == {"short_buildup", "long_buildup"}


def test_iv_skew_term_structure_implied_move_and_zero_ratios():
    front = date(2026, 7, 31)
    next_expiry = date(2026, 8, 28)
    contracts = (
        _contract(symbol="C95", strike=95, right="call", iv=0.30, delta=0.70),
        _contract(symbol="P95", strike=95, right="put", iv=0.32, delta=-0.25),
        _contract(symbol="C100", strike=100, right="call", iv=0.25, delta=0.25),
        _contract(
            symbol="P100", strike=100, right="put", iv=0.27, delta=-0.50, bid=3, ask=5
        ),
        _contract(
            symbol="C2",
            expiry=next_expiry,
            strike=100,
            right="call",
            iv=0.35,
        ),
        _contract(
            symbol="P2",
            expiry=next_expiry,
            strike=100,
            right="put",
            iv=0.37,
        ),
    )
    metrics = compute_chain_metrics(_chain(*contracts))
    assert metrics.front_expiry == front
    assert metrics.next_expiry == next_expiry
    assert metrics.atm_iv == pytest.approx(0.26)
    assert metrics.put_call_iv_skew == pytest.approx(0.07)
    assert metrics.term_structure_slope == pytest.approx(0.10)
    assert metrics.implied_move_pct == pytest.approx(9.0)

    calls_only = _chain(_contract(oi=10, volume=0))
    zero = compute_chain_metrics(calls_only)
    assert zero.pcr is None
    assert zero.pcr_volume is None
    assert safe_ratio(1, 0) is None
    assert safe_ratio(None, 1) is None


def test_oi_classification_unknown_unchanged_and_no_price():
    chain = _chain(
        _contract(symbol="A", oi_change=None),
        _contract(symbol="B", oi_change=0),
        _contract(symbol="C", oi_change=5, previous_close=None),
        _contract(symbol="D", oi_change=-5, previous_close=None),
        _contract(symbol="E", oi_change=-5, last=6, previous_close=5),
        _contract(symbol="F", oi_change=-5, last=4, previous_close=5),
    )
    labels = [row["classification"] for row in classify_oi_changes(chain)]
    assert labels == [
        "unknown",
        "unchanged",
        "new_oi",
        "unwinding",
        "short_covering",
        "long_unwinding",
    ]


def test_metrics_row_and_causal_history_fields():
    row = metrics_row(_chain(_contract(), _contract(symbol="P", right="put")))
    assert row["SYMBOL"] == "ABC"
    assert row["support_strikes"] == "[100.0]"
    assert row["options_volume"] == 40

    rows = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-01") + pd.Timedelta(days=i),
                "SYMBOL": "ABC",
                "median_iv": iv,
                "options_volume": 100 + i * 10,
                "call_oi": 100 + i,
                "put_oi": 90 + i * 2,
                "call_oi_change": None,
                "put_oi_change": None,
                "oi_chg_ratio": None,
            }
            for i, iv in enumerate([0.2, 0.3, 0.25, 0.4, 0.35, 0.5])
        ]
    )
    enriched = enrich_panel_history(rows)
    assert pd.isna(enriched.iloc[0]["iv_rank"])
    assert enriched.iloc[-1]["iv_rank"] == pytest.approx(100)
    assert enriched.iloc[-1]["iv_history_days"] == 6
    assert enriched.iloc[-1]["options_volume_avg_20"] == pytest.approx(120)
    assert enriched.iloc[-1]["call_oi_change"] == 1
    assert enriched.iloc[-1]["put_oi_change"] == 2
    assert enriched.iloc[-1]["oi_chg_ratio"] == 2

    with pytest.raises(ValueError, match="missing columns"):
        enrich_panel_history(pd.DataFrame({"SYMBOL": ["ABC"]}))


def test_append_read_show_and_build_panel(sample_frame: pd.DataFrame, panel_root: Path):
    chain = normalize_bhavcopy_options(sample_frame, as_of=date(2026, 7, 8))["RELIANCE"]
    stored = append_chains({"RELIANCE": chain}, market="india")
    assert len(stored) == 1
    assert len(read_options_panel("india")) == 1
    assert len(show_symbol("india", "NSE:RELIANCE")) == 1
    assert show_symbol("india", "TCS").empty
    with pytest.raises(ValueError, match="unsupported"):
        read_options_panel("mars")
    with pytest.raises(ValueError, match="unsupported"):
        append_chains([], market="mars")

    calls: list[date] = []

    def fetch(day: date) -> pd.DataFrame:
        calls.append(day)
        if day == date(2026, 7, 9):
            raise FileNotFoundError("holiday")
        frame = sample_frame.copy()
        frame["TradDt"] = day.isoformat()
        frame["OpnIntrst"] += (day - date(2026, 7, 8)).days
        return frame

    errors: list[date] = []
    progress: list[date] = []
    result = build_india_panel(
        date(2026, 7, 8),
        date(2026, 7, 10),
        symbols={"RELIANCE.NS"},
        fetcher=fetch,
        trading_day=lambda _day: True,
        on_error=lambda day, _exc: errors.append(day),
        on_progress=lambda day, _count: progress.append(day),
    )
    assert calls == [date(2026, 7, 8), date(2026, 7, 9), date(2026, 7, 10)]
    assert errors == [date(2026, 7, 9)]
    assert progress == [date(2026, 7, 8), date(2026, 7, 10)]
    assert len(result) == 2
    with pytest.raises(ValueError, match="end"):
        build_india_panel(date(2026, 7, 10), date(2026, 7, 8), fetcher=fetch)


def test_options_cli_build_and_show(monkeypatch, panel_root: Path):
    assert _root_cli is not None
    runner = CliRunner()
    panel = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-08"),
                "SYMBOL": "RELIANCE",
                "source": "nse_bhavcopy",
                "spot": 1275.9,
                "front_expiry": "2026-07-28",
                "pcr": 0.85,
                "contract_count": 12,
                "history_days": 1,
            }
        ]
    )
    monkeypatch.setattr(options_cli, "build_india_panel", lambda *a, **k: panel)
    result = runner.invoke(
        options_cli.options,
        ["build-panel", "-m", "india", "--start", "2026-07-08", "--end", "2026-07-08"],
    )
    assert result.exit_code == 0
    assert "1 rows, 1 symbols" in result.output

    monkeypatch.setattr(options_cli, "show_symbol", lambda *a, **k: panel)
    result = runner.invoke(
        options_cli.options, ["show", "-m", "india", "--symbol", "RELIANCE"]
    )
    assert result.exit_code == 0
    assert "coverage=12 contracts" in result.output
    csv_result = runner.invoke(
        options_cli.options,
        ["show", "-m", "india", "--symbol", "RELIANCE", "--csv"],
    )
    assert "SYMBOL" in csv_result.output

    monkeypatch.setattr(options_cli, "show_symbol", lambda *a, **k: pd.DataFrame())
    missing = runner.invoke(
        options_cli.options, ["show", "-m", "us", "--symbol", "AAPL"]
    )
    assert "No US options panel history" in missing.output
