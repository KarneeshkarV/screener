from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import cache
from screener.cli import cli as _root_cli
from screener.operator import fetch as operator_fetch
from screener.operator import output as operator_output
from screener.operator import process as operator_process
from screener.operator import screen as operator_screen
from screener.options import cli as options_cli
from screener.options import nse_bhavcopy
from screener.options.participant import (
    build_participant_panel,
    fetch_market_lots,
    fetch_participant_oi,
    parse_market_lots,
    parse_participant_oi_csv,
    read_participant_panel,
)
from screener.options.regime import (
    build_india_vix_panel,
    build_us_regime_panel,
    fetch_cboe_market_pcr,
    fetch_india_vix_archive,
    fetch_us_volatility,
    parse_cboe_market_stats_html,
    parse_fred_volatility_csv,
    parse_india_vix_archive,
    parse_india_vix_live,
    read_regime_panel,
)
from screener.providers import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"


def _text(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def panel_root(tmp_path: Path):
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "panels"
    finally:
        cache.reset_cache_area_paths()


def test_participant_parser_derives_positioning_fields():
    frame = parse_participant_oi_csv(
        _text("nse_participant_oi_sample.csv"), as_of=date(2026, 7, 8)
    )
    assert frame["participant"].tolist() == ["Client", "DII", "FII", "Pro", "TOTAL"]
    fii = frame[frame["participant"] == "FII"].iloc[0]
    assert fii["index_futures_net"] == 30_225 - 298_811
    assert fii["stock_futures_net"] == 3_804_278 - 3_226_723
    assert fii["index_call_net"] == 486_150 - 769_301
    assert fii["index_put_net"] == 912_286 - 365_622
    assert fii["source"] == "nse_participant_oi"
    with pytest.raises(ValueError, match="Client Type"):
        parse_participant_oi_csv("title\nfoo,bar\n1,2", as_of=date(2026, 7, 8))


def test_participant_fetch_build_and_read(panel_root: Path):
    seen = []

    def fetch_text(url, operation):
        seen.append((url, operation))
        return _text("nse_participant_oi_sample.csv")

    frame = fetch_participant_oi(
        date(2026, 7, 8),
        refresh=True,
        text_fetcher=fetch_text,
        cache_provider=FakeProvider(),
    )
    assert len(frame) == 5
    assert "08072026" in seen[0][0]

    def fetch(day, refresh=False):
        if day == date(2026, 7, 9):
            raise RuntimeError("holiday")
        out = frame.copy()
        out["as_of"] = pd.Timestamp(day)
        return out

    panel = build_participant_panel(
        date(2026, 7, 8),
        date(2026, 7, 10),
        fetcher=fetch,
        trading_day=lambda _day: True,
    )
    assert len(panel) == 10
    assert len(read_participant_panel()) == 10
    with pytest.raises(ValueError, match="end"):
        build_participant_panel(date(2026, 7, 10), date(2026, 7, 8))


def test_market_lots_parser_and_fetch_cache_seam():
    text = _text("nse_market_lots_sample.csv")
    lots = parse_market_lots(text)
    assert lots == {"NIFTY": 65, "BANKNIFTY": 30, "RELIANCE": 500, "TCS": 175}
    fetched = fetch_market_lots(
        refresh=True,
        text_fetcher=lambda _url, _op: text,
        cache_provider=FakeProvider(),
    )
    assert fetched["RELIANCE"] == 500
    with pytest.raises(ValueError, match="no expiry"):
        parse_market_lots("A,B\n1,2")


def test_india_vix_archive_live_and_panel(panel_root: Path):
    archive_text = _text("nse_indices_close_sample.csv")
    frame = parse_india_vix_archive(archive_text, requested_date=date(2026, 7, 8))
    assert frame.iloc[0]["india_vix"] == 12.25
    assert frame.iloc[0]["vol_regime"] == "low"
    assert parse_india_vix_archive(
        archive_text.replace("India VIX", "Other"), requested_date=date(2026, 7, 8)
    ).empty
    with pytest.raises(ValueError, match="missing columns"):
        parse_india_vix_archive("foo,bar\n1,2", requested_date=date(2026, 7, 8))

    live = parse_india_vix_live(
        {"data": [{"index": "INDIA VIX", "last": 26.0}]},
        as_of=date(2026, 7, 10),
    )
    assert live.iloc[0]["vol_regime"] == "high"
    assert parse_india_vix_live([], as_of=date(2026, 7, 10)).empty

    fetched = fetch_india_vix_archive(
        date(2026, 7, 8),
        text_fetcher=lambda _url, _op: archive_text,
        cache_provider=FakeProvider(),
    )
    assert len(fetched) == 1
    panel = build_india_vix_panel(
        date(2026, 7, 8),
        date(2026, 7, 9),
        fetcher=lambda day, **_kwargs: fetched.assign(as_of=pd.Timestamp(day)),
    )
    assert len(panel) == 2
    assert len(read_regime_panel("india")) == 2
    with pytest.raises(ValueError, match="end"):
        build_india_vix_panel(date(2026, 7, 9), date(2026, 7, 8))


class _Response:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, text: str):
        self.text = text
        self.urls = []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return _Response(self.text)


def test_cboe_market_pcr_parser_and_fetch():
    html = _text("cboe_daily_stats_sample.html")
    row = parse_cboe_market_stats_html(html, as_of=date(2026, 7, 8))
    assert row["total_pcr"] == 0.79
    assert row["equity_pcr"] == 0.53
    assert row["spx_pcr"] == 1.07
    with pytest.raises(ValueError, match="no put/call"):
        parse_cboe_market_stats_html("<html/>", as_of=date(2026, 7, 8))

    session = _Session(html)
    frame = fetch_cboe_market_pcr(
        date(2026, 7, 8),
        refresh=True,
        session=session,
        cache_provider=FakeProvider(),
    )
    assert frame.iloc[0]["index_pcr"] == 0.97
    assert session.urls[0][1] == 30


def test_fred_parser_fetch_and_us_regime_panel(panel_root: Path):
    text = _text("fred_vix_sample.csv")
    volatility = parse_fred_volatility_csv(
        text, start=date(2026, 7, 6), end=date(2026, 7, 9)
    )
    assert len(volatility) == 4
    assert volatility.iloc[0]["vol_term_spread"] == pytest.approx(3.21)
    assert volatility.iloc[2]["vol_regime"] == "high"
    with pytest.raises(ValueError, match="observation_date"):
        parse_fred_volatility_csv(
            "date,VIX\n1,2", start=date(2026, 7, 6), end=date(2026, 7, 9)
        )

    fetched = fetch_us_volatility(
        date(2026, 7, 6),
        date(2026, 7, 9),
        session=_Session(text),
        cache_provider=FakeProvider(),
    )
    assert len(fetched) == 4

    def pcr(day, **_kwargs):
        return pd.DataFrame(
            [{"as_of": day.isoformat(), "total_pcr": 0.8, "source_pcr": "cboe"}]
        )

    panel = build_us_regime_panel(
        date(2026, 7, 6),
        date(2026, 7, 9),
        pcr_fetcher=pcr,
        volatility_fetcher=lambda *_args, **_kwargs: volatility,
    )
    assert len(panel) == 4
    assert panel["total_pcr"].notna().all()
    assert set(panel["source"]) == {"cboe_daily_statistics+fred"}
    assert len(read_regime_panel("us")) == 4
    with pytest.raises(ValueError, match="unsupported"):
        read_regime_panel("mars")
    with pytest.raises(ValueError, match="end"):
        build_us_regime_panel(date(2026, 7, 9), date(2026, 7, 6))


def test_operator_loads_explicit_option_chains_for_confirmation(monkeypatch):
    raw = pd.read_csv(FIXTURES / "nse_fo_bhavcopy_options_sample.csv")
    monkeypatch.setattr(operator_fetch, "read_fo_bhavcopy_raw", lambda *a, **k: raw)
    monkeypatch.setattr(nse_bhavcopy, "read_fo_bhavcopy_raw", lambda *a, **k: raw)
    futures = operator_fetch.fetch_fo_bhavcopy(date(2026, 7, 8))
    assert set(futures["SYMBOL"]) == {"RELIANCE"}
    assert futures.attrs == {}
    chains = operator_process.load_bhavcopy_chains(date(2026, 7, 8))
    confirmation = operator_process._options_oi_confirmation(chains)
    row = confirmation.iloc[0]
    assert row["Options_OI_Confirmation"] == "Bearish: call writing"
    assert row["ATM_Call_Writing_OI"] > row["ATM_Put_Writing_OI"]
    assert operator_process._options_oi_confirmation({}).empty


def test_operator_label_and_output_surface_confirmation():
    frame = pd.DataFrame(
        [
            {
                "_is_fno": True,
                "%_Change_Price": 1,
                "%_Change_OI": 2,
                "%_Change_Delivery": 150,
                "Dist_From_52W_High": 5,
                "Options_OI_Confirmation": "Bullish: put writing",
            },
            {
                "_is_fno": True,
                "%_Change_Price": -1,
                "%_Change_OI": 2,
                "%_Change_Delivery": 150,
                "Dist_From_52W_High": 30,
                "Options_OI_Confirmation": "Bullish: put writing",
            },
        ]
    )
    labelled = operator_screen.label(frame)
    assert labelled["Options_Confirms_Futures"].tolist() == [True, False]
    formatted = operator_output._format(labelled)
    assert "Options_OI_Confirmation" in formatted
    assert "Options_Confirms_Futures" in operator_output.OUTPUT_COLUMNS


def test_participant_and_regime_cli_paths(monkeypatch):
    assert _root_cli is not None
    runner = CliRunner()
    participant = parse_participant_oi_csv(
        _text("nse_participant_oi_sample.csv"), as_of=date(2026, 7, 8)
    )
    monkeypatch.setattr(
        options_cli, "build_participant_panel", lambda *a, **k: participant
    )
    result = runner.invoke(
        options_cli.options,
        ["participants", "--start", "2026-07-08", "--end", "2026-07-08"],
    )
    assert result.exit_code == 0
    assert "coverage=5 participant classes" in result.output
    csv_result = runner.invoke(
        options_cli.options,
        ["participants", "--end", "2026-07-08", "--csv"],
    )
    assert "participant" in csv_result.output

    india = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-08"),
                "india_vix": 12.25,
                "source": "nse_index_archive",
            }
        ]
    )
    monkeypatch.setattr(options_cli, "build_india_vix_panel", lambda *a, **k: india)
    result = runner.invoke(
        options_cli.options,
        ["regime", "-m", "india", "--start", "2026-07-08", "--end", "2026-07-08"],
    )
    assert result.exit_code == 0
    assert "INDIA regime coverage" in result.output

    us = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-07-08"),
                "vix": 20,
                "total_pcr": 0.8,
                "source": "cboe_daily_statistics+fred",
            }
        ]
    )
    monkeypatch.setattr(options_cli, "build_us_regime_panel", lambda *a, **k: us)
    result = runner.invoke(
        options_cli.options,
        ["regime", "-m", "us", "--start", "2026-07-08", "--end", "2026-07-08"],
    )
    assert result.exit_code == 0
    assert "US regime coverage" in result.output
