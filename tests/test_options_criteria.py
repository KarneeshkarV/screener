from __future__ import annotations

from datetime import date

from click.testing import CliRunner
import pandas as pd
import pytest

from screener.cli import cli
from screener.criteria import CRITERIA, get_definition
from screener.criteria.plugins import options_signals
from screener.options import criteria as options_criteria
from screener.options.criteria import (
    OptionsCriterionResult,
    latest_panel_rows,
    realized_earnings_moves,
    run_options_criterion,
    screen_options_criterion,
)


def _panel() -> pd.DataFrame:
    rows = []
    for symbol, ivs, volumes in (
        ("AAA", [0.2, 0.22, 0.24, 0.26, 0.28, 0.5], [100, 100, 100, 100, 100, 300]),
        ("BBB", [0.5, 0.48, 0.46, 0.44, 0.42, 0.2], [100, 100, 100, 100, 100, 110]),
    ):
        for offset, (iv, volume) in enumerate(zip(ivs, volumes)):
            rows.append(
                {
                    "as_of": pd.Timestamp("2026-07-01") + pd.Timedelta(days=offset),
                    "SYMBOL": symbol,
                    "source": "fixture",
                    "median_iv": iv,
                    "iv_rank": 100.0 if symbol == "AAA" else 0.0,
                    "iv_history_days": offset + 1,
                    "history_days": offset + 1,
                    "options_volume": volume,
                    "options_volume_avg_20": 100.0 if offset == 5 else None,
                    "unusual_options_ratio": volume / 100 if offset == 5 else None,
                    "call_oi_change": 100.0,
                    "put_oi_change": 250.0,
                    "call_writing_near_spot": 10.0 if symbol == "AAA" else None,
                    "put_writing_near_spot": 80.0 if symbol == "AAA" else None,
                    "implied_move_pct": 4.0 if symbol == "AAA" else 10.0,
                    "pcr": 1.2,
                }
            )
    return pd.DataFrame(rows)


def test_options_plugins_are_registered_as_pipeline_criteria():
    expected = {
        "unusual_options",
        "bullish_oi_buildup",
        "high_iv_rank",
        "low_iv_rank",
        "cheap_earnings_vol",
    }
    assert expected <= set(CRITERIA)
    assert all(get_definition(name).is_pipeline for name in expected)


def test_latest_panel_rows_is_point_in_time_and_validates_schema():
    panel = _panel()
    latest = latest_panel_rows(panel, as_of=date(2026, 7, 4))
    assert set(latest["SYMBOL"]) == {"AAA", "BBB"}
    assert latest["as_of"].max() == pd.Timestamp("2026-07-04")
    assert latest_panel_rows(pd.DataFrame()).empty
    assert latest_panel_rows(panel, as_of=date(2020, 1, 1)).empty
    with pytest.raises(ValueError, match="missing columns"):
        latest_panel_rows(pd.DataFrame({"SYMBOL": ["AAA"]}))


def test_unusual_and_iv_rank_screens_include_coverage():
    panel = _panel()
    unusual = screen_options_criterion(
        "unusual_options",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=panel,
    )
    assert unusual.frame["SYMBOL"].tolist() == ["AAA"]
    assert unusual.frame.iloc[0]["coverage_days"] == 6

    high = screen_options_criterion(
        "high_iv_rank", market="us", limit=10, as_of=date(2026, 7, 6), panel=panel
    )
    low = screen_options_criterion(
        "low_iv_rank", market="us", limit=10, as_of=date(2026, 7, 6), panel=panel
    )
    assert high.frame["SYMBOL"].tolist() == ["AAA"]
    assert low.frame["SYMBOL"].tolist() == ["BBB"]


def test_bullish_oi_uses_exact_india_and_snapshot_proxy():
    result = screen_options_criterion(
        "bullish_oi_buildup",
        market="india",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=_panel(),
    )
    basis = dict(zip(result.frame["SYMBOL"], result.frame["oi_signal_basis"]))
    assert basis == {"AAA": "exact_put_writing", "BBB": "snapshot_diff_proxy"}
    assert "India uses exact" in result.message


def test_thin_panel_messages_and_unknown_criterion():
    thin = _panel()
    thin = thin[thin["as_of"] == pd.Timestamp("2026-07-01")]
    assert (
        "thin"
        in screen_options_criterion(
            "unusual_options", market="us", limit=10, panel=thin
        ).message
    )
    assert (
        "thin"
        in screen_options_criterion(
            "high_iv_rank", market="us", limit=10, panel=thin
        ).message
    )
    thin["call_oi_change"] = None
    assert (
        "no OI-change"
        in screen_options_criterion(
            "bullish_oi_buildup", market="us", limit=10, panel=thin
        ).message
    )
    empty = screen_options_criterion(
        "low_iv_rank", market="us", limit=10, panel=pd.DataFrame()
    )
    assert "No US options panel rows" in empty.message
    with pytest.raises(ValueError, match="unknown"):
        screen_options_criterion("bad", market="us", limit=10, panel=_panel())


def _earnings_fixture(tickers, **_kwargs):
    assert tickers == ["AAA"]
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "earnings_date": pd.to_datetime(["2025-01-10", "2025-04-10", "2025-07-10"]),
        }
    )


def _prices_fixture(tickers, _start, _end):
    assert tickers == ["AAA"]
    index = pd.to_datetime(
        [
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-04-09",
            "2025-04-10",
            "2025-04-11",
            "2025-07-09",
            "2025-07-10",
            "2025-07-11",
        ]
    )
    return {
        "AAA": pd.DataFrame(
            {"close": [100, 102, 105, 100, 104, 108, 100, 101, 106]}, index=index
        )
    }


def test_realized_earnings_moves_and_cheap_vol_screen():
    moves = realized_earnings_moves(
        ["AAA"],
        market="us",
        as_of=date(2026, 7, 6),
        earnings_fetcher=_earnings_fixture,
        price_fetcher=_prices_fixture,
    )
    assert moves["AAA"][0] == pytest.approx(6.0)
    assert moves["AAA"][1] == 3

    panel = _panel()
    panel = panel[panel["SYMBOL"] == "AAA"]
    result = screen_options_criterion(
        "cheap_earnings_vol",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=panel,
        earnings_fetcher=_earnings_fixture,
        price_fetcher=_prices_fixture,
    )
    assert result.frame["SYMBOL"].tolist() == ["AAA"]
    assert result.frame.iloc[0]["realized_earnings_move_pct"] == pytest.approx(6)
    assert result.frame.iloc[0]["vol_edge_pct"] == pytest.approx(2)
    assert result.frame.iloc[0]["earnings_events"] == 3


def test_realized_moves_and_cheap_vol_degrade_cleanly():
    def empty_events(*_args, **_kwargs):
        return pd.DataFrame()

    assert (
        realized_earnings_moves(
            ["AAA"],
            market="us",
            as_of=date(2026, 7, 6),
            earnings_fetcher=empty_events,
            price_fetcher=_prices_fixture,
        )
        == {}
    )
    assert realized_earnings_moves([], market="us", as_of=date(2026, 7, 6)) == {}

    panel = _panel()
    panel["implied_move_pct"] = None
    no_quotes = screen_options_criterion(
        "cheap_earnings_vol", market="us", limit=10, panel=panel
    )
    assert "No front-expiry" in no_quotes.message

    def fail(*_args, **_kwargs):
        raise RuntimeError("offline")

    failed = screen_options_criterion(
        "cheap_earnings_vol",
        market="us",
        limit=10,
        panel=_panel(),
        earnings_fetcher=fail,
        price_fetcher=_prices_fixture,
    )
    assert "unavailable: offline" in failed.message


def test_runner_and_plugin_render_paths(monkeypatch, capsys):
    result = OptionsCriterionResult(
        pd.DataFrame(
            [
                {
                    "as_of": "2026-07-06",
                    "SYMBOL": "AAA",
                    "signal": "high_iv_rank",
                    "iv_rank": 99,
                    "coverage_days": 30,
                }
            ]
        ),
        "coverage note",
    )
    monkeypatch.setattr(
        options_criteria, "screen_options_criterion", lambda *a, **k: result
    )
    run_options_criterion("high_iv_rank", market="us", limit=10, output_csv=False)
    assert "coverage note" in capsys.readouterr().out
    run_options_criterion("high_iv_rank", market="us", limit=10, output_csv=True)
    assert "SYMBOL" in capsys.readouterr().out

    calls = []
    monkeypatch.setattr(
        options_signals,
        "run_options_criterion",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    options_signals.high_iv_rank(market="us", limit=5, output_csv=False)
    options_signals.low_iv_rank(market="us", limit=5, output_csv=False)
    options_signals.unusual_options(market="us", limit=5, output_csv=False)
    options_signals.bullish_oi_buildup(market="us", limit=5, output_csv=False)
    options_signals.cheap_earnings_vol(market="us", limit=5, output_csv=False)
    assert [name for name, _kwargs in calls] == [
        "high_iv_rank",
        "low_iv_rank",
        "unusual_options",
        "bullish_oi_buildup",
        "cheap_earnings_vol",
    ]


def test_screen_cli_dispatches_options_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(
        options_signals,
        "run_options_criterion",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    result = CliRunner().invoke(
        cli, ["screen", "-m", "us", "-c", "high_iv_rank", "-n", "3"]
    )
    assert result.exit_code == 0
    assert calls[0][0] == "high_iv_rank"
    assert calls[0][1]["limit"] == 3
