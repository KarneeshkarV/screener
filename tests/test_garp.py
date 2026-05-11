from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from screener.cli import cli
from screener.garp import (
    INDIA_THRESHOLDS,
    add_garp_score,
    _passes_garp,
)


def _passing_row(**overrides):
    row = {
        "name": "AAA",
        "market_cap": 1500.0,
        "sales": 1600.0,
        "peg": 1.2,
        "sales_growth_5y": 18.0,
        "operating_profit_growth": 12.0,
        "eps_growth_5y": 16.0,
        "roe_5y": 17.0,
        "roce_or_roic": 18.0,
        "quarterly_profit_growth": 20.0,
    }
    row.update(overrides)
    return row


def test_garp_filter_accepts_complete_india_match() -> None:
    assert _passes_garp(_passing_row(), INDIA_THRESHOLDS) is True


def test_garp_filter_rejects_missing_required_input() -> None:
    assert _passes_garp(_passing_row(peg=None), INDIA_THRESHOLDS) is False


def test_garp_score_prefers_lower_peg_and_stronger_growth() -> None:
    df = pd.DataFrame(
        [
            _passing_row(name="LOWPEG", peg=0.8, eps_growth_5y=20.0),
            _passing_row(name="HIGHPEG", peg=1.8, eps_growth_5y=13.0),
        ]
    )
    scored = add_garp_score(df)
    assert scored.iloc[0]["name"] == "LOWPEG"
    assert "garp_score" in scored.columns


def test_garp_cli_emits_csv(monkeypatch) -> None:
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    results = add_garp_score(pd.DataFrame([_passing_row(description="Alpha")]))

    monkeypatch.setattr(
        "screener.commands.garp.load_garp_universe", lambda *a, **k: universe
    )
    monkeypatch.setattr(
        "screener.commands.garp.screen_india_garp", lambda *a, **k: results
    )

    res = CliRunner().invoke(cli, ["garp", "-m", "india", "--csv"])

    assert res.exit_code == 0, res.output
    assert "garp_score" in res.output
    assert "AAA" in res.output
