"""Tests for Pydantic models at CLI / config boundaries."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from screener.commands.requests import GarpRequest, PromoterBuysRequest, ScreenRequest
from screener.config import CliConfig, load_config
from screener.operator.models import OperatorScanRequest


def test_screen_request_rejects_blank_order_by() -> None:
    with pytest.raises(ValidationError):
        ScreenRequest(
            market="us",
            criteria_names=("ema",),
            limit=50,
            order_by="   ",
            output_csv=False,
            detail=False,
            refresh=False,
            cache_ttl="15m",
        )


def test_load_config_returns_cli_config_with_nested_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "log_level: DEBUG\n"
        "screen:\n"
        "  market: india\n"
        "  limit: 10\n"
    )
    cfg = load_config(path)
    assert isinstance(cfg, CliConfig)
    assert cfg.log_level == "DEBUG"
    dumped = cfg.to_click_default_map()
    assert dumped["log_level"] == "DEBUG"
    assert dumped["screen"]["market"] == "india"
    assert dumped["screen"]["limit"] == 10


def test_operator_scan_request_rejects_invalid_universe() -> None:
    with pytest.raises(ValidationError):
        OperatorScanRequest.model_validate(
            {
                "as_of": date.today(),
                "universe": "invalid",
                "out_path": None,
                "only_actions": False,
                "verbose": False,
            }
        )


def test_garp_request_accepts_defaults() -> None:
    r = GarpRequest(market="us")
    assert r.cache_ttl == "1d"
    assert r.universe_size == 200


def test_promoter_buys_request_preserves_optional_floats() -> None:
    r = PromoterBuysRequest(market="india", min_yf_net_pct=0.1, min_market_cap=None)
    assert r.min_yf_net_pct == 0.1
    assert r.min_market_cap is None
