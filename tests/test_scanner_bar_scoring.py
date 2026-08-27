"""Scanner behaviour that is specific to bar-derived scorers.

Two regressions are pinned here: ``--refresh`` must reach the bar cache, not
only the TradingView snapshot fetch, and the over-fetch that widens the field
for scoring must stay small when every extra row costs a price download.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from screener import scanner as scanner_module
from screener.scoring import OUTPUT_SCORE_COLUMN, ScoreSpec, get_scorer

BAR_SCORER = "momentum_12_1"
SNAPSHOT_SCORER = "ema"

_AS_OF = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _scan_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "NSE:AAA",
                "name": "AAA",
                "description": "Acme Ltd",
                "close": 100.0,
                "change": 1.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1_000_000_000.0,
            }
        ]
    )


def _run_scan(monkeypatch, *, refresh: bool, strict: bool = False) -> dict[str, object]:
    """Drive ``scan`` with the network seams faked; return what scoring saw."""
    captured: dict[str, object] = {}

    def fake_fetch(
        plan: scanner_module.ScannerPlan,
        **kwargs: object,
    ) -> tuple[int, pd.DataFrame, datetime]:
        captured["snapshot_refresh"] = refresh
        captured["snapshot_strict"] = kwargs.get("strict", False)
        return 1, _scan_frame(), _AS_OF

    def fake_apply_score(
        df: pd.DataFrame,
        spec: ScoreSpec,
        *,
        market: str | None = None,
        refresh: bool = False,
        strict: bool = False,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["score_refresh"] = refresh
        captured["score_strict"] = strict
        captured["score_market"] = market
        captured["score_name"] = spec.name
        return df.assign(**{OUTPUT_SCORE_COLUMN: [1.0]})

    monkeypatch.setattr(scanner_module.TRADINGVIEW_SCANNER, "fetch", fake_fetch)
    monkeypatch.setattr(scanner_module, "apply_score", fake_apply_score)

    scanner_module.scan(
        "india",
        [],
        limit=5,
        order_by=OUTPUT_SCORE_COLUMN,
        refresh=refresh,
        strict=strict,
        scorer=get_scorer(BAR_SCORER),
    )
    return captured


def test_scan_refresh_reaches_the_bar_score_not_only_tradingview(monkeypatch):
    captured = _run_scan(monkeypatch, refresh=True)

    assert captured["snapshot_refresh"] is True
    assert captured["score_refresh"] is True
    assert captured["score_strict"] is False
    assert captured["score_market"] == "india"
    assert captured["score_name"] == BAR_SCORER


def test_scan_without_refresh_leaves_the_bar_cache_alone(monkeypatch):
    captured = _run_scan(monkeypatch, refresh=False)

    assert captured["snapshot_refresh"] is False
    assert captured["score_refresh"] is False
    assert captured["score_strict"] is False


def test_scan_strict_refresh_reaches_the_bar_score_not_only_tradingview(monkeypatch):
    captured = _run_scan(monkeypatch, refresh=True, strict=True)

    assert captured["snapshot_strict"] is True
    assert captured["score_refresh"] is True
    assert captured["score_strict"] is True


def test_scan_strict_without_refresh_still_forwards_strict_to_scoring(monkeypatch):
    """strict without refresh still reaches scoring so the fetcher can no-op it."""
    captured = _run_scan(monkeypatch, refresh=False, strict=True)

    assert captured["snapshot_strict"] is True
    assert captured["score_refresh"] is False
    assert captured["score_strict"] is True


def test_shape_scan_results_forwards_refresh_to_the_scorer(monkeypatch):
    seen: dict[str, object] = {}

    def fake_apply_score(
        df: pd.DataFrame,
        spec: ScoreSpec,
        *,
        market: str | None = None,
        refresh: bool = False,
        **kwargs: object,
    ) -> pd.DataFrame:
        seen["refresh"] = refresh
        return df.assign(**{OUTPUT_SCORE_COLUMN: [1.0]})

    monkeypatch.setattr(scanner_module, "apply_score", fake_apply_score)

    scanner_module.shape_scan_results(
        _scan_frame(),
        limit=5,
        order_by=OUTPUT_SCORE_COLUMN,
        scorer=get_scorer(BAR_SCORER),
        market="india",
        refresh=True,
    )

    assert seen["refresh"] is True


def test_shape_scan_results_forwards_strict_to_the_scorer(monkeypatch):
    seen: dict[str, object] = {}

    def fake_apply_score(
        df: pd.DataFrame,
        spec: ScoreSpec,
        *,
        market: str | None = None,
        refresh: bool = False,
        strict: bool = False,
        **kwargs: object,
    ) -> pd.DataFrame:
        seen["refresh"] = refresh
        seen["strict"] = strict
        return df.assign(**{OUTPUT_SCORE_COLUMN: [1.0]})

    monkeypatch.setattr(scanner_module, "apply_score", fake_apply_score)

    scanner_module.shape_scan_results(
        _scan_frame(),
        limit=5,
        order_by=OUTPUT_SCORE_COLUMN,
        scorer=get_scorer(BAR_SCORER),
        market="india",
        refresh=True,
        strict=True,
    )

    assert seen["refresh"] is True
    assert seen["strict"] is True


def _plan_limit(scorer_name: str | None, *, limit: int, order_by: str) -> int:
    return scanner_module.build_scanner_plan(
        market="india",
        filters=[],
        limit=limit,
        order_by=order_by,
        scorer=get_scorer(scorer_name) if scorer_name else None,
    ).fetch_limit


def test_bar_derived_scorer_over_fetches_far_less_than_a_snapshot_scorer():
    limit = 50
    bars = _plan_limit(BAR_SCORER, limit=limit, order_by=OUTPUT_SCORE_COLUMN)
    snapshot = _plan_limit(SNAPSHOT_SCORER, limit=limit, order_by=OUTPUT_SCORE_COLUMN)

    # Each bar-path row is one price download, so the field is only wide
    # enough to survive the eligibility floor, price-fetch outages, and dedupe.
    assert bars == 250
    assert snapshot == 500
    assert bars > limit


def test_over_fetch_floors_hold_for_a_small_limit():
    assert _plan_limit(BAR_SCORER, limit=5, order_by=OUTPUT_SCORE_COLUMN) == 200
    assert _plan_limit(SNAPSHOT_SCORER, limit=5, order_by=OUTPUT_SCORE_COLUMN) == 500


def test_over_fetch_multipliers_hold_for_a_large_limit():
    assert _plan_limit(BAR_SCORER, limit=200, order_by=OUTPUT_SCORE_COLUMN) == 1000
    assert _plan_limit(SNAPSHOT_SCORER, limit=200, order_by=OUTPUT_SCORE_COLUMN) == 2000


def test_unscored_order_keeps_its_own_over_fetch():
    assert _plan_limit(None, limit=200, order_by="volume") == 600
    assert _plan_limit(None, limit=5, order_by="volume") == 100


def test_raw_aux_column_is_hidden_unless_detail(monkeypatch):
    """``mom_12_1`` is a diagnostic; the default table already has setup_score."""

    def fake_apply_score(
        df: pd.DataFrame,
        spec: ScoreSpec,
        **kwargs: object,
    ) -> pd.DataFrame:
        return df.assign(**{OUTPUT_SCORE_COLUMN: [80.0], "mom_12_1": [0.4]})

    monkeypatch.setattr(scanner_module, "apply_score", fake_apply_score)

    hidden = scanner_module.shape_scan_results(
        _scan_frame(),
        limit=5,
        order_by=OUTPUT_SCORE_COLUMN,
        scorer=get_scorer(BAR_SCORER),
        market="india",
        detail=False,
    )
    shown = scanner_module.shape_scan_results(
        _scan_frame(),
        limit=5,
        order_by=OUTPUT_SCORE_COLUMN,
        scorer=get_scorer(BAR_SCORER),
        market="india",
        detail=True,
    )

    assert "mom_12_1" not in hidden.columns
    assert OUTPUT_SCORE_COLUMN in hidden.columns
    assert shown["mom_12_1"].tolist() == [0.4]
