"""Regressions for the justfile options recipes and two signal gates.

Three independent defects, one file because each is a one-line wiring bug whose
fix is only visible through the caller:

* the ``options-backfill`` recipes still invoked the deleted ``main.py`` entry
  point, so they failed with "No such file or directory" instead of running the
  CLI;
* ``mark-minervini --refresh`` built its price fetcher without ``refresh``, so
  the flag quietly kept serving cached bars;
* ``momentum_12_1_riskadj`` measured its volatility denominator without the
  volume series, so the zero-volume stub filter that every sibling plugin
  applies was silently disabled in ``mom_12_1 / vol_252``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.commands import minervini as minervini_command
from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.momentum_12_1 import risk_adjusted_momentum

_ROOT = Path(__file__).resolve().parents[1]


def test_justfile_does_not_reference_deleted_main_entry_point() -> None:
    justfile = (_ROOT / "justfile").read_text(encoding="utf-8")

    assert "main.py" not in justfile


def test_options_backfill_recipes_use_the_screener_entry_point() -> None:
    justfile = (_ROOT / "justfile").read_text(encoding="utf-8")

    assert "@{{screener}} options build-panel -m india" in justfile
    assert "@{{screener}} options participants --start {{start}}" in justfile
    assert "@{{screener}} options regime -m india --start {{start}}" in justfile


def test_mark_minervini_refresh_reaches_the_price_fetcher(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_price_fetcher(ctx_obj, *, builder, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(minervini_command, "get_price_fetcher", fake_get_price_fetcher)
    monkeypatch.setattr(minervini_command, "scan_minervini", lambda *a, **kw: [])

    result = CliRunner().invoke(
        cli, ["mark-minervini", "-m", "us", "-n", "1", "--refresh"]
    )

    assert result.exit_code == 0, result.output
    assert captured["refresh"] is True


def test_mark_minervini_without_refresh_keeps_fetcher_cached(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_price_fetcher(ctx_obj, *, builder, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(minervini_command, "get_price_fetcher", fake_get_price_fetcher)
    monkeypatch.setattr(minervini_command, "scan_minervini", lambda *a, **kw: [])

    result = CliRunner().invoke(cli, ["mark-minervini", "-m", "us", "-n", "1"])

    assert result.exit_code == 0, result.output
    assert captured["refresh"] is False


def _stub_then_real_bars(n: int = 400) -> tuple[pd.Series, pd.Series]:
    """A real price path whose first 300 sessions never traded.

    The closes stay above the price floor, so only the volume gate can reject
    the trailing 252-day window: 152 zero-volume sessions inside it are well
    below the 50% traded-fraction threshold.
    """
    index = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(10.0, 40.0, n), index=index)
    volume = pd.Series(1_000_000.0, index=index)
    volume.iloc[:300] = 0.0
    return close, volume


def test_risk_adjusted_momentum_guards_vol_with_volume() -> None:
    close, volume = _stub_then_real_bars()

    _, vol, score = risk_adjusted_momentum(close, volume)

    guarded = realized_volatility(close, volume=volume)
    unguarded = realized_volatility(close)

    pd.testing.assert_series_equal(vol, guarded, check_names=False)
    assert np.isnan(vol.iloc[-1])
    assert np.isnan(score.iloc[-1])
    # Without the volume series the stub window looks like a real, calm market.
    assert np.isfinite(unguarded.iloc[-1])
    assert not np.isnan(unguarded.iloc[-1])


def test_risk_adjusted_momentum_unchanged_for_fully_traded_bars() -> None:
    close, volume = _stub_then_real_bars()
    volume.iloc[:] = 1_000_000.0

    _, vol, _ = risk_adjusted_momentum(close, volume)

    pd.testing.assert_series_equal(vol, realized_volatility(close), check_names=False)


if __name__ == "__main__":  # pragma: no cover - exercised via pytest
    raise SystemExit(pytest.main([__file__]))
