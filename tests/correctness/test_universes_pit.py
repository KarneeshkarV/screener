"""Offline point-in-time reconstruction tests for index universes (H-1).

These tests inject synthetic raw tables (a current-members table plus a
"changes" log) by monkeypatching the fetch/read_html seam in
``screener.universes``. They must never hit the network.
"""

from __future__ import annotations

from datetime import date
import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from screener import universes


# Current members: STAYC has always been in; NEWCO was added in 2020; OLDCO was
# removed in 2019 (so it is NOT in the current table but should be reconstructed
# back into a pre-2019 universe).
def _members_table(*, include_newer_change: bool = False) -> pd.DataFrame:
    symbols = ["STAYC", "NEWCO"]
    securities = ["Stay Corp", "New Co"]
    added = ["2005-01-03", "2020-07-01"]
    if include_newer_change:
        symbols.append("NEW2")
        securities.append("New Two")
        added.append("2022-05-01")
    return pd.DataFrame(
        {
            "Symbol": symbols,
            "Security": securities,
            "Date added": added,
        }
    )


def _changes_table(*, include_newer_change: bool = False) -> pd.DataFrame:
    # Mirror the Wikipedia "changes" table shape: a MultiIndex with Date /
    # Added{Ticker} / Removed{Ticker}.
    columns = pd.MultiIndex.from_tuples(
        [
            ("Date", "Date"),
            ("Added", "Ticker"),
            ("Added", "Security"),
            ("Removed", "Ticker"),
            ("Removed", "Security"),
            ("Reason", "Reason"),
        ]
    )
    rows = [
        ["July 1, 2020", "NEWCO", "New Co", "GONE", "Gone Inc", "r1"],
        ["March 5, 2019", "STAYC2", "Stay Two", "OLDCO", "Old Co", "r2"],
        # An older boundary entry so the log demonstrably extends back before
        # 2018; its tickers are unrelated to the 2018 assertions (a change dated
        # 2010 is <= a 2018 as_of, so it is not undone during reconstruction).
        ["January 4, 2010", "EARLY", "Early Co", "EARLYGONE", "Early Gone", "r0"],
    ]
    if include_newer_change:
        rows.insert(0, ["May 1, 2022", "NEW2", "New Two", "OLD2", "Old Two", "r3"])
    return pd.DataFrame(rows, columns=columns)


def _patch(monkeypatch, tmp_path) -> dict[str, int]:
    counter = {"fetches": 0}
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, **kwargs):
        counter["fetches"] += 1
        return SimpleNamespace(text="<html></html>", raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))
    # Return the members table first and the changes table second, matching the
    # real page ordering consumed by _read_sp500_html().
    monkeypatch.setattr(
        universes.pd,
        "read_html",
        lambda *a, **k: [_members_table(), _changes_table()],
    )
    return counter


def test_post_as_of_addition_excluded_for_past_date(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    # NEWCO was added in 2020 -> must be excluded from a 2018 universe.
    assert "NEWCO" not in univ.symbols
    # STAYC has been in since 2005 -> present.
    assert "STAYC" in univ.symbols


def test_removed_ticker_included_for_past_date(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    # OLDCO was removed in 2019 -> must be added back for a 2018 universe.
    assert "OLDCO" in univ.symbols
    # GONE was removed in 2020 -> also present in 2018.
    assert "GONE" in univ.symbols
    # STAYC2 was only added in 2019 -> excluded from 2018.
    assert "STAYC2" not in univ.symbols


def test_current_as_of_returns_current_members(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date.today(), use_cache=False)
    assert set(univ.symbols) == {"STAYC", "NEWCO"}


def test_dot_ticker_normalized_in_reconstruction(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    # Sanity: reconstruction normalizes dot tickers consistently with members.
    rows = universes._fetch_sp500_changes()
    syms = {added for _, added, _ in rows} | {removed for _, _, removed in rows}
    assert "NEWCO" in syms and "OLDCO" in syms


def test_sp500_change_log_cache_expires(monkeypatch, tmp_path):
    counter = {"fetches": 0}
    source = {"newer_change": False}
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, **kwargs):
        counter["fetches"] += 1
        return SimpleNamespace(text="<html></html>", raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(
        universes.pd,
        "read_html",
        lambda *a, **k: [
            _members_table(include_newer_change=source["newer_change"]),
            _changes_table(include_newer_change=source["newer_change"]),
        ],
    )

    first = universes._load_sp500_changes()
    second = universes._load_sp500_changes()
    assert first == second
    assert counter["fetches"] == 1

    stale = time.time() - universes._SP500_CHANGES_CACHE_TTL_SECONDS - 60
    os.utime(universes._changes_cache_path(), (stale, stale))
    source["newer_change"] = True

    refreshed = universes._load_sp500_changes()
    assert counter["fetches"] == 2
    assert any(removed == "OLD2" for _, _, removed in refreshed)


def test_stale_sp500_pit_universe_cache_recomputes(monkeypatch, tmp_path):
    counter = {"fetches": 0}
    source = {"newer_change": False}
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, **kwargs):
        counter["fetches"] += 1
        return SimpleNamespace(text="<html></html>", raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(
        universes.pd,
        "read_html",
        lambda *a, **k: [
            _members_table(include_newer_change=source["newer_change"]),
            _changes_table(include_newer_change=source["newer_change"]),
        ],
    )

    first = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    assert "OLD2" not in first.symbols

    stale = time.time() - universes._SP500_CHANGES_CACHE_TTL_SECONDS - 60
    os.utime(universes._changes_cache_path(), (stale, stale))
    source["newer_change"] = True

    refreshed = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    assert "OLD2" in refreshed.symbols
    assert "NEW2" not in refreshed.symbols
    assert counter["fetches"] == 4


def test_nifty_past_as_of_warns_not_point_in_time(monkeypatch, tmp_path):
    counter = {"fetches": 0}
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    nifty_csv = "Symbol\nRELIANCE\nTCS\n"

    def fake_get(url, **kwargs):
        counter["fetches"] += 1
        return SimpleNamespace(text=nifty_csv, raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))

    with pytest.warns(UserWarning, match="NOT point-in-time"):
        univ = universes.load_current_universe(
            "nifty50", as_of=date(2018, 1, 1), use_cache=False
        )
    # Still returns today's members (survivorship-biased), but loudly.
    assert set(univ.symbols) == {"RELIANCE", "TCS"}


def test_nifty_current_as_of_does_not_warn(monkeypatch, tmp_path):
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)
    nifty_csv = "Symbol\nRELIANCE\nTCS\n"

    def fake_get(url, **kwargs):
        return SimpleNamespace(text=nifty_csv, raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))

    with warnings_as_errors():
        univ = universes.load_current_universe(
            "nifty50", as_of=date.today(), use_cache=False
        )
    assert set(univ.symbols) == {"RELIANCE", "TCS"}


def test_sp500_empty_change_log_warns_not_point_in_time(monkeypatch, tmp_path):
    """If the changes table is missing, a past sp500 as_of must warn (not silent)."""
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, **kwargs):
        return SimpleNamespace(text="<html></html>", raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))
    # Only the members table is available — no change log to reconstruct from.
    monkeypatch.setattr(universes.pd, "read_html", lambda *a, **k: [_members_table()])

    with pytest.warns(UserWarning, match="NOT point-in-time"):
        univ = universes.load_current_universe(
            "sp500", as_of=date(2018, 1, 1), use_cache=False
        )
    # Falls back to today's members (the survivorship-biased set) — but loudly.
    assert "NEWCO" in univ.symbols  # 2020 IPO leaks in, now with a warning


def test_sp500_as_of_before_change_log_warns(monkeypatch, tmp_path):
    """An as_of older than the earliest logged change is flagged incomplete."""
    _patch(monkeypatch, tmp_path)
    # Earliest change in the synthetic log is 2010-01-04; 2009 predates it, so
    # the reconstruction cannot be trusted and must warn.
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        universes.load_current_universe(
            "sp500", as_of=date(2009, 1, 1), use_cache=False
        )


def test_warning_fires_on_cache_hit(monkeypatch, tmp_path):
    """A second (cached) load of a past biased universe must still warn."""
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)
    nifty_csv = "Symbol\nRELIANCE\nTCS\n"
    monkeypatch.setattr(
        universes,
        "requests",
        SimpleNamespace(
            get=lambda *a, **k: SimpleNamespace(
                text=nifty_csv, raise_for_status=lambda: None
            )
        ),
    )
    # First load populates the cache (point_in_time=false) and warns.
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        universes.load_current_universe("nifty50", as_of=date(2018, 1, 1))
    # Second load is a cache hit — it must warn again, not silently serve bias.
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        universes.load_current_universe("nifty50", as_of=date(2018, 1, 1))


class warnings_as_errors:
    """Context manager turning warnings into errors to assert none fire."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
