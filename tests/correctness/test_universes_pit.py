"""Offline point-in-time reconstruction tests for index universes (H-1).

These tests inject a synthetic Wikipedia revision history by monkeypatching the
fetch/read_html seam in ``screener.universes``. They must never hit the network.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from screener import universes

# A synthetic revision history: each entry is the constituent list the article
# carried on that date. STAYC is in throughout; OLDCO and EARLY are dropped
# along the way; NEWCO only joins in 2020.
_HISTORY: dict[date, tuple[str, ...]] = {
    date(2010, 1, 4): ("STAYC", "OLDCO", "EARLY"),
    date(2019, 3, 5): ("STAYC", "OLDCO"),
    date(2020, 7, 1): ("STAYC", "NEWCO"),
}
_CURRENT = ("STAYC", "NEWCO")
_FILLER_PREFIX = "FILL"


def _revid(when: date) -> int:
    return int(when.strftime("%Y%m%d"))


def _table(symbols: tuple[str, ...], *, column: str = "Symbol", dated: bool = False):
    """A constituent table padded past the loader's minimum row count.

    The padding is not decoration: ``_sp500_symbols_from_revision_html`` picks
    the constituent table out of a revision by row count, so a two-row fixture
    would be skipped exactly as a navbox is.
    """
    padded = list(symbols) + [
        f"{_FILLER_PREFIX}{index:04d}"
        for index in range(universes._SP500_MIN_CONSTITUENT_ROWS)
    ]
    frame = {column: padded}
    if dated:
        frame["Date added"] = ["2005-01-03"] * len(padded)
    return pd.DataFrame(frame)


def _meaningful(symbols) -> set[str]:
    """Drop the row-count padding so assertions read as the fixture is written."""
    return {s for s in symbols if not s.startswith(_FILLER_PREFIX)}


def _patch(monkeypatch, tmp_path, *, history=None, column="Symbol") -> dict[str, int]:
    """Serve ``history`` through the MediaWiki seam the loader really calls."""
    history = _HISTORY if history is None else history
    counter = {"revision_lookups": 0, "revision_parses": 0}
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, *, params=None, **kwargs):
        params = params or {}
        if params.get("action") == "query":
            counter["revision_lookups"] += 1
            as_of = date.fromisoformat(str(params["rvstart"])[:10])
            candidates = [when for when in sorted(history) if when <= as_of]
            revisions = [{"revid": _revid(candidates[-1])}] if candidates else []
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"query": {"pages": [{"revisions": revisions}]}},
            )
        if params.get("action") == "parse":
            counter["revision_parses"] += 1
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"parse": {"text": f"REVISION:{params['oldid']}"}},
            )
        return SimpleNamespace(text="CURRENT", raise_for_status=lambda: None)

    def fake_read_html(buf, *args, **kwargs):
        raw = buf.getvalue() if hasattr(buf, "getvalue") else str(buf)
        if raw.startswith("REVISION:"):
            when = next(w for w in history if _revid(w) == int(raw.split(":", 1)[1]))
            return [_table(history[when], column=column)]
        return [_table(_CURRENT, dated=True)]

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(universes.pd, "read_html", fake_read_html)
    return counter


def test_post_as_of_addition_excluded_for_past_date(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    # NEWCO only appears in the 2020 revision -> absent from a 2018 universe.
    assert "NEWCO" not in univ.symbols
    # STAYC is in every revision -> present.
    assert "STAYC" in univ.symbols


def test_removed_ticker_included_for_past_date(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    # OLDCO and EARLY are gone from the current table but were members in 2018.
    assert "OLDCO" in univ.symbols
    assert "EARLY" in univ.symbols


def test_reconstruction_reads_the_revision_current_at_as_of(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date(2019, 6, 1))
    # The 2019-03-05 revision is the newest on or before 2019-06-01: EARLY has
    # already gone, NEWCO has not yet arrived.
    assert _meaningful(univ.symbols) == {"STAYC", "OLDCO"}
    assert univ.source.endswith(str(_revid(date(2019, 3, 5))))


def test_current_as_of_returns_current_members(monkeypatch, tmp_path):
    counter = _patch(monkeypatch, tmp_path)
    univ = universes.load_current_universe("sp500", as_of=date.today(), use_cache=False)
    assert _meaningful(univ.symbols) == set(_CURRENT)
    # Today needs no reconstruction, so the revision API is never consulted.
    assert counter["revision_lookups"] == 0


def test_old_revisions_ticker_symbol_column_is_read(monkeypatch, tmp_path):
    """Revisions before 2022 head the column "Ticker symbol", not "Symbol"."""
    _patch(monkeypatch, tmp_path, column="Ticker symbol")
    univ = universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    assert "OLDCO" in univ.symbols


def test_revision_table_below_minimum_rows_is_not_a_constituent_table(
    monkeypatch, tmp_path
):
    """A navbox-sized table must not be mistaken for the constituent list."""
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(
        universes.pd,
        "read_html",
        lambda *a, **k: [pd.DataFrame({"Symbol": ["STAYC", "OLDCO"]})],
    )
    assert universes._sp500_symbols_from_revision_html("REVISION:20100104") is None


def test_parsed_revision_is_cached_and_reused(monkeypatch, tmp_path):
    counter = _patch(monkeypatch, tmp_path)
    first = universes._sp500_revision_members(_revid(date(2010, 1, 4)))
    second = universes._sp500_revision_members(_revid(date(2010, 1, 4)))
    assert first == second
    # A revision's content is immutable, so it is parsed exactly once.
    assert counter["revision_parses"] == 1


def test_past_universe_cache_records_its_source_revision(monkeypatch, tmp_path):
    counter = _patch(monkeypatch, tmp_path)
    universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    cached = universes._read_cache("sp500", date(2018, 1, 1))
    assert cached is not None
    _, point_in_time, metadata = cached
    assert point_in_time
    assert metadata["sp500_pit_revid"] == str(_revid(date(2010, 1, 4)))

    lookups_before = counter["revision_lookups"]
    universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    # The revision behind a past date can never change, so the entry is reused.
    assert counter["revision_lookups"] == lookups_before


def test_biased_fallback_cache_is_not_reused(monkeypatch, tmp_path):
    """An entry with no source revision must be recomputed, not served again."""
    _patch(monkeypatch, tmp_path, history={})
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        universes.load_current_universe("sp500", as_of=date(2018, 1, 1))
    cached = universes._read_cache("sp500", date(2018, 1, 1))
    assert cached is not None
    assert "sp500_pit_revid" not in cached[2]
    assert not universes._sp500_pit_cache_is_reusable(cached[2])


def test_membership_windows_span_the_backtest_range(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    windows = universes.load_sp500_membership_windows(
        start=date(2018, 1, 1), end=date(2021, 1, 1)
    )
    by_symbol = {
        symbol: (start, until)
        for symbol, start, until in windows
        if not symbol.startswith(_FILLER_PREFIX)
    }
    # EARLY was a member at the window start and left before it ended, so it
    # gets a bounded window rather than being dropped entirely.
    assert by_symbol["EARLY"][0] == date(2018, 1, 1)
    assert by_symbol["EARLY"][1] is not None
    # NEWCO only becomes eligible once the 2020 revision lists it.
    assert by_symbol["NEWCO"][0] >= date(2020, 7, 1)
    assert by_symbol["STAYC"][1] is None


def test_membership_windows_empty_without_revision_history(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, history={})
    windows = universes.load_sp500_membership_windows(
        start=date(2018, 1, 1), end=date(2021, 1, 1)
    )
    # No history is reported as "unknown", never as "nobody was a member". The
    # caller (load_universe_selection) is what turns that into the warning; this
    # function stays quiet so a per-quarter sweep cannot emit one warning a quarter.
    assert windows == ()


def test_point_in_time_selection_carries_windows(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    selection = universes.load_universe_selection(
        "sp500",
        market="us",
        as_of=date(2021, 1, 1),
        point_in_time=True,
        start=date(2018, 1, 1),
    )
    assert selection.membership_windows
    assert "EARLY" in selection.symbols


def test_selection_without_point_in_time_has_no_windows(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    selection = universes.load_universe_selection(
        "sp500", market="us", as_of=date.today()
    )
    assert selection.membership_windows == ()


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


def test_sp500_unreadable_revision_warns_not_point_in_time(monkeypatch, tmp_path):
    """If no revision can be read, a past sp500 as_of must warn (not silent)."""
    _patch(monkeypatch, tmp_path, history={})

    with pytest.warns(UserWarning, match="NOT point-in-time"):
        univ = universes.load_current_universe(
            "sp500", as_of=date(2018, 1, 1), use_cache=False
        )
    # Falls back to today's members (the survivorship-biased set) - but loudly.
    assert "NEWCO" in univ.symbols  # 2020 IPO leaks in, now with a warning


def test_sp500_as_of_before_the_oldest_revision_warns(monkeypatch, tmp_path):
    """An as_of older than the oldest revision is flagged incomplete."""
    _patch(monkeypatch, tmp_path)
    # The oldest synthetic revision is 2010-01-04; 2009 predates the article's
    # history, so there is nothing to reconstruct from and it must warn.
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
