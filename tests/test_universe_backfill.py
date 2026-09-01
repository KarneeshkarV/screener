from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from screener import universe_backfill, universes


def _csv(symbols: list[str]) -> str:
    header = "Company Name,Industry,Symbol,Series,ISIN Code"
    rows = "\n".join(f"Co {s},Sector,{s},EQ,INE000{s}" for s in symbols)
    return f"{header}\n{rows}\n"


class _FakeResponse:
    def __init__(self, *, text: str = "", payload=None, status: int = 200) -> None:
        self.text = text
        self._payload = payload
        self._status = status

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise requests.HTTPError(f"status {self._status}")


class _FakeSession:
    """Serves the CDX index from ``captures`` and replays from ``bodies``."""

    def __init__(self, captures, bodies, *, failing_urls=()) -> None:
        self._captures = captures
        self._bodies = bodies
        self._failing_urls = set(failing_urls)
        self.closed = False

    def get(self, url, *, params=None, headers=None, timeout=None):
        if params is not None:
            target = params["url"]
            if target in self._failing_urls:
                raise requests.ConnectionError(f"boom: {target}")
            rows = self._captures.get(target, [])
            if not rows:
                return _FakeResponse(payload=[])
            header = [["timestamp", "original", "digest", "statuscode"]]
            return _FakeResponse(payload=header + rows)
        return _FakeResponse(text=self._bodies[url])

    def close(self) -> None:
        self.closed = True


_NIFTY500_MIRRORS = universe_backfill.BACKFILL_SOURCES["nifty500"].urls


def _replay(timestamp: str, url: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/https://{url}"


def test_parse_nse_index_csv_matches_the_live_nifty500_vocabulary(monkeypatch) -> None:
    text = _csv(["RELIANCE", "TCS", "RELIANCE"])
    monkeypatch.setattr(
        universes,
        "call_with_resilience",
        lambda *a, **k: _FakeResponse(text=text),
    )

    live, _source = universes._fetch_nifty500()
    archived = universes.parse_nse_index_csv(
        text, label="Nifty 500 constituents", suffix=universes.NSE_SYMBOL_SUFFIX
    )

    # The backfill must not drift from the live loader: a window naming a
    # symbol the price panel never keys would drop the name silently.
    assert live == archived == ["RELIANCE.NS", "TCS.NS"]


def test_snapshots_drop_repeat_digests_and_keep_the_last_crawl_of_a_day() -> None:
    mirror_a, mirror_b = _NIFTY500_MIRRORS[0], _NIFTY500_MIRRORS[1]
    captures = {
        mirror_a: [
            ["20240101120000", f"https://{mirror_a}", "DIGEST1", "200"],
            ["20240301120000", f"https://{mirror_a}", "DIGEST1", "200"],
            ["20240601120000", f"https://{mirror_a}", "DIGEST2", "200"],
        ],
        # Same day as the DIGEST2 crawl, and later: one date carries one
        # membership, and the later crawl is the one that saw the change.
        mirror_b: [["20240601180000", f"https://{mirror_b}", "DIGEST3", "200"]],
    }
    session = _FakeSession(captures, {})

    snapshots, warnings = universe_backfill.list_archived_snapshots(
        "nifty500", session=session
    )

    assert warnings == ()
    assert [s.observed for s in snapshots] == [date(2024, 1, 1), date(2024, 6, 1)]
    assert [s.digest for s in snapshots] == ["DIGEST1", "DIGEST3"]


def test_a_membership_revert_is_not_deduped_away() -> None:
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "A", "200"],
            ["20240201120000", f"https://{mirror}", "B", "200"],
            # Back to the January membership. Deduping against every digest
            # ever seen would drop this and leave B eligible forever.
            ["20240301120000", f"https://{mirror}", "A", "200"],
        ]
    }
    session = _FakeSession(captures, {})

    snapshots, _ = universe_backfill.list_archived_snapshots(
        "nifty500", session=session
    )

    assert [s.digest for s in snapshots] == ["A", "B", "A"]


def test_unreachable_mirror_warns_and_keeps_the_others() -> None:
    mirror_a, mirror_b = _NIFTY500_MIRRORS[0], _NIFTY500_MIRRORS[1]
    captures = {mirror_b: [["20240101120000", f"https://{mirror_b}", "D", "200"]]}
    session = _FakeSession(captures, {}, failing_urls=[mirror_a])

    snapshots, warnings = universe_backfill.list_archived_snapshots(
        "nifty500", session=session
    )

    assert [s.observed for s in snapshots] == [date(2024, 1, 1)]
    assert any(mirror_a in warning for warning in warnings)


def test_since_and_until_bound_the_crawl_window() -> None:
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20220101120000", f"https://{mirror}", "OLD", "200"],
            ["20240101120000", f"https://{mirror}", "MID", "200"],
            ["20260101120000", f"https://{mirror}", "NEW", "200"],
        ]
    }
    session = _FakeSession(captures, {})

    snapshots, _ = universe_backfill.list_archived_snapshots(
        "nifty500",
        since=date(2023, 1, 1),
        until=date(2025, 1, 1),
        session=session,
    )

    assert [s.digest for s in snapshots] == ["MID"]


def test_backfill_writes_dated_snapshots_and_skips_unchanged_membership(
    tmp_path,
) -> None:
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "D1", "200"],
            ["20240401120000", f"https://{mirror}", "D2", "200"],
            ["20240701120000", f"https://{mirror}", "D3", "200"],
        ]
    }
    bodies = {
        _replay("20240101120000", mirror): _csv(["AAA", "BBB"]),
        # Different bytes (reordered), same membership: not a new boundary.
        _replay("20240401120000", mirror): _csv(["BBB", "AAA"]),
        _replay("20240701120000", mirror): _csv(["BBB", "CCC"]),
    }
    output = tmp_path / "snapshots.csv"
    session = _FakeSession(captures, bodies)

    result = universe_backfill.backfill_snapshots(
        "nifty500", output=output, session=session, min_symbols=1
    )

    assert [s.observed for s in result.snapshots] == [
        date(2024, 1, 1),
        date(2024, 7, 1),
    ]
    frame = pd.read_csv(output, dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01", "2024-07-01"}
    assert set(frame.loc[frame["effective_date"] == "2024-07-01", "symbol"]) == {
        "BBB.NS",
        "CCC.NS",
    }


def test_backfill_rejects_a_truncated_crawl(tmp_path) -> None:
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "D1", "200"],
            ["20240401120000", f"https://{mirror}", "D2", "200"],
        ]
    }
    bodies = {
        _replay("20240101120000", mirror): _csv(["AAA", "BBB", "CCC"]),
        # A partial capture would otherwise erase the index for its whole window.
        _replay("20240401120000", mirror): _csv(["AAA"]),
    }
    output = tmp_path / "snapshots.csv"
    session = _FakeSession(captures, bodies)

    result = universe_backfill.backfill_snapshots(
        "nifty500", output=output, session=session, min_symbols=3
    )

    assert [s.observed for s in result.snapshots] == [date(2024, 1, 1)]
    assert any("min_symbols" in warning for warning in result.warnings)


def test_backfill_preserves_rows_at_dates_it_did_not_produce(tmp_path) -> None:
    output = tmp_path / "snapshots.csv"
    pd.DataFrame(
        {"effective_date": ["2026-01-01", "2026-01-01"], "symbol": ["ZZZ.NS", "YYY.NS"]}
    ).to_csv(output, index=False)
    mirror = _NIFTY500_MIRRORS[0]
    captures = {mirror: [["20240101120000", f"https://{mirror}", "D1", "200"]]}
    bodies = {_replay("20240101120000", mirror): _csv(["AAA"])}
    session = _FakeSession(captures, bodies)

    universe_backfill.backfill_snapshots(
        "nifty500", output=output, session=session, min_symbols=1
    )

    frame = pd.read_csv(output, dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01", "2026-01-01"}
    assert set(frame.loc[frame["effective_date"] == "2026-01-01", "symbol"]) == {
        "YYY.NS",
        "ZZZ.NS",
    }


def test_backfill_raises_when_no_crawl_is_usable(tmp_path) -> None:
    session = _FakeSession({}, {})

    with pytest.raises(RuntimeError, match="no usable archived snapshots"):
        universe_backfill.backfill_snapshots(
            "nifty500", output=tmp_path / "snapshots.csv", session=session
        )


def test_unknown_universe_names_the_available_sources() -> None:
    with pytest.raises(ValueError, match="nifty500"):
        universe_backfill.get_backfill_source("sensex")


def test_backfill_cli_reports_snapshot_dates_and_counts(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    from screener.cli import cli

    mirror = _NIFTY500_MIRRORS[0]
    captures = {mirror: [["20240101120000", f"https://{mirror}", "D1", "200"]]}
    bodies = {_replay("20240101120000", mirror): _csv(["AAA", "BBB"])}
    session = _FakeSession(captures, bodies)
    monkeypatch.setattr(universe_backfill.requests, "Session", lambda: session)
    output = tmp_path / "snapshots.csv"

    result = CliRunner().invoke(
        cli,
        [
            "universes",
            "backfill",
            "nifty500",
            "--output",
            str(output),
            "--min-symbols",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2024-01-01" in result.output
    assert "2 distinct symbols" in result.output
    assert output.exists()


def test_backfill_cli_reports_a_dead_archive_as_a_clean_error(
    monkeypatch, tmp_path
) -> None:
    from click.testing import CliRunner

    from screener.cli import cli

    monkeypatch.setattr(
        universe_backfill.requests, "Session", lambda: _FakeSession({}, {})
    )

    result = CliRunner().invoke(
        cli,
        [
            "universes",
            "backfill",
            "nifty500",
            "--output",
            str(tmp_path / "snapshots.csv"),
        ],
    )

    assert result.exit_code != 0
    assert "no usable archived snapshots" in result.output


def test_backfill_keeps_an_existing_snapshot_for_a_date_it_also_crawled(
    tmp_path,
) -> None:
    """A live ``sync`` row is first-hand; an archived crawl of that day is not."""
    output = tmp_path / "snapshots.csv"
    pd.DataFrame({"effective_date": ["2024-01-01"], "symbol": ["LIVE.NS"]}).to_csv(
        output, index=False
    )
    mirror = _NIFTY500_MIRRORS[0]
    captures = {mirror: [["20240101120000", f"https://{mirror}", "D1", "200"]]}
    bodies = {_replay("20240101120000", mirror): _csv(["AAA"])}
    session = _FakeSession(captures, bodies)

    result = universe_backfill.backfill_snapshots(
        "nifty500", output=output, session=session, min_symbols=1
    )

    frame = pd.read_csv(output, dtype=str)
    assert set(frame["symbol"]) == {"LIVE.NS"}
    assert any("kept the existing snapshot" in warning for warning in result.warnings)


def test_replace_existing_overwrites_a_conflicting_date(tmp_path) -> None:
    output = tmp_path / "snapshots.csv"
    pd.DataFrame({"effective_date": ["2024-01-01"], "symbol": ["OLD.NS"]}).to_csv(
        output, index=False
    )
    mirror = _NIFTY500_MIRRORS[0]
    captures = {mirror: [["20240101120000", f"https://{mirror}", "D1", "200"]]}
    bodies = {_replay("20240101120000", mirror): _csv(["AAA"])}
    session = _FakeSession(captures, bodies)

    universe_backfill.backfill_snapshots(
        "nifty500",
        output=output,
        session=session,
        min_symbols=1,
        replace_existing=True,
    )

    frame = pd.read_csv(output, dtype=str)
    assert set(frame["symbol"]) == {"AAA.NS"}


def test_min_symbols_defaults_to_the_index_floor(tmp_path) -> None:
    """The Nifty 500 floor must reject a two-symbol crawl without being asked."""
    mirror = _NIFTY500_MIRRORS[0]
    captures = {mirror: [["20240101120000", f"https://{mirror}", "D1", "200"]]}
    bodies = {_replay("20240101120000", mirror): _csv(["AAA", "BBB"])}
    session = _FakeSession(captures, bodies)

    with pytest.raises(RuntimeError, match="no usable archived snapshots"):
        universe_backfill.backfill_snapshots(
            "nifty500", output=tmp_path / "snapshots.csv", session=session
        )


def test_min_symbols_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="min_symbols must be at least 1"):
        universe_backfill.backfill_snapshots(
            "nifty500",
            output=tmp_path / "snapshots.csv",
            session=_FakeSession({}, {}),
            min_symbols=0,
        )


def test_a_timestamp_tie_is_resolved_the_same_way_whatever_the_source_order() -> None:
    """Two mirrors crawled in the same second must not depend on list order.

    The winner is arbitrary, but it has to be stable: editing BACKFILL_SOURCES
    must not silently change which membership a date carries.
    """
    mirror_a, mirror_b = _NIFTY500_MIRRORS[0], _NIFTY500_MIRRORS[1]
    captures = {
        mirror_a: [["20240601120000", f"https://{mirror_a}", "DIGEST_A", "200"]],
        mirror_b: [["20240601120000", f"https://{mirror_b}", "DIGEST_B", "200"]],
    }
    forward, _ = universe_backfill.list_archived_snapshots(
        "nifty500", session=_FakeSession(captures, {})
    )
    reversed_captures = dict(reversed(list(captures.items())))
    backward, _ = universe_backfill.list_archived_snapshots(
        "nifty500", session=_FakeSession(reversed_captures, {})
    )

    assert len(forward) == 1
    assert [s.digest for s in forward] == [s.digest for s in backward]


def test_replace_existing_clears_a_date_the_crawl_saw_but_deduped_away(
    tmp_path,
) -> None:
    """A date dropped as unchanged still has its membership from the archive.

    Its window comes from the earlier snapshot, so a stale row left behind
    would reopen a membership the archive says had already ended.
    """
    output = tmp_path / "snapshots.csv"
    pd.DataFrame({"effective_date": ["2024-02-01"], "symbol": ["OLD.NS"]}).to_csv(
        output, index=False
    )
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "D1", "200"],
            ["20240201120000", f"https://{mirror}", "D2", "200"],
        ]
    }
    # Distinct bytes, same membership: the second snapshot is deduped away.
    bodies = {
        _replay("20240101120000", mirror): _csv(["AAA"]),
        _replay("20240201120000", mirror): _csv(["AAA", "AAA"]),
    }
    session = _FakeSession(captures, bodies)

    universe_backfill.backfill_snapshots(
        "nifty500",
        output=output,
        session=session,
        min_symbols=1,
        replace_existing=True,
    )

    frame = pd.read_csv(output, dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01"}
    assert set(frame["symbol"]) == {"AAA.NS"}


def test_a_crawl_that_replaces_most_of_the_membership_is_rejected(tmp_path) -> None:
    """A different 500-row CSV passes the count floor but is not the index."""
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "D1", "200"],
            ["20240201120000", f"https://{mirror}", "D2", "200"],
        ]
    }
    bodies = {
        _replay("20240101120000", mirror): _csv(["AAA", "BBB", "CCC", "DDD"]),
        _replay("20240201120000", mirror): _csv(["WWW", "XXX", "YYY", "ZZZ"]),
    }
    session = _FakeSession(captures, bodies)

    result = universe_backfill.backfill_snapshots(
        "nifty500", output=tmp_path / "snapshots.csv", session=session, min_symbols=1
    )

    frame = pd.read_csv(tmp_path / "snapshots.csv", dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01"}
    assert any(
        "reads as a different document" in warning for warning in result.warnings
    )


def test_a_real_rebalance_is_not_rejected_as_a_different_document(tmp_path) -> None:
    mirror = _NIFTY500_MIRRORS[0]
    captures = {
        mirror: [
            ["20240101120000", f"https://{mirror}", "D1", "200"],
            ["20240201120000", f"https://{mirror}", "D2", "200"],
        ]
    }
    bodies = {
        _replay("20240101120000", mirror): _csv(["AAA", "BBB", "CCC", "DDD"]),
        _replay("20240201120000", mirror): _csv(["AAA", "BBB", "CCC", "NEW"]),
    }
    session = _FakeSession(captures, bodies)

    universe_backfill.backfill_snapshots(
        "nifty500", output=tmp_path / "snapshots.csv", session=session, min_symbols=1
    )

    frame = pd.read_csv(tmp_path / "snapshots.csv", dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01", "2024-02-01"}
