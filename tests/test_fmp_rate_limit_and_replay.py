"""Regression tests for FMP section pacing and aliased ``--from-run`` replay.

Two independent defects share this file because both were invisible to the
suite: the FMP fundamentals fetcher charged one rate-limit slot for five HTTP
calls, and a strategy-aliased criterion could never be replayed from a bare
``MARKET:CRITERIA`` spec.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import pytest

from screener import history as history_mod
from screener.backtester import fundamentals


# --- FMP fundamentals: one resilience call per endpoint ------------------------


@pytest.fixture(autouse=True)
def _reset_fmp_breaker():
    """Leave the module-level ``fmp`` breaker clean for the rest of the suite."""
    yield
    from screener.resilience import get_breaker

    get_breaker("fmp").record_success()


def _record_resilience_calls(monkeypatch) -> list[tuple[str, str]]:
    """Record every ``(provider, operation)`` while keeping real retry logic.

    Sleeps and rate-limit waits are neutralised so the test stays offline and
    instant; retry counts and fallbacks still come from the real helper.
    """
    calls: list[tuple[str, str]] = []
    real = fundamentals.call_with_resilience

    def recording(provider, operation, func, *, fallback, retry=None, **_kwargs):
        calls.append((provider, operation))
        return real(
            provider,
            operation,
            func,
            fallback=fallback,
            retry=retry,
            sleep=lambda _seconds: None,
            rate_limiter=None,
        )

    monkeypatch.setattr(fundamentals, "call_with_resilience", recording)
    return calls


def test_fmp_sections_are_wrapped_and_paced_individually(monkeypatch):
    """Five endpoints must take five resilience slots, not one.

    Wrapping the whole payload meant ``ProviderRateLimiter`` charged one token
    for five HTTP calls (5x the configured FMP rate), which is how a cold
    fetch earned 429s.
    """
    calls = _record_resilience_calls(monkeypatch)
    requested: list[str] = []

    def fake_get(session, path, params, api_key):
        requested.append(path.split("/", 1)[0])
        return [{"date": "2024-01-31"}]

    monkeypatch.setattr(fundamentals, "_fmp_get", fake_get)

    payload = fundamentals._fetch_fmp_sections(
        "AAA",
        api_key="k",
        session=object(),
        limit=8,  # type: ignore[arg-type]
    )

    assert [operation for _provider, operation in calls] == [
        "backtest fundamentals AAA income-statement",
        "backtest fundamentals AAA balance-sheet-statement",
        "backtest fundamentals AAA ratios",
        "backtest fundamentals AAA key-metrics",
        "backtest fundamentals AAA enterprise-values",
    ]
    # Every request shares the provider breaker and rate limiter.
    assert {provider for provider, _operation in calls} == {"fmp"}
    assert requested == [
        "income-statement",
        "balance-sheet-statement",
        "ratios",
        "key-metrics",
        "enterprise-values",
    ]
    assert set(payload) == {
        "income",
        "balance",
        "ratios",
        "key_metrics",
        "enterprise_values",
    }


def test_a_transient_failure_on_the_last_endpoint_retries_only_that_endpoint(
    monkeypatch,
):
    """A failure on endpoint five must not re-fetch endpoints one to four.

    The old single wrapper retried the whole payload, so one transient error on
    ``enterprise-values`` produced up to 15 HTTP requests for one ticker.
    """
    calls = _record_resilience_calls(monkeypatch)
    requested: list[str] = []
    enterprise_attempts = 0

    def fake_get(session, path, params, api_key):
        nonlocal enterprise_attempts
        requested.append(path.split("/", 1)[0])
        if path.startswith("enterprise-values/"):
            enterprise_attempts += 1
            if enterprise_attempts == 1:
                raise RuntimeError("429 Too Many Requests")
        return [{"date": "2024-01-31"}]

    monkeypatch.setattr(fundamentals, "_fmp_get", fake_get)

    payload = fundamentals._fetch_fmp_sections(
        "AAA",
        api_key="k",
        session=object(),
        limit=8,  # type: ignore[arg-type]
    )

    assert len(calls) == 5
    assert requested == [
        "income-statement",
        "balance-sheet-statement",
        "ratios",
        "key-metrics",
        "enterprise-values",
        "enterprise-values",
    ]
    assert payload["enterprise_values"] == [{"date": "2024-01-31"}]


def test_an_exhausted_endpoint_fails_the_ticker_instead_of_caching_partials(
    monkeypatch,
):
    """A section that never answers must not be cached as if it were empty.

    The provider seam serves a stale entry (or the empty fallback) on failure;
    a partial payload would instead be persisted for a full TTL as authoritative
    data, silently dropping the failed section's fundamentals.
    """
    _record_resilience_calls(monkeypatch)
    requested: list[str] = []

    def fake_get(session, path, params, api_key):
        requested.append(path.split("/", 1)[0])
        if path.startswith("ratios/"):
            raise RuntimeError("upstream is down")
        return [{"date": "2024-01-31"}]

    monkeypatch.setattr(fundamentals, "_fmp_get", fake_get)

    with pytest.raises(RuntimeError, match="FMP ratios request failed for AAA"):
        fundamentals._fetch_fmp_sections(
            "AAA",
            api_key="k",
            session=object(),
            limit=8,  # type: ignore[arg-type]
        )

    # Default retry budget is three attempts for the failing endpoint alone.
    assert requested == [
        "income-statement",
        "balance-sheet-statement",
        "ratios",
        "ratios",
        "ratios",
    ]


# --- History: bare criteria fall back to their persisted alias -----------------


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


def _seed_run(market: str, criteria: str, tickers: list[str], run_date: date) -> int:
    frame = pd.DataFrame(
        {"name": tickers, "close": [10.0 + i for i in range(len(tickers))]}
    )
    run_id = history_mod.save_run(market, criteria, len(tickers), frame)
    conn = sqlite3.connect(history_mod.DB_PATH)
    conn.execute(
        "UPDATE runs SET run_ts = ? WHERE id = ?",
        (f"{run_date.isoformat()}T16:00:00+00:00", run_id),
    )
    conn.commit()
    conn.close()
    return run_id


def test_bare_criteria_find_the_default_universe_alias(history_db):
    """``--from-run india:ema`` must reach the row the daily screen wrote.

    Strategy-aliased criteria persist as ``ema@tv#<fingerprint>``, so an exact
    lookup finds nothing and the replay reports the same "not old enough"
    failure as a genuinely missing run.
    """
    aliased = _seed_run("india", "ema@tv#deadbeef", ["AAA", "BBB"], date(2024, 3, 1))

    snap = history_mod.resolve_replay_run("india:ema")

    assert snap.run_id == aliased
    assert snap.criteria == "ema@tv#deadbeef"
    assert snap.tickers == ["AAA", "BBB"]


def test_an_exact_run_always_wins_over_its_aliases(history_db):
    """The exact label is tried first, so ordering by alias cannot shadow it."""
    bare = _seed_run("india", "ema", ["OLD"], date(2024, 2, 1))
    _seed_run("india", "ema@tv#deadbeef", ["NEW"], date(2024, 4, 1))

    assert history_mod.find_run("india", "ema") == bare
    assert history_mod.resolve_replay_run("india:ema").run_id == bare


def test_a_newer_default_universe_alias_wins_over_older_ones(history_db):
    """Among aliases, the newest eligible fingerprint is the one replayed."""
    _seed_run("india", "ema@tv#deadbeef", ["OLD"], date(2024, 1, 1))
    newest = _seed_run("india", "ema@tv#12345678", ["NEW"], date(2024, 2, 1))

    assert history_mod.find_run("india", "ema") == newest


def test_a_criterion_with_underscores_is_matched_literally(history_db):
    """``momentum_12_1`` must not treat its ``_`` as a single-char wildcard."""
    aliased = _seed_run("us", "momentum_12_1@tv#deadbeef", ["AAA"], date(2024, 3, 1))

    assert history_mod.find_run("us", "momentum_12_1") == aliased
    assert history_mod.find_run("us", "momentum-12-1") is None


def test_an_alias_spec_is_matched_strictly_and_never_widened(history_db):
    """A spec that names a fingerprint must not fall back to another one."""
    _seed_run("india", "ema@tv#deadbeef", ["AAA"], date(2024, 3, 1))

    with pytest.raises(ValueError, match="no persisted india/ema@tv#12345678"):
        history_mod.resolve_replay_run("india:ema@tv#12345678")

    assert history_mod.find_run("india", "ema@tv#dead") is None


def test_replay_age_filters_the_alias_like_an_exact_match(history_db):
    """``--run-age-days`` applies to alias candidates too, not just exact rows."""
    recent = _seed_run("india", "ema@tv#deadbeef", ["RECENT"], date(2099, 1, 1))

    with pytest.raises(ValueError, match="at least 7 days old"):
        history_mod.resolve_replay_run("india:ema", min_age_days=7)

    assert history_mod.find_run("india", "ema", on_or_before=date(2100, 1, 1)) == recent


def test_previous_run_never_crosses_fingerprints(history_db):
    """Screen-to-screen diffing keeps its exact-label contract (D28).

    ``previous_run`` is what the screen uses for added/removed; widening the
    ``--from-run`` lookup must not widen it, or a run made with different
    gates would be diffed against this one.
    """
    older = _seed_run("india", "ema@tv#deadbeef", ["AAA"], date(2024, 2, 1))
    newer = _seed_run("india", "ema@tv#deadbeef", ["BBB"], date(2024, 3, 1))
    other = _seed_run("india", "ema@tv#12345678", ["CCC"], date(2024, 3, 2))

    same_fingerprint = history_mod.previous_run("india", "ema@tv#deadbeef", newer)
    assert same_fingerprint is not None
    assert same_fingerprint["ticker"].tolist() == ["AAA"]

    cross_fingerprint = history_mod.previous_run("india", "ema@tv#deadbeef", other)
    assert cross_fingerprint is not None
    assert cross_fingerprint["ticker"].tolist() == ["BBB"]
    assert older < newer < other
