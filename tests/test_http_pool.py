"""The repo keeps one pooled-session shape; these pin it at both FMP callers."""

from __future__ import annotations

import requests

from screener.backtester import fundamentals
from screener.backtester.data import FMPPriceFetcher
from screener.http_pool import pooled_session


def test_pooled_session_sizes_the_pool_to_the_worker_count():
    """An unsized adapter pools 10, so a 16-worker caller discards connections."""
    session = pooled_session(16)

    adapter = session.get_adapter("https://financialmodelingprep.com")

    assert adapter._pool_maxsize == 16
    assert adapter._pool_connections == 16


def test_pooled_session_configures_the_callers_session_rather_than_replacing_it():
    """An injected session is the caller's transport; it comes back, sized."""
    injected = requests.Session()

    resolved = pooled_session(4, session=injected)

    assert resolved is injected
    assert resolved.get_adapter("https://x")._pool_maxsize == 4


def test_pooled_session_passes_through_a_double_with_no_mount():
    """Only a real session has a pool to size, so a stub is left alone."""

    class StubSession:
        pass

    stub = StubSession()

    assert pooled_session(4, session=stub) is stub  # type: ignore[arg-type]


def test_pooled_session_floors_the_pool_at_one():
    assert pooled_session(0).get_adapter("https://x")._pool_maxsize == 1


def test_both_fmp_fetchers_size_their_pool_the_same_way():
    """The consolidation itself.

    These two hit one host from a ``ThreadPoolExecutor`` and used to disagree:
    the price fetcher shared a sized session, the fundamental fetcher kept one
    per worker thread. Whichever shape wins, they must not diverge again.
    """
    prices = FMPPriceFetcher(api_key="x", max_workers=12)
    facts = fundamentals.FMPFundamentalFetcher(api_key="x", max_workers=12)

    for fetcher in (prices, facts):
        adapter = fetcher.session.get_adapter("https://financialmodelingprep.com")
        assert adapter._pool_maxsize == 12
        assert adapter._pool_connections == 12
