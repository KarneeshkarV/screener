"""Fixtures and gating for the independent correctness suite.

Network tests are skipped unless ``SCREENER_LIVE_TESTS=1`` is set in the
environment, so the default ``uv run pytest`` stays fully offline.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    """Skip ``@pytest.mark.network`` tests unless explicitly opted in."""
    if os.environ.get("SCREENER_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(
        reason="set SCREENER_LIVE_TESTS=1 to run live network tests"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
