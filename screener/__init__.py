"""Screening, ranking, and backtesting for US and Indian equities.

The supported surface for outside callers is re-exported here and defined in
:mod:`screener.api`; see that module's docstring for the usage contract.
Everything else under ``screener.`` is internal.

Attribute access is lazy (PEP 562), so a bare ``import screener`` imports no
pandas and no scanner. ``__version__`` is deferred the same way: resolving it
eagerly imports :mod:`importlib.metadata`, and with it :mod:`email.message`,
which costs about 20 ms on every import of the package - including one that
only depends on it transitively and never reads the version at all. Touching
any other name below loads :mod:`screener.api` and with it the screen
workflow. Keep the laziness when adding exports; an eager re-export here
would put pandas on every import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from screener.api import (
        ScreenMode,
        ScreenOutcome,
        ScreenRequest,
        SignalRow,
        StaleDataError,
        list_criteria,
        list_markets,
        list_universes,
        run_screen_workflow,
        screen,
    )


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("screener")
    except PackageNotFoundError:  # running from a source tree, not installed
        return "0.0.0+unknown"


_EXPORTS = frozenset(
    {
        "DEFAULT_INTERVAL",
        "IntervalNotScreenableError",
        "ScreenMode",
        "ScreenOutcome",
        "ScreenRequest",
        "SignalRow",
        "StaleDataError",
        "StrategyProfile",
        "UnscreenableStrategyError",
        "list_criteria",
        "list_markets",
        "list_universes",
        "run_screen_workflow",
        "screen",
    }
)


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return _version()
    if name in _EXPORTS:
        import screener.api

        return getattr(screener.api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_EXPORTS | {"__version__"})


__all__ = [
    "DEFAULT_INTERVAL",
    "IntervalNotScreenableError",
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "SignalRow",
    "StaleDataError",
    "StrategyProfile",
    "UnscreenableStrategyError",
    "__version__",
    "list_criteria",
    "list_markets",
    "list_universes",
    "run_screen_workflow",
    "screen",
]
