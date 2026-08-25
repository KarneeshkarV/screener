"""Import guards for the packaging extras.

The default install carries the screen path only. Every other workflow's
third-party dependency sits behind a ``[project.optional-dependencies]`` extra
in ``pyproject.toml``, so that a codebase embedding :mod:`screener.api` does
not drag in plotly, yfinance and a bhavcopy client to run one screen.

A bare ``ModuleNotFoundError: No module named 'plotly'`` tells a consumer
nothing about how to fix it, so the import sites for these route through
:func:`load`, which names the extra instead.

Internal use only. Adding a module here means adding it to an extra in
``pyproject.toml`` too; ``tests/test_optional_extras.py`` pins that the two
agree, and that nothing declared optional is imported on the screen path.
"""

from __future__ import annotations

import importlib
from typing import Any

# Top-level module -> the extra that installs its distribution.
EXTRA_FOR_MODULE: dict[str, str] = {
    "plotly": "report",
    "yfinance": "prices",
    "openscreener": "prices",
    "jugaad_data": "india",
    "libsql_client": "usage",
}


def load(module: str) -> Any:
    """Import ``module``, or raise an ImportError naming the extra to install.

    Only a failure to find ``module``'s own top-level package is rewritten. An
    ImportError raised from *inside* an installed optional dependency is a real
    fault in that package and propagates unchanged, so this never disguises a
    broken plotly as a missing one.
    """
    root = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra = EXTRA_FOR_MODULE.get(root)
        failed_root = (getattr(exc, "name", None) or "").split(".")[0]
        if extra is None or failed_root != root:
            raise
        raise ImportError(
            f"{module} is not installed. It ships with the optional "
            f"{extra!r} extra: install `screener[{extra}]` (or "
            f"`screener[all]`) to use this feature."
        ) from exc


__all__ = ["EXTRA_FOR_MODULE", "load"]
