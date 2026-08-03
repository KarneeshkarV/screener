"""Regression checks for the repo's neutral-layer dependency directions.

Three directions are pinned here:

* the neutral trade ledger and the per-feature contracts built on it;
* the transport/provider seam, which every feature calls into and which must
  therefore never call back out into one;
* the screens' domain modules, which sit below their Click adapters and must
  therefore never import one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TRADE_EXTENSION_MODULES = {
    "screener/backtester/models.py": "screener.backtester",
    "screener/strategies/trades.py": "screener.strategies",
    "screener/earnings_backtest/models.py": "screener.earnings_backtest",
    "screener/earnings_backtest/metrics.py": "screener.earnings_backtest",
    "screener/options/bt_models.py": "screener.options",
    "screener/options/cli.py": "screener.options",
}
_FEATURE_PACKAGES = (
    "screener.backtester",
    "screener.strategies",
    "screener.earnings_backtest",
    "screener.options",
)

# Modules every feature calls *into* for configuration, transport, caching and
# resilience. They sit below the features, so an import pointing back up is an
# inversion. ``screener/fmp.py`` is the one that regressed: the module whose
# docstring claims to own "the single transport" for every FMP call imported
# ``load_env_file`` from ``screener/backtester/data.py`` just to resolve an API
# key, so the transport depended on one of its own callers. The loader now
# lives in ``screener/config.py``.
#
# ``screener/cache.py`` is deliberately absent: it resolves cache-area
# directories by importing the modules that own them (``price_cache``,
# ``universes``, ``operator.fetch``) lazily inside functions, which is a
# registry lookup rather than a layering inversion.
_SEAM_MODULES = (
    "screener/config.py",
    "screener/fmp.py",
    "screener/resilience.py",
    "screener/providers.py",
)

# The screens' domain modules. ``screener/commands/`` holds their Click
# adapters, which sit *above* them, so an import pointing at one is an
# inversion. ``screener/minervini.py`` is the one that regressed: it reached
# into ``screener.commands.rs_breakout`` for a TradingView universe loader, so
# one screen's domain module depended on another screen's CLI. The loader now
# lives in ``screener/universes.py``.
_SCREEN_MODULES = (
    "screener/conviction.py",
    "screener/garp.py",
    "screener/minervini.py",
    "screener/rs_breakout.py",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


@pytest.mark.parametrize("relative_path,own_package", _TRADE_EXTENSION_MODULES.items())
def test_trade_extension_modules_do_not_depend_on_other_features(
    relative_path: str, own_package: str
) -> None:
    """Shared contracts may point inward to ledger, never across features."""
    imports = _imports(_ROOT / relative_path)
    cross_feature_imports = {
        module
        for module in imports
        for feature in _FEATURE_PACKAGES
        if module.startswith(feature) and not module.startswith(own_package)
    }
    assert not cross_feature_imports


@pytest.mark.parametrize("relative_path", _SEAM_MODULES)
def test_seam_modules_do_not_import_feature_packages(relative_path: str) -> None:
    """Config/transport/cache/resilience are called into, never out of."""
    imports = _imports(_ROOT / relative_path)
    feature_imports = {
        module
        for module in imports
        for feature in _FEATURE_PACKAGES
        if module.startswith(feature)
    }
    assert not feature_imports


@pytest.mark.parametrize("relative_path", _SCREEN_MODULES)
def test_screen_domain_modules_do_not_import_click_adapters(
    relative_path: str,
) -> None:
    """A screen's signal math never depends on another screen's command layer."""
    imports = _imports(_ROOT / relative_path)
    assert not {module for module in imports if module.startswith("screener.commands")}


def test_fmp_transport_owns_api_key_resolution() -> None:
    """The FMP key comes from neutral config, not from a data-fetch module."""
    imports = _imports(_ROOT / "screener/fmp.py")
    assert "screener.config" in imports
    assert not any(module.startswith("screener.backtester") for module in imports)


def test_neutral_ledger_has_no_feature_dependency() -> None:
    imports = _imports(_ROOT / "screener/ledger.py")
    feature_imports = {
        module
        for module in imports
        for feature in _FEATURE_PACKAGES
        if module.startswith(feature)
    }
    assert not feature_imports
