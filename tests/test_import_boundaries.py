"""Regression checks for neutral trade-ledger dependency direction."""

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


def test_neutral_ledger_has_no_feature_dependency() -> None:
    imports = _imports(_ROOT / "screener/ledger.py")
    feature_imports = {
        module
        for module in imports
        for feature in _FEATURE_PACKAGES
        if module.startswith(feature)
    }
    assert not feature_imports
