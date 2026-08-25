"""Contract tests for the packaging extras.

The default install is the screen path only. These pin the three ways that
promise can rot: an extra and its guard disagreeing, an optional dependency
creeping back onto the screen path, and an import site bypassing the guard so
a consumer sees a bare ``ModuleNotFoundError`` instead of the extra's name.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from screener import _optional

_ROOT = Path(__file__).resolve().parents[1]

# Import name -> distribution name on the index. They differ for the two
# packages that use a dash.
_DISTRIBUTION = {
    "plotly": "plotly",
    "yfinance": "yfinance",
    "openscreener": "openscreener",
    "jugaad_data": "jugaad-data",
    "libsql_client": "libsql-client",
}


def _pyproject() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_name(spec: str) -> str:
    """``"jugaad-data>=0.33.1"`` and ``"x @ git+..."`` -> the bare name."""
    for separator in (" @ ", ">=", "==", "~=", ">", "<", "["):
        spec = spec.split(separator)[0]
    return spec.strip()


def test_every_guarded_module_is_declared_in_its_extra() -> None:
    """``_optional.EXTRA_FOR_MODULE`` and pyproject must name the same sets."""
    extras = _pyproject()["project"]["optional-dependencies"]
    for module, extra in _optional.EXTRA_FOR_MODULE.items():
        assert extra in extras, f"{module} points at undeclared extra {extra!r}"
        declared = {_requirement_name(spec) for spec in extras[extra]}
        assert _DISTRIBUTION[module] in declared, (
            f"{_DISTRIBUTION[module]} is guarded as extra {extra!r} "
            f"but is not declared in it"
        )


def test_no_optional_dependency_sits_in_the_core_requirements() -> None:
    """A guarded dependency in ``dependencies`` would defeat the whole split."""
    core = {_requirement_name(spec) for spec in _pyproject()["project"]["dependencies"]}
    guarded = {_DISTRIBUTION[module] for module in _optional.EXTRA_FOR_MODULE}
    assert not (core & guarded)


def test_the_all_extra_covers_every_other_extra() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    referenced = {_requirement_name(spec) for spec in extras["all"]}
    # `screener[report]` -> "screener" after stripping at "[".
    assert referenced == {"screener"}
    covered = {spec.split("[")[1].rstrip("]") for spec in extras["all"]}
    assert covered == set(extras) - {"all"}


@pytest.mark.parametrize("module", sorted(_optional.EXTRA_FOR_MODULE))
def test_optional_dependencies_stay_off_the_screen_path(module: str) -> None:
    """The default install must be able to run a screen.

    Importing the workflow is what a consumer's ``from screener import screen``
    does, so anything loaded here is effectively a core dependency regardless
    of where pyproject files it. Run in a subprocess: pytest's own imports have
    already pulled several of these into this interpreter.
    """
    probe = (
        f"import sys; import screener.screen_workflow; print({module!r} in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", (
        f"{module} loads on the screen path but is declared optional "
        f"({_optional.EXTRA_FOR_MODULE[module]} extra)"
    )


def _runtime_imports(path: Path) -> set[str]:
    """Imports that actually execute, ignoring ``if TYPE_CHECKING:`` bodies.

    The module-level guards are written as a ``TYPE_CHECKING`` import paired
    with an ``else:`` branch holding the ``_optional.load`` call, so a checker
    keeps the real module's types while the runtime goes through the guard.
    Only the ``else:`` branch runs, so only it is scanned here.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for stmt in child.orelse:
                    visit(stmt)
                continue
            if isinstance(child, ast.Import):
                found.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.add(child.module)
            visit(child)

    visit(tree)
    return found


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def test_no_import_site_bypasses_the_guard() -> None:
    """Every optional dependency is reached through ``_optional.load``.

    A plain ``import yfinance`` anywhere would raise a bare
    ``ModuleNotFoundError`` on a default install, telling the consumer nothing
    about ``screener[prices]``.
    """
    offenders: list[str] = []
    for path in (_ROOT / "screener").rglob("*.py"):
        if path.name == "_optional.py":
            continue
        for imported in _runtime_imports(path):
            root = imported.split(".")[0]
            if root in _optional.EXTRA_FOR_MODULE:
                offenders.append(f"{path.relative_to(_ROOT)}: {imported}")
    assert not offenders, f"ungated optional imports: {offenders}"


def test_missing_module_names_the_extra_to_install() -> None:
    _optional.EXTRA_FOR_MODULE["definitely_not_installed_xyz"] = "prices"
    try:
        with pytest.raises(ImportError, match=r"screener\[prices\]"):
            _optional.load("definitely_not_installed_xyz")
    finally:
        del _optional.EXTRA_FOR_MODULE["definitely_not_installed_xyz"]


def test_an_unguarded_module_propagates_unchanged() -> None:
    with pytest.raises(ImportError) as excinfo:
        _optional.load("some_module_nobody_declared")
    assert "extra" not in str(excinfo.value)


def test_a_fault_inside_an_installed_extra_is_not_disguised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ImportError raised *by* an installed dependency is a real bug.

    Rewriting it to "install the extra" would send someone chasing a packaging
    problem that does not exist.
    """
    package = tmp_path / "fake_extra_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("import a_module_that_is_not_there\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(_optional.EXTRA_FOR_MODULE, "fake_extra_pkg", "report")

    with pytest.raises(ImportError) as excinfo:
        _optional.load("fake_extra_pkg")
    assert "a_module_that_is_not_there" in str(excinfo.value)
    assert "screener[report]" not in str(excinfo.value)


def test_load_returns_the_real_module() -> None:
    assert _optional.load("json").dumps({"a": 1}) == '{"a": 1}'
