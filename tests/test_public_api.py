"""Contract tests for the package's public surface.

``screener.api`` is what an outside codebase imports, so the properties pinned
here are the ones a consumer would notice breaking: the exported names, the
laziness of ``import screener``, the no-side-effect default, and the fact that
nothing internal imports the facade back.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import screener
from screener import api
from screener.screen_workflow import ScreenMode, ScreenOutcome, ScreenRequest

_ROOT = Path(__file__).resolve().parents[1]

_PUBLIC_NAMES = {
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "list_criteria",
    "list_markets",
    "run_screen_workflow",
    "screen",
}


def test_package_reexports_the_documented_public_names() -> None:
    """Every name in the contract resolves through the lazy ``__getattr__``."""
    for name in _PUBLIC_NAMES:
        assert getattr(screener, name) is getattr(api, name)
    assert set(api.__all__) == _PUBLIC_NAMES
    assert set(screener.__all__) == _PUBLIC_NAMES | {"__version__"}


def test_unknown_attribute_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        screener.nope  # type: ignore[attr-defined]


def test_version_is_exposed() -> None:
    assert isinstance(screener.__version__, str)
    assert screener.__version__


def test_bare_import_pulls_in_neither_pandas_nor_the_scanner() -> None:
    """``import screener`` must stay free for transitive dependants.

    An eager re-export in ``__init__`` would put pandas on the import path of
    every consumer, including ones that only read ``__version__``. Run in a
    subprocess because pytest has already imported pandas in this one.
    """
    probe = (
        "import sys; import screener; "
        "print('pandas' in sys.modules, 'screener.scanner' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_bare_import_does_not_pay_for_importlib_metadata() -> None:
    """A bare ``import screener`` must not import ``importlib.metadata``.

    Resolving ``__version__`` eagerly would pull in ``importlib.metadata``
    and ``email.message``, about 20 ms on every import of the package, so
    the lookup stays behind the lazy ``__getattr__``. Run in a subprocess
    because pytest has already imported ``importlib.metadata`` in this one.
    """
    probe = "import sys; import screener; print('importlib.metadata' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_py_typed_marker_ships_with_the_package() -> None:
    """Without it, a downstream mypy reads every import as ``Any``."""
    assert (_ROOT / "screener" / "py.typed").is_file()


def test_list_helpers_report_the_real_registries() -> None:
    criteria = api.list_criteria()
    assert "ema" in criteria
    assert criteria == sorted(criteria)
    assert set(api.list_markets()) == {"india", "us"}


def _capture_request(monkeypatch: pytest.MonkeyPatch) -> list[ScreenRequest]:
    """Swap the workflow for a recorder, so no scan or network is attempted."""
    seen: list[ScreenRequest] = []

    def fake_workflow(request: ScreenRequest) -> ScreenOutcome:
        seen.append(request)
        return ScreenOutcome(
            mode=ScreenMode.CSV,
            market=request.market,
            label="+".join(request.criteria_names),
            total=0,
            df=pd.DataFrame(),
        )

    monkeypatch.setattr(api, "run_screen_workflow", fake_workflow)
    return seen


def test_screen_defaults_are_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``persist=False`` must select the workflow's no-write path.

    ``output_csv=True`` is what makes ``run_screen_workflow`` return before the
    history insert and the report render, so an embedded call touches neither
    ``~/.screener/history.db`` nor the filesystem.
    """
    seen = _capture_request(monkeypatch)
    api.screen()
    (request,) = seen
    assert request.output_csv is True
    assert request.report_path is None
    assert request.open_report is False


def test_screen_defaults_match_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drift here would make the library and the CLI disagree silently."""
    seen = _capture_request(monkeypatch)
    api.screen()
    (request,) = seen
    assert request.market == "us"
    assert request.criteria_names == ("ema",)
    assert request.limit == 50
    assert request.order_by == "setup_score"
    assert request.cache_ttl == "15m"


def test_persist_selects_the_recording_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _capture_request(monkeypatch)
    report = tmp_path / "out.html"
    api.screen(persist=True, report_path=report)
    (request,) = seen
    assert request.output_csv is False
    assert request.report_path == report


def test_a_single_criterion_string_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``criteria="ema"`` is the obvious call; it must not iterate the string."""
    seen = _capture_request(monkeypatch)
    api.screen(criteria="ema")
    (request,) = seen
    assert request.criteria_names == ("ema",)


def test_report_path_without_persist_is_rejected() -> None:
    """Silently dropping the report would look like a broken render."""
    with pytest.raises(ValueError, match="requires persist=True"):
        api.screen(report_path="out.html")


def test_negative_earnings_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        api.screen(earnings_buffer=-1)


def _module_scope_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                imported.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                imported.add(child.module)
            visit(child)

    visit(tree)
    return imported


def test_nothing_internal_imports_the_public_facade() -> None:
    """``api.py`` is the outermost layer: it calls in, nothing calls back out.

    An internal module importing it would invert the layering and risk a cycle
    through ``screener/__init__.py``. Internal code should import the concrete
    module (``screener.screen_workflow``) instead.
    """
    offenders: list[str] = []
    for path in (_ROOT / "screener").rglob("*.py"):
        if path.name in {"api.py", "__init__.py"}:
            continue
        if "screener.api" in _module_scope_imports(path):
            offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"internal modules importing screener.api: {offenders}"


def test_api_does_not_import_click_adapters() -> None:
    """The library seam must not depend on the CLI layer."""
    imports = _module_scope_imports(_ROOT / "screener" / "api.py")
    assert not {m for m in imports if m.startswith("screener.commands")}
    assert "click" not in imports


def test_screen_outcome_carries_the_frame() -> None:
    """``outcome.df`` is the documented result handle in both modes."""
    outcome = ScreenOutcome(
        mode=ScreenMode.CSV,
        market="us",
        label="ema",
        total=1,
        df=pd.DataFrame({"name": ["AAPL"]}),
    )
    assert isinstance(outcome.df, pd.DataFrame)
    assert list(outcome.df["name"]) == ["AAPL"]


def test_public_dataclasses_stay_frozen() -> None:
    """Consumers may hash and share requests; mutation would surprise them."""
    request = ScreenRequest(
        market="us",
        criteria_names=("ema",),
        limit=1,
        order_by="setup_score",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
    )
    with pytest.raises(Exception):
        request.market = "india"  # type: ignore[misc]


def test_dir_lists_the_public_surface() -> None:
    listed: list[Any] = dir(screener)
    assert set(listed) == _PUBLIC_NAMES | {"__version__"}
