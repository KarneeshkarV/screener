"""Shared reporting mechanics for the CLI output sinks.

Two families of helpers live here:

* HTML report helpers (`temp_report_path`, `open_report`).
* The genuinely-repeated JSON/Markdown *mechanics* shared by the scan output
  modules (`json_safe`, `dump_json_file`, `markdown_row`). These centralise
  behaviour that was copied across ``unusual_volume.output`` and
  ``rs_breakout`` so the exact ``json.dumps`` kwargs and JSON-safe coercions
  live (and are tested) in one place. Domain-specific table *layouts* stay in
  their own modules — only the leaf mechanics are shared.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import webbrowser
from collections.abc import Sequence
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serialisable form.

    * ``None`` and missing values (anything ``pd.isna`` flags, incl. ``NaN``,
      ``pd.NA``, ``NaT``) become ``None``.
    * dicts/lists/tuples are walked recursively (tuples become lists).
    * Real numbers (numpy scalars included, but *not* ``bool``) are narrowed to
      ``int``/``float``; non-finite floats (``inf``/``-inf``) become ``None``.
    * Everything else is returned unchanged.

    ``pd.isna`` on array-like values raises; those are left untouched so a
    stray Series/array flows through to ``json.dumps``' ``default`` handler.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Real) and not isinstance(value, bool):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        if isinstance(value, Integral):
            return int(value)
        return as_float
    return value


def dump_json_file(payload: Any, path: str | Path, *, allow_nan: bool = True) -> None:
    """Write ``payload`` to ``path`` as JSON with the shared dump conventions.

    Uses the exact kwargs the scan output writers rely on: ``indent=2`` and
    ``default=str`` (so dates/Decimals serialise). ``allow_nan`` is exposed
    because the unusual-volume writer pre-sanitises via :func:`json_safe` and
    then asserts finiteness with ``allow_nan=False``, while the RS-breakout
    writer serialises an already-clean pydantic dump with the JSON default.
    No trailing newline is appended (matches the prior behaviour).
    """
    Path(path).write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=allow_nan)
    )


def markdown_row(cells: Sequence[str]) -> str:
    """Build one GitHub-flavoured Markdown table row from pre-formatted cells.

    Returns ``"| a | b | c |"``. Callers own column formatting/alignment; this
    only owns the pipe/spacing convention that was duplicated across writers.
    """
    return "| " + " | ".join(cells) + " |"


def temp_report_path(prefix: str) -> Path:
    """Return a timestamped report path under the system temp directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = Path(tempfile.gettempdir()) / "screener-reports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}-{stamp}.html"


def open_report(path: str | Path) -> None:
    """Open a report in the default browser."""
    webbrowser.open(Path(path).resolve().as_uri())


def _is_wsl() -> bool:
    """Detect WSL from the kernel release string (WSL1 and WSL2 both tag it)."""
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text()
    except OSError:
        return False
    return "microsoft" in release.lower()


def windows_report_path(path: str | Path) -> str | None:
    """Return the Windows-visible path for ``path``, or ``None`` when there is none.

    A Linux path such as ``/tmp/screener-reports/x.html`` cannot be opened by a
    Windows browser, so on WSL the CLI prints the UNC form
    (``\\\\wsl.localhost\\<distro>\\tmp\\...``) beside it. Paths already under a
    Windows drive mount (``/mnt/c/...``) map to the plain drive letter instead.
    """
    if not _is_wsl():
        return None
    resolved = Path(path).resolve()
    parts = resolved.parts
    if len(parts) > 2 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = f"{parts[2].upper()}:"
        return "\\".join([drive, *parts[3:]]) if len(parts) > 3 else f"{drive}\\"
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return None
    return "\\\\wsl.localhost\\" + distro + "\\" + "\\".join(parts[1:])


def report_path_lines(path: str | Path | None) -> list[str]:
    """Return the ``Report:`` line, plus the Windows path line when on WSL."""
    lines = [f"Report: {path}"]
    windows = windows_report_path(path) if path is not None else None
    if windows:
        lines.append(f"Windows: {windows}")
    return lines
