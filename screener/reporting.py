"""Shared helpers for CLI-generated HTML reports."""

from __future__ import annotations

import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path


def temp_report_path(prefix: str) -> Path:
    """Return a timestamped report path under the system temp directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = Path(tempfile.gettempdir()) / "screener-reports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}-{stamp}.html"


def open_report(path: str | Path) -> None:
    """Open a report in the default browser."""
    webbrowser.open(Path(path).resolve().as_uri())
