"""Pydantic models for operator CLI and module boundaries."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

OperatorUniverseMode = Literal["fo", "fo+cash"]


class OperatorScanRequest(BaseModel):
    """Validated inputs for ``operator-scan`` (NSE operator intent screen)."""

    as_of: date
    universe: OperatorUniverseMode = "fo+cash"
    out_path: Path | None = None
    only_actions: bool = False
    verbose: bool = False

    model_config = ConfigDict(frozen=True)
