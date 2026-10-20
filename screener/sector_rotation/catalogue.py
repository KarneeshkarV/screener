"""Validated sector group catalogue with explicit research membership provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener import _optional


class SectorGroup(BaseModel):
    """One editable research basket; overlapping groups are permitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    name: str = Field(min_length=1)
    parent: str = Field(min_length=1)
    kind: str = Field(pattern="^(sector|industry|theme)$")
    video_at: str = Field(pattern=r"^\d+:\d{2}$")
    members: tuple[str, ...] = Field(min_length=1)
    notes: str = ""

    @field_validator("id")
    @classmethod
    def validate_group_id(cls, value: str) -> str:
        if value in {"benchmark", "search", "parent", "count", "empty"}:
            raise ValueError("Sector catalogue: reserved group ID")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError("Sector catalogue: group IDs must be lowercase snake_case")
        return value

    @field_validator("members")
    @classmethod
    def validate_group_members(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not re.fullmatch(r"NSE:[A-Z0-9_&-]+", member) for member in value
        ):
            raise ValueError("Sector catalogue: members must be unique NSE symbols")
        return value


class SectorCatalogue(BaseModel):
    """Current curated membership is never a point-in-time constituent history."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = Field(ge=1, le=1)
    market: str = Field(pattern="^india$")
    benchmark: str = Field(min_length=1)
    benchmark_name: str = Field(min_length=1)
    observed_on: date
    source: str = Field(pattern=r"^https://www\.youtube\.com/watch\?v=d-I6BG5F-Yg$")
    membership_basis: str = Field(pattern="^curated_research$")
    membership_note: str = Field(min_length=1)
    groups: tuple[SectorGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_groups(self) -> SectorCatalogue:
        ids = [group.id for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Sector catalogue: duplicate group ID")
        return self


@dataclass(frozen=True)
class SectorSelection:
    """Selected groups preserve catalogue order and reject misspelled IDs."""

    catalogue: SectorCatalogue
    groups: tuple[SectorGroup, ...]


def load_sector_catalogue(path: Path | None = None) -> SectorCatalogue:
    """Load the bundled YAML catalogue or a complete user-supplied replacement."""
    yaml = _optional.load("yaml")
    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else files("screener.sector_rotation")
        .joinpath("sector_groups.yaml")
        .read_text(encoding="utf-8")
    )
    try:
        return SectorCatalogue.model_validate(yaml.safe_load(text))
    except Exception as exc:
        raise ValueError(f"Sector catalogue: invalid configuration: {exc}") from exc


def select_sector_groups(
    catalogue: SectorCatalogue, group_ids: tuple[str, ...] = ()
) -> SectorSelection:
    """Select exact group IDs, or all groups if no IDs were requested."""
    unknown = set(group_ids) - {group.id for group in catalogue.groups}
    if unknown:
        raise ValueError(
            f"Sector catalogue: unknown groups: {', '.join(sorted(unknown))}"
        )
    return SectorSelection(
        catalogue,
        tuple(
            group
            for group in catalogue.groups
            if not group_ids or group.id in group_ids
        ),
    )
