"""Validated sector group catalogue with explicit research membership provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener import _optional
from screener.symbols import tv_to_yf


def _validate_source_url(value: str) -> str:
    """Require an external HTTP source URL without credentials or control characters."""
    decoded = unquote(value)
    if any(c.isspace() for c in value) or any(
        ord(c) < 32 or 127 <= ord(c) <= 159 for c in decoded
    ):
        raise ValueError(
            "Sector catalogue: source URL contains whitespace or control characters"
        )
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        port = parts.port
        if (
            parts.scheme not in {"http", "https"}
            or not host
            or parts.username is not None
            or parts.password is not None
            or "\\" in decoded
            or "%" in parts.netloc
            or port == 0
        ):
            raise ValueError
        try:
            address = ip_address(host)
        except ValueError:
            labels = host.encode("idna").decode("ascii").split(".")
            if (
                len(host) > 253
                or len(labels) < 2
                or not re.fullmatch(r"[a-zA-Z]{2,63}|xn--[a-zA-Z0-9-]+", labels[-1])
                or labels[-1].lower()
                in {
                    "localhost",
                    "local",
                    "localdomain",
                    "internal",
                    "invalid",
                    "test",
                    "lan",
                    "home",
                    "corp",
                    "intranet",
                    "private",
                    "onion",
                    "example",
                }
                or host.lower().endswith(".home.arpa")
                or any(
                    not re.fullmatch(
                        r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label
                    )
                    for label in labels
                )
            ):
                raise ValueError
        else:
            if not address.is_global:
                raise ValueError
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "Sector catalogue: source must be an HTTP(S) URL with a real host and no credentials"
        ) from exc
    return value


class SectorGroup(BaseModel):
    """One editable research basket; overlapping groups are permitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    name: str = Field(min_length=1)
    parent: str = Field(min_length=1)
    kind: str = Field(pattern="^(sector|industry|theme)$")
    video_at: str | None = Field(default=None, pattern=r"^\d+:\d{2}$")
    members: tuple[str, ...] = Field(min_length=1)
    notes: str = ""
    description: str = ""
    drivers: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

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
        if any(
            not re.fullmatch(
                r"NSE:[A-Z0-9]+(?:[_&-][A-Z0-9]+)*|(?:NASDAQ|NYSE|AMEX):[A-Z][A-Z0-9]*(?:-[A-Z])?",
                member,
            )
            for member in value
        ):
            raise ValueError(
                "Sector catalogue: members must be exchange-prefixed NSE or US symbols; "
                "use provider dash notation for US share classes"
            )
        provider_symbols = [
            tv_to_yf(member, "india" if member.startswith("NSE:") else "us")
            for member in value
        ]
        if len(provider_symbols) != len(set(provider_symbols)):
            raise ValueError("Sector catalogue: duplicate provider-normalized members")
        return value

    @field_validator("sources")
    @classmethod
    def validate_group_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validate_source_url(source) for source in value)


class SectorCatalogue(BaseModel):
    """Current curated membership is never a point-in-time constituent history."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = Field(ge=1, le=1)
    market: Literal["india", "us"]
    benchmark: str = Field(min_length=1)
    benchmark_name: str = Field(min_length=1)
    observed_on: date
    source: str
    membership_basis: str = Field(pattern="^curated_research$")
    membership_note: str = Field(min_length=1)
    groups: tuple[SectorGroup, ...] = Field(min_length=1)

    @field_validator("source")
    @classmethod
    def validate_catalogue_source(cls, value: str) -> str:
        return _validate_source_url(value)

    @model_validator(mode="after")
    def validate_unique_groups(self) -> SectorCatalogue:
        ids = [group.id for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Sector catalogue: duplicate group ID")
        exchanges = {"NSE"} if self.market == "india" else {"NASDAQ", "NYSE", "AMEX"}
        for group in self.groups:
            if any(
                member.split(":", 1)[0] not in exchanges for member in group.members
            ):
                raise ValueError(
                    f"Sector catalogue: members in {group.id} must use exchanges for market {self.market}"
                )
        return self


@dataclass(frozen=True)
class SectorSelection:
    """Selected groups preserve catalogue order and reject misspelled IDs."""

    catalogue: SectorCatalogue
    groups: tuple[SectorGroup, ...]


def load_sector_catalogue(
    path: Path | None = None, *, market: str = "india"
) -> SectorCatalogue:
    """Load a market default or replacement; callers check replacement market mismatch."""
    if market not in ("india", "us"):
        raise ValueError("Sector catalogue: unsupported market; expected india or us")
    yaml = _optional.load("yaml")
    filename = "sector_groups.yaml" if market == "india" else "sector_groups_us.yaml"
    try:
        text = (
            path.read_text(encoding="utf-8")
            if path is not None
            else files("screener.sector_rotation")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
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
