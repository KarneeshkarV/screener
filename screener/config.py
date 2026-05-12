"""Configuration file loading for the screener CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml  # type: ignore[import-untyped]


ConfigMap = dict[str, Any]


class CliConfig(BaseModel):
    log_level: str | None = None
    log_json: bool | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _validate_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("Config file must contain a top-level mapping.")
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Config file keys must be strings.")
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("log_level must not be empty.")
        return normalized

    def to_click_default_map(self) -> ConfigMap:
        return self.model_dump(exclude_none=True, mode="python")


def load_config(path: str | Path) -> ConfigMap:
    """Load a YAML or JSON config file as a Click default map."""
    config_path = Path(path)
    if not config_path.exists():
        raise click.UsageError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise click.UsageError(f"Config path is not a file: {config_path}")

    suffix = config_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(config_path.read_text()) or {}
        elif suffix == ".json":
            loaded = json.loads(config_path.read_text())
        else:
            raise click.UsageError(
                "Unsupported config file extension. Use .yaml, .yml, or .json."
            )
    except click.UsageError:
        raise
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise click.UsageError(
            f"Could not load config file {config_path}: {exc}"
        ) from exc

    try:
        return CliConfig.model_validate(loaded).to_click_default_map()
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise click.UsageError(message) from exc
