"""Configuration file loading for the screener CLI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click
import yaml  # type: ignore[import-untyped]


ConfigMap = dict[str, Any]


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
        raise click.UsageError(f"Could not load config file {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise click.UsageError("Config file must contain a top-level mapping.")
    if not all(isinstance(key, str) for key in loaded):
        raise click.UsageError("Config file keys must be strings.")
    return cast(ConfigMap, loaded)
