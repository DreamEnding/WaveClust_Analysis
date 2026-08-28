from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Use relative paths based on current package location
PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
# Changed: Use current working directory as workspace root instead of hardcoded path
WORKSPACE_DIR = Path.cwd()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        # Look for config relative to CODE_DIR first
        config_path = CODE_DIR / path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_workspace_path(path_value: str | Path) -> Path:
    """Resolve paths relative to workspace directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    # All relative paths are resolved from current working directory
    return WORKSPACE_DIR / path

