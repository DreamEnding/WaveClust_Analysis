from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = Path.cwd()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = CODE_DIR / path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_workspace_path(path_value: str | Path) -> Path:
    """Resolve paths relative to workspace directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return WORKSPACE_DIR / path
