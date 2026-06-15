from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_root"] = config_path.resolve().parent.parent
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config["_root"] / path


def env_value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

