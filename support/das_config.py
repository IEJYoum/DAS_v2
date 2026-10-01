"""Local DAS preferences stored beside the code, never in project configs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from shared_utils import load_key_value_config, write_key_value_config


DAS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "das_config.txt"
DAS_CONFIG_HEADER = "# DAS_v2 local preferences (this file is gitignored)"


def load_das_config(path: str | Path | None = None) -> dict[str, str]:
    """Read machine-local preferences without creating or prompting for them."""
    return load_key_value_config(Path(path) if path is not None else DAS_CONFIG_PATH)


def load_pixel_size_um(path: str | Path | None = None) -> float | None:
    """Return a positive configured pixel size without prompting for one."""
    try:
        value = float(load_das_config(path).get("pixel_size_um", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def save_das_config_updates(
    updates: dict[str, Any],
    path: str | Path | None = None,
) -> dict[str, str]:
    """Update local preferences while preserving unrelated future settings."""
    config_path = Path(path) if path is not None else DAS_CONFIG_PATH
    values = load_das_config(config_path)
    for key, value in updates.items():
        text = str(value).strip()
        if text == "":
            values.pop(str(key), None)
        else:
            values[str(key)] = text
    write_key_value_config(config_path, values, header=DAS_CONFIG_HEADER)
    return values
