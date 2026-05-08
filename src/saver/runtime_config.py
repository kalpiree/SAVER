"""Runtime helpers for lightweight environment-based config overrides."""

from __future__ import annotations

import os
from typing import Any

_EDITOR_ENV_OVERRIDES = {
    "model_name_override": "SAVER_MODEL_NAME_OVERRIDE",
}


def apply_env_editor_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied config with supported env overrides applied."""

    editor_config = config.get("editor")
    if not isinstance(editor_config, dict):
        return config

    resolved_overrides = {}
    for key, env_name in _EDITOR_ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            resolved_overrides[key] = value

    if not resolved_overrides:
        return config

    updated_config = dict(config)
    updated_editor_config = dict(editor_config)
    updated_editor_config.update(resolved_overrides)
    updated_config["editor"] = updated_editor_config
    return updated_config
