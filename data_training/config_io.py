"""Small JSON/YAML helpers for the data-training scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping from ``path``.

    YAML support intentionally stays optional at import time. The EWALD
    environment already includes PyYAML, but keeping the import lazy
    makes the data-training modules cheap to inspect in minimal Python
    sessions.
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Use JSON configs or "
                "install PyYAML."
            ) from exc
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config suffix: {config_path.suffix!r}")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping.")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic, human-readable JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def require_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    """Return ``payload`` as a mapping or raise a clear validation
    error."""

    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping.")
    return payload


def require_sequence(payload: Any, *, context: str) -> list[Any]:
    """Return ``payload`` as a list or raise a clear validation
    error."""

    if not isinstance(payload, list):
        raise ValueError(f"{context} must be a list.")
    return payload


def tuple_of_strings(value: Any) -> tuple[str, ...]:
    """Normalize scalar/list config values to an immutable string
    tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)
