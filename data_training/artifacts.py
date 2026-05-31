"""Artifact profile records for synthetic detector realism."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_io import load_config, require_mapping


@dataclass(frozen=True, slots=True)
class ArtifactOperation:
    """One configurable image artifact operation."""

    name: str
    enabled: bool = True
    probability: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ArtifactOperation":
        data = require_mapping(payload, context="artifact operation")
        if not data.get("name"):
            raise ValueError("artifact operation is missing name.")
        return cls(
            name=str(data["name"]),
            enabled=bool(data.get("enabled", True)),
            probability=float(data.get("probability", 1.0)),
            parameters=dict(data.get("parameters") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "probability": self.probability,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ArtifactProfile:
    """A named set of artifact operations for one generated image."""

    profile_id: str
    description: str = ""
    seed_offset: int = 0
    operations: tuple[ArtifactOperation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ArtifactProfile":
        data = require_mapping(payload, context="artifact profile")
        profile_id = str(data.get("profile_id") or data.get("id") or "")
        if not profile_id:
            raise ValueError("artifact profile is missing profile_id.")
        return cls(
            profile_id=profile_id,
            description=str(data.get("description") or ""),
            seed_offset=int(data.get("seed_offset", 0)),
            operations=tuple(
                ArtifactOperation.from_mapping(item)
                for item in data.get("operations", [])
            ),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "description": self.description,
            "seed_offset": self.seed_offset,
            "operations": [item.to_mapping() for item in self.operations],
            "metadata": dict(self.metadata),
        }


def read_artifact_profiles(path: str | Path) -> tuple[ArtifactProfile, ...]:
    """Read artifact profiles from a JSON/YAML config file."""

    payload = load_config(path)
    profiles = tuple(
        ArtifactProfile.from_mapping(item)
        for item in payload.get("profiles", [])
    )
    if not profiles:
        raise ValueError("artifact config must contain at least one profile.")
    return profiles


def artifact_profile_index(
    profiles: tuple[ArtifactProfile, ...],
) -> dict[str, ArtifactProfile]:
    """Return profiles keyed by ``profile_id``."""

    return {profile.profile_id: profile for profile in profiles}
