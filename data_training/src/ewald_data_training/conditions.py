"""Condition sweep expansion for training-data generation."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .catalog import load_structure_catalog
from .schemas import (
    ArtifactProfile,
    ConfigError,
    DetectorGeometry,
    SimulationCondition,
    StructureRecord,
)


def load_artifact_profiles(path: str | Path) -> dict[str, ArtifactProfile]:
    """Load artifact profiles from YAML."""

    artifact_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(artifact_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("artifact_profiles", payload)
    if isinstance(entries, Mapping):
        entries = [
            {"profile_id": key, **(value or {})}
            for key, value in entries.items()
        ]
    if not isinstance(entries, list):
        raise ConfigError("Artifact profiles must be a list or mapping.")
    profiles = [ArtifactProfile.from_mapping(entry) for entry in entries]
    return {profile.profile_id: profile for profile in profiles}


def load_generation_plan(path: str | Path) -> dict[str, Any]:
    """Load a full dataset generation plan."""

    plan_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    detector_entries = _load_detector_geometries(payload)
    detector = detector_entries[0][1]
    catalog_path = _resolve_from(plan_path, payload.get("structures", ""))
    structures = load_structure_catalog(catalog_path)
    artifact_path = payload.get("artifacts")
    artifacts = {}
    if artifact_path:
        artifacts = load_artifact_profiles(
            _resolve_from(plan_path, artifact_path)
        )
    conditions: list[SimulationCondition] = []
    for detector_index, (detector_id, detector_geometry) in enumerate(
        detector_entries
    ):
        detector_conditions = expand_conditions(
            payload.get("sweep", {}) or {},
            detector=detector_geometry,
            default_artifact_profile_id=payload.get(
                "default_artifact_profile_id", "default"
            ),
        )
        for condition in detector_conditions:
            condition.metadata.update(
                {
                    "detector_sweep_id": detector_id,
                    "detector_sweep_index": detector_index,
                }
            )
        conditions.extend(detector_conditions)
    output_root = _resolve_from(plan_path, payload.get("output_root", "runs"))
    return {
        "plan_path": plan_path,
        "dataset": str(payload.get("dataset", plan_path.stem)),
        "output_root": output_root,
        "structures_path": catalog_path,
        "structures": structures,
        "detector": detector,
        "detectors": [entry[1] for entry in detector_entries],
        "artifact_profiles": artifacts,
        "conditions": conditions,
        "metadata": dict(payload.get("metadata", {}) or {}),
    }


def _load_detector_geometries(
    payload: Mapping[str, Any],
) -> list[tuple[str, DetectorGeometry]]:
    """Return one or more detector geometries from a generation plan."""

    if "detectors" not in payload:
        return [
            (
                str(
                    (payload.get("detector", {}) or {}).get(
                        "detector_id",
                        (payload.get("detector", {}) or {}).get(
                            "detector", "detector_0"
                        ),
                    )
                ),
                DetectorGeometry.from_mapping(
                    payload.get("detector", {}) or {}
                ),
            )
        ]

    entries = payload.get("detectors") or []
    if isinstance(entries, Mapping):
        entries = [
            {"detector_id": key, **(value or {})}
            for key, value in entries.items()
        ]
    if not isinstance(entries, list) or not entries:
        raise ConfigError("detectors must be a non-empty list or mapping.")

    detectors: list[tuple[str, DetectorGeometry]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ConfigError("Each detector entry must be a mapping.")
        detector_id = str(
            entry.get("detector_id")
            or entry.get("name")
            or entry.get("detector")
            or f"detector_{index}"
        )
        detectors.append((detector_id, DetectorGeometry.from_mapping(entry)))
    return detectors


def expand_conditions(
    sweep: Mapping[str, Any],
    *,
    detector: DetectorGeometry | None = None,
    default_artifact_profile_id: str = "default",
) -> list[SimulationCondition]:
    """Expand a compact sweep mapping into deterministic conditions."""

    if not sweep:
        sweep = {}
    geometry = detector or DetectorGeometry()
    keys = [
        "theta_x_deg",
        "theta_y_deg",
        "sigma_theta",
        "sigma_phi",
        "sigma_r",
        "q_dependent_sigma_r",
        "q_dependent_sigma_z",
        "hkl_extent",
        "orientation_label",
        "texture_model",
        "artifact_profile_id",
        "seed",
    ]
    defaults: dict[str, Any] = {
        "theta_x_deg": 90.0,
        "theta_y_deg": 0.0,
        "sigma_theta": 0.03,
        "sigma_phi": 0.25,
        "sigma_r": 0.035,
        "q_dependent_sigma_r": 0.0,
        "q_dependent_sigma_z": 0.0,
        "hkl_extent": 4,
        "orientation_label": "fiber",
        "texture_model": "fiber_gaussian",
        "artifact_profile_id": default_artifact_profile_id,
        "seed": 0,
    }
    values = [_as_list(sweep.get(key, defaults[key])) for key in keys]
    conditions: list[SimulationCondition] = []
    for index, combination in enumerate(product(*values)):
        payload = dict(zip(keys, combination))
        payload["metadata"] = {"sweep_index": index}
        conditions.append(
            SimulationCondition.from_mapping(payload, detector=geometry)
        )
    return conditions


def iter_structure_conditions(
    structures: Iterable[StructureRecord],
    conditions: Iterable[SimulationCondition],
) -> Iterable[tuple[StructureRecord, SimulationCondition]]:
    """Yield the Cartesian product of structures and conditions."""

    for structure in structures:
        for condition in conditions:
            yield structure, condition


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_from(base_file: Path, raw_path: Any) -> Path:
    if not raw_path:
        raise ConfigError(f"Missing path in {base_file}")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = base_file.parent / path
    return path.resolve()
