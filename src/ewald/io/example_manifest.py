"""Helpers for reading source-checkout example dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ewald.data.models import DataFileRef, ProcessingRecord, ProjectState
from ewald.processing.calibration import CalibrationInputs

DEFAULT_MANIFEST = Path("example") / "manifest.json"


def load_example_manifest(
    path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Load an example manifest and resolve file paths relative to
    it."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    for item in payload.get("data", []):
        item["path"] = str((base_dir / item["path"]).resolve())
    calibration = payload.get("calibration", {})
    for key in ("poni_file", "mask_file"):
        if calibration.get(key):
            calibration[key] = str((base_dir / calibration[key]).resolve())
    return payload


def project_from_example_manifest(
    path: str | Path = DEFAULT_MANIFEST,
) -> ProjectState:
    """Create a ProjectState from an example dataset manifest."""

    manifest = load_example_manifest(path)
    project = ProjectState(name=manifest["name"])
    for item in manifest.get("data", []):
        project.data_files.append(
            DataFileRef(
                path=Path(item["path"]),
                data_id=item.get("data_id"),
                kind=item.get("kind", "detector-image"),
                metadata=dict(item.get("metadata", {})),
            )
        )
    project.processing_history.append(
        ProcessingRecord(
            stage="example.manifest.loaded",
            parameters={"manifest": str(Path(path))},
        )
    )
    return project


def calibration_from_example_manifest(
    path: str | Path = DEFAULT_MANIFEST,
) -> CalibrationInputs:
    """Create CalibrationInputs from an example dataset manifest."""

    manifest = load_example_manifest(path)
    calibration = manifest["calibration"]
    inputs = CalibrationInputs(
        poni_file=Path(calibration["poni_file"]),
        mask_file=Path(calibration["mask_file"]),
        incident_angle_deg=float(
            manifest["data"][0]["metadata"]["incident_angle_deg"]
        ),
        tilt_angle_deg=float(calibration["tilt_angle_deg"]),
        sample_orientation=int(calibration["sample_orientation"]),
        polarization=float(calibration["polarization"]),
        solid_angle=bool(calibration["solid_angle"]),
    )
    inputs.validate()
    return inputs
