#!/usr/bin/env python3
"""Generate EWALD training-data shards from a plan file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from ewald_data_training.conditions import load_generation_plan  # noqa: E402
from ewald_data_training.manifests import write_jsonl_manifest  # noqa: E402
from ewald_data_training.schemas import (  # noqa: E402
    ArtifactProfile,
    DetectorGeometry,
    SimulationCondition,
    StructureRecord,
)
from ewald_data_training.simulator import generate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--structures")
    parser.add_argument("--artifacts")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan).expanduser().resolve()
    loaded = _load_plan(
        plan_path, structures=args.structures, artifacts=args.artifacts
    )
    output_root = Path(args.output_root or loaded["output_root"]).resolve()
    samples = generate_dataset(
        structures=loaded["structures"],
        conditions=loaded["conditions"],
        catalog_root=loaded["catalog_root"],
        output_root=output_root,
        artifact_profiles=loaded["artifact_profiles"],
        dry_run=args.dry_run,
    )
    manifest = Path(args.manifest or output_root / "manifest.jsonl").resolve()
    write_jsonl_manifest(manifest, samples)
    print(f"wrote {len(samples)} samples to {manifest}")
    return 0


def _load_plan(
    path: Path,
    *,
    structures: str | None = None,
    artifacts: str | None = None,
) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        if structures or artifacts:
            loaded = _load_yaml_plan_with_overrides(
                path,
                structures=structures,
                artifacts=artifacts,
            )
        else:
            loaded = load_generation_plan(path)
        loaded["catalog_root"] = Path(loaded["structures_path"]).parent
        return loaded

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "structures" in payload:
        loaded = load_generation_plan(path)
        loaded["catalog_root"] = Path(loaded["structures_path"]).parent
        return loaded
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError(f"{path} must contain a JSON object or array.")

    structures: list[StructureRecord] = []
    conditions: list[SimulationCondition] = []
    profiles: dict[str, ArtifactProfile] = {}
    catalog_root = path.parent
    for index, row in enumerate(rows):
        structure = _row_structure(row)
        condition, profile = _row_condition(row, index)
        structures.append(structure)
        conditions.append(condition)
        profiles[profile.profile_id] = profile
    return {
        "structures": structures,
        "conditions": conditions,
        "artifact_profiles": profiles,
        "catalog_root": catalog_root,
        "output_root": path.parent / "generated",
    }


def _load_yaml_plan_with_overrides(
    path: Path,
    *,
    structures: str | None,
    artifacts: str | None,
) -> dict[str, Any]:
    import tempfile

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if structures:
        payload["structures"] = str(_resolve_cli_path(structures))
    elif payload.get("structures"):
        payload["structures"] = str(
            _resolve_relative(path, payload["structures"])
        )
    if artifacts:
        payload["artifacts"] = str(_resolve_cli_path(artifacts))
    elif payload.get("artifacts"):
        payload["artifacts"] = str(
            _resolve_relative(path, payload["artifacts"])
        )
    if payload.get("output_root"):
        payload["output_root"] = str(
            _resolve_relative(path, payload["output_root"])
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=path.suffix,
        encoding="utf-8",
        delete=False,
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        temp_path = Path(handle.name)
    try:
        return load_generation_plan(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _resolve_relative(base_file: Path, raw_path: object) -> Path:
    value = Path(str(raw_path)).expanduser()
    if not value.is_absolute():
        value = base_file.parent / value
    return value.resolve()


def _resolve_cli_path(raw_path: object) -> Path:
    value = Path(str(raw_path)).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.resolve()


def _row_structure(row: dict[str, Any]) -> StructureRecord:
    source_file = row.get("source_structure_file") or row.get("path")
    if not source_file:
        raise ValueError("JSON plan row requires source_structure_file.")
    return StructureRecord.from_mapping(
        {
            "structure_id": row.get("structure_id"),
            "name": row.get("structure_id") or Path(str(source_file)).stem,
            "path": source_file,
            "file_format": Path(str(source_file)).suffix.lstrip(".") or "cif",
            "family": row.get("family", ""),
            "phase_class": row.get("phase_class", ""),
            "metadata": {
                "plan_id": row.get("plan_id", ""),
                "source": "cluster_jsonl",
            },
        }
    )


def _row_condition(
    row: dict[str, Any],
    index: int,
) -> tuple[SimulationCondition, ArtifactProfile]:
    detector_payload = row.get("detector", {}) or {}
    shape = detector_payload.get("shape")
    resolution = (256, 128)
    if isinstance(shape, list) and len(shape) == 2:
        resolution = (int(shape[1]), int(shape[0]))
    detector = DetectorGeometry.from_mapping(
        {
            "qxy_range": detector_payload.get("qxy_range", [-3.0, 3.0]),
            "qz_range": detector_payload.get("qz_range", [0.0, 3.0]),
            "resolution": detector_payload.get("resolution", resolution),
            "wavelength_angstrom": detector_payload.get("wavelength_a"),
            "incident_angle_deg": detector_payload.get("incident_angle_deg"),
            "detector": detector_payload.get("name", "cluster-plan"),
        }
    )
    texture = row.get("texture", {}) or {}
    artifacts = row.get("artifacts", {}) or {}
    seed = int(row.get("random_seed", row.get("seed", index)))
    profile_id = str(row.get("artifact_profile_id", "cluster_row"))
    condition = SimulationCondition.from_mapping(
        {
            "condition_id": row.get("condition_id"),
            "theta_x_deg": texture.get("theta_x_deg", 90.0),
            "theta_y_deg": texture.get("theta_y_deg", 0.0),
            "sigma_theta": texture.get("mosaic_sigma_deg", 8.0) / 180.0,
            "sigma_phi": texture.get("azimuth_sigma_deg", 18.0) / 180.0,
            "sigma_r": texture.get("sigma_r", 0.035),
            "hkl_extent": row.get("hkl_extent", 4),
            "orientation_label": row.get("orientation_family", "fiber"),
            "texture_model": texture.get("model", "fibril"),
            "artifact_profile_id": profile_id,
            "seed": seed,
            "metadata": {
                "plan_id": row.get("plan_id", ""),
                "physical_detector": detector_payload,
                "texture": texture,
            },
        },
        detector=detector,
    )
    profile = ArtifactProfile.from_mapping(
        {
            "profile_id": profile_id,
            "poisson_counts": artifacts.get("poisson_scale", 2500.0),
            "gaussian_read_noise": artifacts.get("read_noise_sigma", 2.0)
            / 255.0,
            "background_level": artifacts.get("dark_current", 12.0) / 255.0,
            "dead_pixel_fraction": artifacts.get(
                "dead_pixel_fraction", 0.0004
            ),
            "hot_pixel_fraction": artifacts.get("hot_pixel_fraction", 0.0002),
            "beamstop": artifacts.get("beamstop_radius_px", 1) != 0,
        }
    )
    return condition, profile


if __name__ == "__main__":
    raise SystemExit(main())
