"""Dataset generation orchestration around EWALD's GIWAXS backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile

from .artifact_features import (
    annotate_peaks_with_artifacts,
    build_artifact_assessment,
    estimate_retrieval_quality,
)
from .artifacts import apply_artifacts
from .conditions import iter_structure_conditions
from .schemas import (
    ArtifactProfile,
    DatasetSample,
    SimulationCondition,
    StructureRecord,
    stable_id,
)


def generate_dataset(
    *,
    structures: Iterable[StructureRecord],
    conditions: Iterable[SimulationCondition],
    catalog_root: str | Path,
    output_root: str | Path,
    artifact_profiles: dict[str, ArtifactProfile] | None = None,
    dry_run: bool = False,
) -> list[DatasetSample]:
    """Generate a dataset shard and return manifest rows."""

    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    profiles = artifact_profiles or {"default": ArtifactProfile()}
    samples: list[DatasetSample] = []
    for structure, condition in iter_structure_conditions(
        structures, conditions
    ):
        sample = simulate_training_sample(
            structure=structure,
            condition=condition,
            catalog_root=catalog_root,
            output_root=output_path,
            artifact_profiles=profiles,
            dry_run=dry_run,
        )
        samples.append(sample)
    return samples


def simulate_training_sample(
    *,
    structure: StructureRecord,
    condition: SimulationCondition,
    catalog_root: str | Path,
    output_root: str | Path,
    artifact_profiles: dict[str, ArtifactProfile],
    dry_run: bool = False,
) -> DatasetSample:
    """Simulate and label one structure/condition pair."""

    sample_payload = {
        "structure_id": structure.structure_id,
        "condition_id": condition.condition_id,
        "seed": condition.seed,
    }
    sample_id = stable_id(sample_payload, "smp_")
    sample_dir = Path(output_root) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_path = sample_dir / "clean.tiff"
    artifact_path = sample_dir / "artifact.tiff"
    peaks_path = sample_dir / "peaks.json"
    labels_path = sample_dir / "labels.json"
    profile = artifact_profiles.get(
        condition.artifact_profile_id,
        artifact_profiles.get("default", ArtifactProfile()),
    )

    if dry_run:
        image = _placeholder_image(condition)
        peaks: list[dict[str, Any]] = []
        sample_context: dict[str, Any] = {}
        simulation_metadata: dict[str, Any] = {
            "dry_run": True,
            "missing_wedge_correction": bool(
                condition.detector.missing_wedge_correction
            ),
        }
    else:
        structure_path = structure.resolved_path(Path(catalog_root))
        image, peaks, simulation_metadata = _run_ewald_simulation(
            structure_path,
            condition,
        )
        sample_context = _sample_scattering_context(
            structure_path,
            condition,
        )
    artifact_image, artifact_metadata = apply_artifacts(
        image,
        profile,
        seed=condition.seed,
        detector=condition.detector,
        sample_context=sample_context,
    )
    artifact_assessment = build_artifact_assessment(
        artifact_metadata=artifact_metadata,
        artifact_profile=profile,
        detector=condition.detector,
        image_shape=image.shape,
    )
    assessed_peaks = annotate_peaks_with_artifacts(
        peaks,
        artifact_assessment,
    )
    quality_assessment = estimate_retrieval_quality(
        image,
        artifact_image,
        artifact_assessment=artifact_assessment,
        detector=condition.detector,
    )

    tifffile.imwrite(clean_path, np.asarray(image, dtype=np.float32))
    tifffile.imwrite(
        artifact_path, np.asarray(artifact_image, dtype=np.float32)
    )
    peaks_path.write_text(
        json.dumps(
            {
                "peaks": peaks,
                "artifact_assessed_peaks": assessed_peaks,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    labels = {
        "sample_id": sample_id,
        "structure": structure.as_dict(),
        "source_structure_path": str(
            structure.resolved_path(Path(catalog_root))
        ),
        "condition": condition.as_dict(),
        "simulation_metadata": simulation_metadata,
        "sample_scattering": sample_context,
        "artifact_profile": profile.as_dict(),
        "artifact_metadata": artifact_metadata,
        "artifact_assessment": artifact_assessment,
        "quality_assessment": quality_assessment,
        "peak_table_path": peaks_path.name,
    }
    labels_path.write_text(
        json.dumps(labels, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return DatasetSample(
        sample_id=sample_id,
        structure_id=structure.structure_id,
        condition_id=condition.condition_id,
        image_path=str(artifact_path.relative_to(output_root)),
        label_path=str(labels_path.relative_to(output_root)),
        clean_image_path=str(clean_path.relative_to(output_root)),
        peak_table_path=str(peaks_path.relative_to(output_root)),
        artifact_profile_id=profile.profile_id,
        seed=condition.seed,
        metadata={
            "structure_name": structure.name,
            "orientation_label": condition.orientation_label,
            "texture_model": condition.texture_model,
        },
    )


def _run_ewald_simulation(
    structure_path: Path,
    condition: SimulationCondition,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    from ewald.simulation.giwaxs import (  # noqa: PLC0415
        GIWAXSSimulationParameters,
        calculate_giwaxs_peak_rows,
        simulate_giwaxs_image,
    )

    params = GIWAXSSimulationParameters.from_mapping(
        condition.as_giwaxs_parameters()
    )
    data = simulate_giwaxs_image(structure_path, params)
    image = np.asarray(data.values if hasattr(data, "values") else data)
    peaks = calculate_giwaxs_peak_rows(structure_path, params)
    attrs = getattr(data, "attrs", {}) or {}
    metadata = _json_safe_mapping(
        {
            **dict(attrs),
            "missing_wedge_correction": bool(params.missing_wedge_correction),
            "solid_angle_correction": bool(params.solid_angle_correction),
        }
    )
    return image.astype(np.float32, copy=False), peaks, metadata


def _json_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        safe[str(key)] = _json_safe_value(value)
    return safe


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _sample_scattering_context(
    structure_path: Path,
    condition: SimulationCondition,
) -> dict[str, Any]:
    from ewald.simulation.giwaxs import load_structure  # noqa: PLC0415

    try:
        structure = load_structure(structure_path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return {"error": str(exc)}
    volume = abs(
        float(np.linalg.det(np.asarray(structure.lattice, dtype=float)))
    )
    electron_count = sum(
        _atomic_number(symbol) for symbol in structure.species
    )
    if volume <= 1.0e-12 or electron_count <= 0:
        return {
            "atom_count": len(structure.species),
            "electron_count": int(electron_count),
            "unit_cell_volume_angstrom3": float(volume),
        }
    electron_density = electron_count / volume
    wavelength = condition.detector.wavelength_angstrom or 1.0
    classical_e_radius_angstrom = 2.8179403262e-5
    delta = (
        classical_e_radius_angstrom
        * float(wavelength) ** 2
        * electron_density
        / (2.0 * np.pi)
    )
    critical_angle_rad = float(np.sqrt(max(0.0, 2.0 * delta)))
    return {
        "atom_count": len(structure.species),
        "electron_count": int(electron_count),
        "electron_density_e_per_angstrom3": float(electron_density),
        "unit_cell_volume_angstrom3": float(volume),
        "critical_angle_deg": float(np.degrees(critical_angle_rad)),
        "critical_angle_model": "sqrt(2*delta), delta=r_e*lambda^2*rho_e/(2*pi)",
    }


def _atomic_number(symbol: str) -> int:
    periodic = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "K": 19,
        "Ca": 20,
        "Ti": 22,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Br": 35,
        "Ag": 47,
        "Sn": 50,
        "I": 53,
        "Ba": 56,
        "W": 74,
        "Au": 79,
        "Hg": 80,
        "Pb": 82,
        "Bi": 83,
    }
    clean = "".join(char for char in str(symbol) if char.isalpha())
    if not clean:
        return 0
    clean = clean[0].upper() + clean[1:].lower()
    return periodic.get(clean, 0)


def _placeholder_image(condition: SimulationCondition) -> np.ndarray:
    """Create a deterministic dry-run image without importing EWALD."""

    x_pixels, z_pixels = condition.detector.resolution
    yy, xx = np.mgrid[0:z_pixels, 0:x_pixels]
    rng = np.random.default_rng(condition.seed)
    image = np.zeros((z_pixels, x_pixels), dtype=np.float32)
    center_x = (x_pixels - 1) / 2.0
    for index in range(8):
        radius = (index + 1) / 10.0
        x0 = center_x + rng.choice([-1.0, 1.0]) * radius * x_pixels * 0.42
        z0 = rng.uniform(z_pixels * 0.12, z_pixels * 0.88)
        width = rng.uniform(2.0, 7.0)
        amplitude = rng.uniform(0.25, 1.0)
        image += amplitude * np.exp(
            -0.5 * (((xx - x0) ** 2 + (yy - z0) ** 2) / width**2)
        )
    return image
