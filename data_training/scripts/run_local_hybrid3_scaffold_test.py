#!/usr/bin/env python3
"""Run a local HybriD3 2D lead-iodide GIWAXS scaffold smoke study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.hybrid3 import (  # noqa: E402
    HYBRID3_BASE_URL,
    Hybrid3DatasetRecord,
    download_structure_files,
    fetch_atomic_structure_datasets,
)
from ewald_data_training.manifests import read_jsonl_manifest  # noqa: E402
from ewald_data_training.artifacts import apply_artifacts  # noqa: E402
from ewald_data_training.schemas import (
    ArtifactProfile,
    DetectorGeometry,
)  # noqa: E402

DEFAULT_DATASET_IDS = [
    2788,
    2787,
    2786,
    2736,
    2735,
    2734,
    2733,
    2732,
    2731,
    2730,
]
DEFAULT_Q_MAX = 2.8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data_training/runs/hybrid3_2d_fibril_smoke_20260530",
    )
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--use-default-ids", action="store_true")
    parser.add_argument("--variants-per-profile", type=int, default=1)
    parser.add_argument("--guess-max-samples", type=int, default=10)
    parser.add_argument(
        "--q-max",
        type=float,
        default=DEFAULT_Q_MAX,
        help=(
            "Default maximum |qxy| and qz in A^-1. The scaffold defaults "
            "to a lower-q 2.8 A^-1 window to train recognition with less "
            "information than a wide WAXS detector frame."
        ),
    )
    parser.add_argument(
        "--qxy-max",
        type=float,
        help="Override the maximum |qxy| in A^-1.",
    )
    parser.add_argument(
        "--qz-max",
        type=float,
        help="Override the maximum qz in A^-1.",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "docs" / "development"
    assets_dir = (
        ROOT / "docs" / "assets" / "reports" / "hybrid3_local_scaffold"
    )
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = _select_records(
        target_count=args.target_count,
        timeout=args.timeout,
        use_default_ids=args.use_default_ids,
    )
    library_root = output_root / "hybrid3_library"
    summary = download_structure_files(
        records,
        output_root=library_root,
        timeout=args.timeout,
    )
    if summary["ready"] < args.target_count:
        raise RuntimeError(
            f"expected {args.target_count} ready structures, got {summary['ready']}"
        )

    qxy_max = float(args.qxy_max if args.qxy_max is not None else args.q_max)
    qz_max = float(args.qz_max if args.qz_max is not None else args.q_max)
    detector_qxy_range = (-qxy_max, qxy_max)
    detector_qz_range = (0.0, qz_max)
    hkl_extent = _recommended_hkl_extent(
        library_root / "hybrid3_structure_catalog.yaml",
        catalog_root=library_root,
        qxy_range=detector_qxy_range,
        qz_range=detector_qz_range,
    )
    plan_path = _write_generation_plan(
        output_root,
        catalog_path=library_root / "hybrid3_structure_catalog.yaml",
        hkl_extent=hkl_extent,
        qxy_range=detector_qxy_range,
        qz_range=detector_qz_range,
    )
    sim_root = output_root / "simulations"
    sim_manifest = sim_root / "manifest.jsonl"
    artifact_root = output_root / "artifacts"
    artifact_manifest = artifact_root / "artifact_manifest.jsonl"
    model_path = output_root / "model" / "vector_ranker.json"
    metrics_path = output_root / "metrics" / "feedback_metrics.json"
    history_path = output_root / "metrics" / "feedback_history.jsonl"
    guesses_root = output_root / "structure_guesses"

    _run(
        [
            sys.executable,
            str(ROOT / "data_training/scripts/generate_dataset.py"),
            "--plan",
            str(plan_path),
            "--structures",
            str(library_root / "hybrid3_structure_catalog.yaml"),
            "--artifacts",
            str(ROOT / "data_training/configs/artifacts.example.yaml"),
            "--output-root",
            str(sim_root),
            "--manifest",
            str(sim_manifest),
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "data_training/scripts/apply_artifact_variants.py"),
            "--manifest",
            str(sim_manifest),
            "--root",
            str(sim_root),
            "--profiles",
            str(ROOT / "data_training/configs/artifacts.example.yaml"),
            "--output-root",
            str(artifact_root),
            "--output-manifest",
            str(artifact_manifest),
            "--variants-per-profile",
            str(args.variants_per_profile),
            "--seed",
            "3000",
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "data_training/scripts/train_ranker.py"),
            "--manifest",
            str(sim_manifest),
            "--root",
            str(sim_root),
            "--output",
            str(model_path),
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "data_training/scripts/feedback_evaluate.py"),
            "--model",
            str(model_path),
            "--manifest",
            str(artifact_manifest),
            "--root",
            str(artifact_root),
            "--output",
            str(metrics_path),
            "--history",
            str(history_path),
            "--top-k",
            "5",
        ],
        cwd=ROOT,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "data_training/scripts/export_structure_guesses.py"),
            "--model",
            str(model_path),
            "--manifest",
            str(artifact_manifest),
            "--root",
            str(artifact_root),
            "--output-root",
            str(guesses_root),
            "--top-k",
            "5",
            "--max-samples",
            str(args.guess_max_samples),
        ],
        cwd=ROOT,
    )

    clean_preview = assets_dir / "hybrid3_clean_examples.png"
    artifact_preview = assets_dir / "hybrid3_artifact_examples.png"
    surface_preview = assets_dir / "hybrid3_surface_scattering_example.png"
    missing_wedge_preview = assets_dir / "hybrid3_missing_wedge_example.png"
    _write_preview_grid(
        sim_root,
        sim_manifest,
        clean_preview,
        image_attr="clean_image_path",
        title="Clean GIWAXS Simulation Examples",
    )
    _write_missing_wedge_preview(
        sim_root,
        sim_manifest,
        missing_wedge_preview,
    )
    _write_preview_grid(
        artifact_root,
        artifact_manifest,
        artifact_preview,
        image_attr="image_path",
        title="Artifact-Augmented Examples",
    )
    _write_surface_scattering_preview(
        sim_root,
        artifact_root,
        artifact_manifest,
        surface_preview,
    )
    report_path = report_dir / "hybrid3-local-training-scaffold-test.md"
    pdf_path = report_dir / "hybrid3-local-training-scaffold-test.pdf"
    _write_report(
        report_path,
        output_root=output_root,
        library_root=library_root,
        sim_root=sim_root,
        artifact_root=artifact_root,
        guesses_root=guesses_root,
        clean_preview=clean_preview,
        artifact_preview=artifact_preview,
        surface_preview=surface_preview,
        missing_wedge_preview=missing_wedge_preview,
        metrics_path=metrics_path,
        hkl_extent=hkl_extent,
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/render_workflow_report_pdf.py"),
            "--input",
            str(report_path),
            "--output",
            str(pdf_path),
        ],
        cwd=ROOT,
    )

    print(f"catalog={library_root / 'hybrid3_structure_catalog.yaml'}")
    print(f"sim_manifest={sim_manifest}")
    print(f"artifact_manifest={artifact_manifest}")
    print(f"metrics={metrics_path}")
    print(f"report={report_path}")
    print(f"pdf={pdf_path}")
    return 0


def _select_records(
    *,
    target_count: int,
    timeout: float,
    use_default_ids: bool,
) -> list[Hybrid3DatasetRecord]:
    if use_default_ids:
        return [
            _fetch_record(dataset_id, timeout)
            for dataset_id in DEFAULT_DATASET_IDS[:target_count]
        ]

    candidates = []
    for record in fetch_atomic_structure_datasets(
        page_size=50,
        limit=160,
        timeout=timeout,
    ):
        system = record.metadata.get("system") or {}
        inorganic = str(system.get("inorganic") or "")
        dimensionality = str(
            system.get("dimensionality") or record.dimensionality
        )
        if "PbI4" not in inorganic:
            continue
        if dimensionality != "2":
            continue
        candidates.append(record)
        if len(candidates) >= target_count:
            break
    if len(candidates) >= target_count:
        return candidates[:target_count]
    return [
        _fetch_record(dataset_id, timeout)
        for dataset_id in DEFAULT_DATASET_IDS[:target_count]
    ]


def _fetch_record(dataset_id: int, timeout: float) -> Hybrid3DatasetRecord:
    response = requests.get(
        f"{HYBRID3_BASE_URL}/materials/datasets/{dataset_id}/",
        timeout=timeout,
    )
    response.raise_for_status()
    return Hybrid3DatasetRecord.from_api_payload(response.json())


def _write_generation_plan(
    output_root: Path,
    *,
    catalog_path: Path,
    hkl_extent: int,
    qxy_range: tuple[float, float],
    qz_range: tuple[float, float],
) -> Path:
    plan_path = output_root / "hybrid3_2d_fibril_smoke.yaml"
    payload: dict[str, Any] = {
        "dataset": "hybrid3_2d_fibril_smoke_20260530",
        "structures": str(catalog_path),
        "artifacts": str(
            ROOT / "data_training/configs/artifacts.example.yaml"
        ),
        "output_root": str(output_root / "simulations"),
        "default_artifact_profile_id": "clean",
        "detector": {
            "qxy_range": list(qxy_range),
            "qz_range": list(qz_range),
            "resolution": [160, 128],
            "wavelength_angstrom": 1.0,
            "incident_angle_deg": 0.2,
            "tilt_angle_deg": 0.0,
            "solid_angle_correction": True,
            "missing_wedge_correction": True,
            "detector": "pilatus1m_pyfai",
        },
        "sweep": {
            "theta_x_deg": [90],
            "theta_y_deg": [0, 90],
            "sigma_theta": [0.035],
            "sigma_phi": [0.28],
            "sigma_r": [0.04],
            "q_dependent_sigma_r": [0.25],
            "q_dependent_sigma_z": [0.15],
            "hkl_extent": [int(hkl_extent)],
            "orientation_label": ["fiber_out_of_plane"],
            "texture_model": ["fiber_gaussian"],
            "artifact_profile_id": ["clean"],
            "seed": [4301],
        },
        "metadata": {
            "purpose": "Local HybriD3 2D Pb-I fibril-texture scaffold test.",
            "cohort": "visible HybriD3 atomic structures with PbI4 inorganic sublattices",
            "q_range_policy": (
                "default smoke tests use a 2-3 A^-1 q-window so structure "
                "ranking is evaluated with limited GIWAXS information"
            ),
            "hkl_extent_policy": (
                "max recommended extent across pulled structures for the requested q-range"
            ),
        },
    }
    plan_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return plan_path


def _recommended_hkl_extent(
    catalog_path: Path,
    *,
    catalog_root: Path,
    qxy_range: tuple[float, float],
    qz_range: tuple[float, float],
) -> int:
    from ewald.simulation.giwaxs import (  # noqa: PLC0415
        recommend_hkl_extent_for_q_range,
    )

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    extents: list[int] = []
    for item in catalog.get("structures", []):
        raw_path = Path(str(item.get("path", ""))).expanduser()
        structure_path = (
            raw_path
            if raw_path.is_absolute()
            else (catalog_root / raw_path).resolve()
        )
        if not structure_path.exists():
            continue
        extents.append(
            recommend_hkl_extent_for_q_range(
                structure_path,
                qxy_range=qxy_range,
                qz_range=qz_range,
                theta_x_deg=90.0,
                theta_y_deg=0.0,
                min_extent=12,
                max_extent=32,
            )
        )
    return max(extents, default=18)


def _write_preview_grid(
    root: Path,
    manifest_path: Path,
    output: Path,
    *,
    image_attr: str,
    title: str,
    max_images: int = 6,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    samples = _select_preview_samples(
        read_jsonl_manifest(manifest_path),
        max_images=max_images,
        prefer_artifact="artifact" in title.lower(),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for axis, sample in zip(axes.ravel(), samples):
        rel_path = getattr(sample, image_attr)
        image = tifffile.imread(root / rel_path)
        axis.imshow(image, cmap="magma", origin="lower")
        axis.set_title(sample.structure_id, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(samples) :]:
        axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _select_preview_samples(
    samples: list[Any],
    *,
    max_images: int,
    prefer_artifact: bool,
) -> list[Any]:
    selected = []
    seen_structures = set()
    for sample in samples:
        if prefer_artifact and sample.artifact_profile_id == "clean":
            continue
        if sample.structure_id in seen_structures:
            continue
        selected.append(sample)
        seen_structures.add(sample.structure_id)
        if len(selected) >= max_images:
            return selected
    for sample in samples:
        if sample in selected:
            continue
        selected.append(sample)
        if len(selected) >= max_images:
            break
    return selected


def _write_missing_wedge_preview(
    sim_root: Path,
    sim_manifest: Path,
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    samples = read_jsonl_manifest(sim_manifest)
    if not samples:
        raise RuntimeError(
            "no clean simulation samples for missing-wedge preview"
        )
    sample = samples[0]
    image = tifffile.imread(sim_root / sample.clean_image_path).astype(float)
    labels = json.loads(
        (sim_root / sample.label_path).read_text(encoding="utf-8")
    )
    condition = labels.get("condition", {})
    detector = condition.get("detector", {}) or {}
    metadata = _missing_wedge_metadata(labels)
    qxy_range = tuple(
        detector.get("qxy_range", (-DEFAULT_Q_MAX, DEFAULT_Q_MAX))
    )
    qz_range = tuple(detector.get("qz_range", (0.0, DEFAULT_Q_MAX)))
    qxy = np.linspace(float(qxy_range[0]), float(qxy_range[1]), image.shape[1])
    qz = np.linspace(float(qz_range[0]), float(qz_range[1]), image.shape[0])
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    mask = _approximate_missing_wedge_mask(
        qxy_grid,
        qz_grid,
        detector=detector,
        metadata=metadata,
    )

    display = np.log1p(np.clip(image, 0.0, None) * 20.0)
    vmax = np.percentile(display, 99.2) if np.any(display) else 1.0
    masked_profile = mask.mean(axis=1)
    intensity_profile = _normalize_trace(
        np.nanmean(np.clip(image, 0.0, None), axis=1)
    )
    horizon = float(metadata.get("missing_wedge_horizon_qz", 0.0))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.8),
        gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]},
        constrained_layout=True,
    )
    fig.suptitle(
        "Fiber/GI Missing-Wedge Correction Diagnostic",
        fontsize=14,
        fontweight="bold",
    )
    image_axis, mask_axis, profile_axis = axes
    image_axis.imshow(
        display,
        cmap="magma",
        origin="lower",
        extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    image_axis.axhline(horizon, color="#67e8f9", lw=1.1, alpha=0.9)
    image_axis.set_title(f"{sample.structure_id} clean image", fontsize=9)
    image_axis.set_xlabel("qxy (A^-1)")
    image_axis.set_ylabel("qz (A^-1)")

    mask_axis.imshow(
        mask.astype(float),
        cmap="Blues",
        origin="lower",
        extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    mask_axis.axhline(horizon, color="#0f172a", lw=1.1, alpha=0.8)
    mask_axis.set_title("Masked inaccessible q bins", fontsize=9)
    mask_axis.set_xlabel("qxy (A^-1)")
    mask_axis.set_yticks([])

    profile_axis.plot(intensity_profile, qz, color="#0f172a", label="mean I")
    profile_axis.plot(
        masked_profile,
        qz,
        color="#2563eb",
        label="masked fraction",
    )
    profile_axis.axhline(horizon, color="#06b6d4", lw=1.2, alpha=0.9)
    profile_axis.set_ylim(
        float(qz[0]), min(float(qz[-1]), max(0.35, horizon + 0.6))
    )
    profile_axis.set_xlim(0.0, 1.05)
    profile_axis.set_xlabel("Normalized value")
    profile_axis.set_title(
        f"alpha_i={metadata.get('missing_wedge_incident_angle_deg', 0.0):.3f} deg; "
        f"masked={metadata.get('missing_wedge_masked_fraction', 0.0):.3f}",
        fontsize=9,
    )
    profile_axis.grid(alpha=0.18)
    profile_axis.legend(loc="lower right", fontsize=8, frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _missing_wedge_metadata(
    labels: dict[str, Any],
) -> dict[str, float | bool | str]:
    import numpy as np  # noqa: PLC0415

    metadata = labels.get("simulation_metadata")
    if isinstance(metadata, dict) and metadata:
        return dict(metadata)
    detector = labels.get("condition", {}).get("detector", {}) or {}
    wavelength = float(detector.get("wavelength_angstrom") or 1.0)
    incident_angle = float(detector.get("incident_angle_deg") or 0.0)
    k0 = 2.0 * np.pi / max(wavelength, 1.0e-12)
    horizon = k0 * np.sin(np.radians(incident_angle))
    fallback = {
        "missing_wedge_correction_applied": bool(
            detector.get("missing_wedge_correction", True)
        ),
        "missing_wedge_model": "pyfai_fiber_qip_qoop_accessibility",
        "missing_wedge_horizon_qz": float(horizon),
        "missing_wedge_incident_angle_deg": float(incident_angle),
        "missing_wedge_tilt_angle_deg": float(
            detector.get("tilt_angle_deg", 0.0)
        ),
        "missing_wedge_wavelength_angstrom": float(wavelength),
        "missing_wedge_masked_fraction": 0.0,
    }
    try:
        qxy_range = tuple(
            detector.get("qxy_range", (-DEFAULT_Q_MAX, DEFAULT_Q_MAX))
        )
        qz_range = tuple(detector.get("qz_range", (0.0, DEFAULT_Q_MAX)))
        resolution = tuple(detector.get("resolution", (160, 128)))
        qxy = np.linspace(
            float(qxy_range[0]), float(qxy_range[1]), int(resolution[0])
        )
        qz = np.linspace(
            float(qz_range[0]), float(qz_range[1]), int(resolution[1])
        )
        qxy_grid, qz_grid = np.meshgrid(qxy, qz)
        fallback["missing_wedge_masked_fraction"] = float(
            np.mean(
                _approximate_missing_wedge_mask(
                    qxy_grid,
                    qz_grid,
                    detector=detector,
                    metadata=fallback,
                )
            )
        )
    except Exception:
        fallback["missing_wedge_masked_fraction"] = 0.0
    return fallback


def _approximate_missing_wedge_mask(
    qxy_grid: Any,
    qz_grid: Any,
    *,
    detector: dict[str, Any],
    metadata: dict[str, Any],
) -> Any:
    import numpy as np  # noqa: PLC0415

    if not metadata.get("missing_wedge_correction_applied", False):
        return np.zeros_like(qz_grid, dtype=bool)
    wavelength = float(
        metadata.get(
            "missing_wedge_wavelength_angstrom",
            detector.get("wavelength_angstrom") or 1.0,
        )
    )
    incident_angle = float(
        metadata.get(
            "missing_wedge_incident_angle_deg",
            detector.get("incident_angle_deg") or 0.0,
        )
    )
    tilt_angle = float(
        metadata.get(
            "missing_wedge_tilt_angle_deg",
            detector.get("tilt_angle_deg", 0.0),
        )
    )
    k0 = 2.0 * np.pi / max(wavelength, 1.0e-12)
    tilt = np.radians(tilt_angle)
    if abs(tilt) > 0.0:
        qip_eff = qxy_grid * np.cos(tilt) + qz_grid * np.sin(tilt)
        qoop_eff = qz_grid * np.cos(tilt) - qxy_grid * np.sin(tilt)
    else:
        qip_eff = qxy_grid
        qoop_eff = qz_grid
    qip_abs = np.abs(qip_eff)
    alpha = np.radians(incident_angle)
    sin_alpha = np.sin(alpha)
    cos_alpha = max(abs(float(np.cos(alpha))), 1.0e-12)
    qbeam = (
        2.0 * k0 * sin_alpha * qoop_eff - qip_abs**2 - qoop_eff**2
    ) / (2.0 * k0 * cos_alpha)
    q_scale = max(
        abs(float(np.nanmin(qxy_grid))),
        abs(float(np.nanmax(qxy_grid))),
        abs(float(np.nanmin(qz_grid))),
        abs(float(np.nanmax(qz_grid))),
        1.0,
    )
    tolerance = 1.0e-9 + 1.0e-6 * q_scale
    valid = (np.abs(qbeam) <= qip_abs + tolerance) & (
        qoop_eff >= k0 * sin_alpha - tolerance
    )
    return ~valid


def _normalize_trace(values: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    trace = np.asarray(values, dtype=float)
    trace = trace - float(np.nanmin(trace))
    scale = float(np.nanmax(trace))
    if scale <= 0.0:
        return np.zeros_like(trace)
    return trace / scale


def _write_surface_scattering_preview(
    sim_root: Path,
    artifact_root: Path,
    artifact_manifest: Path,
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    sample, labels = _select_surface_sample(artifact_root, artifact_manifest)
    source_sample = labels.get("source_sample", {})
    clean_path = sim_root / str(source_sample.get("clean_image_path", ""))
    clean_image = tifffile.imread(clean_path).astype(float)
    source_label_path = (
        source_sample.get("label_path") or sample.clean_image_path
    )
    if str(source_label_path).endswith((".tif", ".tiff")):
        source_label_path = Path(str(source_label_path)).with_name(
            "labels.json"
        )
    source_label = json.loads(
        (sim_root / source_label_path).read_text(encoding="utf-8")
    )
    detector = source_label.get("condition", {}).get(
        "detector", {}
    ) or labels.get("source_sample", {}).get("condition", {}).get(
        "detector", {}
    )
    detector_geometry = DetectorGeometry.from_mapping(detector)
    source_profile = ArtifactProfile.from_mapping(
        labels.get("artifact_profile", {}) or {}
    )
    diagnostic_detector = replace(detector_geometry, detector="")
    surface_profile = replace(
        source_profile,
        profile_id="surface_diagnostic",
        poisson_counts=0.0,
        gaussian_read_noise=0.0,
        background_level=0.0,
        background_gradient=(0.0, 0.0),
        q_dependent_background=0.0,
        beamstop=False,
        detector_layout="",
        detector_gap_fraction=0.0,
        detector_gap_jitter_pixels=0,
        dead_pixel_fraction=0.0,
        dead_pixel_cluster_count=0,
        hot_pixel_fraction=0.0,
        flat_field_strength=0.0,
        parasitic_streaks=0,
        diffuse_ring_count=0,
        saturation_level=1.0,
    )
    image, surface_metadata = apply_artifacts(
        clean_image,
        surface_profile,
        seed=int(sample.seed),
        detector=diagnostic_detector,
        sample_context=labels.get("sample_scattering", {}),
    )
    image = image.astype(float)
    qxy_range = tuple(detector.get("qxy_range", (-4.0, 4.0)))
    qz_range = tuple(detector.get("qz_range", (0.0, 4.0)))
    qxy = np.linspace(float(qxy_range[0]), float(qxy_range[1]), image.shape[1])
    qz = np.linspace(float(qz_range[0]), float(qz_range[1]), image.shape[0])
    surface = surface_metadata.get("surface_scattering", {})
    operations = surface_metadata.get("operations", [])

    qxy_target = 0.9
    qxy_band = 0.18
    trace_mask = np.abs(qxy - qxy_target) <= qxy_band
    if not np.any(trace_mask):
        trace_mask[np.argmin(np.abs(qxy - qxy_target))] = True
    trace = image[:, trace_mask].mean(axis=1)
    display = np.log1p(np.clip(image, 0.0, None) * 18.0)
    vmax = np.percentile(display, 99.2)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.6, 8.0),
        gridspec_kw={"width_ratios": [1.3, 1.0]},
        constrained_layout=True,
    )
    fig.suptitle(
        "Surface-Scattering Artifact Diagnostic",
        fontsize=15,
        fontweight="bold",
    )

    full_axis = axes[0, 0]
    full_axis.imshow(
        display,
        cmap="magma",
        origin="lower",
        extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    full_axis.axvspan(
        qxy_target - qxy_band,
        qxy_target + qxy_band,
        color="#67e8f9",
        alpha=0.22,
        lw=0,
    )
    for label, color in _surface_line_styles().items():
        value = surface.get(label)
        if value is not None:
            full_axis.axhline(value, color=color, lw=1.1, alpha=0.82)
    full_axis.set_title(
        f"{sample.structure_id} / surface-only diagnostic from {source_profile.profile_id}",
        fontsize=10,
    )
    full_axis.set_xlabel("qxy (A^-1)")
    full_axis.set_ylabel("qz (A^-1)")

    lowq_axis = axes[1, 0]
    lowq_axis.imshow(
        display,
        cmap="magma",
        origin="lower",
        extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    for label, color in _surface_line_styles().items():
        value = surface.get(label)
        if value is not None:
            lowq_axis.axhline(
                value,
                color=color,
                lw=1.4,
                alpha=0.9,
                label=_surface_label(label),
            )
    lowq_axis.set_xlim(-2.0, 2.0)
    lowq_top = max(
        0.16,
        float(surface.get("specular_qz", 0.0)) + 0.12,
        float(surface.get("yoneda_qz", 0.0)) + 0.12,
    )
    lowq_axis.set_ylim(0.0, min(float(qz[-1]), lowq_top))
    lowq_axis.set_title(
        "Low-q horizon / Yoneda / specular region",
        fontsize=10,
    )
    lowq_axis.set_xlabel("qxy (A^-1)")
    lowq_axis.set_ylabel("qz (A^-1)")
    lowq_axis.legend(loc="upper right", fontsize=8, frameon=True)

    trace_axis = axes[0, 1]
    clean_preview = np.asarray(clean_image, dtype=float)
    clean_preview -= float(np.nanmin(clean_preview))
    clean_scale = float(np.nanmax(clean_preview))
    if clean_scale > 0.0:
        clean_preview /= clean_scale
    clean_trace = clean_preview[:, trace_mask].mean(axis=1)
    trace_axis.plot(trace, qz, color="#0f172a", lw=1.8, label="surface-only")
    trace_axis.plot(
        clean_trace,
        qz,
        color="#64748b",
        lw=1.1,
        alpha=0.8,
        label="clean",
    )
    for label, color in _surface_line_styles().items():
        value = surface.get(label)
        if value is not None:
            trace_axis.axhline(value, color=color, lw=1.2, alpha=0.9)
    trace_axis.set_ylim(0.0, min(float(qz[-1]), max(0.22, lowq_top + 0.08)))
    trace_axis.set_title(
        f"qz trace averaged near qxy = {qxy_target:.1f} A^-1",
        fontsize=10,
    )
    trace_axis.set_xlabel("Mean intensity")
    trace_axis.set_ylabel("qz (A^-1)")
    trace_axis.grid(alpha=0.18)
    trace_axis.legend(loc="lower right", fontsize=8, frameon=False)

    info_axis = axes[1, 1]
    info_axis.axis("off")
    info_axis.text(
        0.0,
        1.0,
        _surface_metadata_text(surface, operations),
        ha="left",
        va="top",
        fontsize=8.3,
        family="monospace",
        linespacing=1.25,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _select_surface_sample(
    artifact_root: Path,
    artifact_manifest: Path,
) -> tuple[Any, dict[str, Any]]:
    required = {
        "direct_beam_specular",
        "yoneda_band",
        "critical_angle_peak_splitting",
        "substrate_horizon_shadow",
    }
    preferred_structure = "hybrid3_2788"
    best: tuple[int, Any, dict[str, Any]] | None = None
    for sample in read_jsonl_manifest(artifact_manifest):
        label_path = artifact_root / sample.label_path
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        metadata = labels.get("artifact_metadata", {})
        operations = set(metadata.get("operations", []))
        if not required.issubset(operations):
            continue
        surface = metadata.get("surface_scattering", {})
        spillage = float(surface.get("footprint_spillage_fraction", 0.0))
        score = int(sample.structure_id == preferred_structure) * 10
        score += int("footprint_spillage_peak_broadening" in operations) * 4
        score += int(metadata.get("profile_id") == "default") * 3
        score += int(metadata.get("profile_id") == "pilatus1m_reference") * 2
        score += int(0.01 <= spillage <= 0.35) * 2
        score -= int(metadata.get("profile_id") == "harsh_detector") * 4
        if best is None or score > best[0]:
            best = (score, sample, labels)
    if best is None:
        raise RuntimeError("no surface-scattering artifact sample found")
    return best[1], best[2]


def _surface_line_styles() -> dict[str, str]:
    return {
        "direct_beam_qz": "#38bdf8",
        "horizon_qz": "#facc15",
        "yoneda_qz": "#fb7185",
        "specular_qz": "#34d399",
    }


def _surface_label(metadata_key: str) -> str:
    return {
        "direct_beam_qz": "direct",
        "horizon_qz": "horizon",
        "yoneda_qz": "Yoneda",
        "specular_qz": "specular",
    }.get(metadata_key, metadata_key)


def _surface_metadata_text(
    surface: dict[str, Any],
    operations: list[str],
) -> str:
    active = [
        op
        for op in (
            "footprint_spillage_peak_broadening",
            "direct_beam_specular",
            "yoneda_band",
            "critical_angle_peak_splitting",
            "substrate_horizon_shadow",
            "detector_module_gap_mask",
            "beamstop_shadow",
        )
        if op in operations
    ]
    return "\n".join(
        [
            "Geometry",
            f"  alpha_i: {surface.get('incident_angle_deg', 0.0):.3f} deg",
            f"  lambda:  {surface.get('wavelength_angstrom', 0.0):.3f} A",
            f"  alpha_c: {surface.get('critical_angle_deg', 0.0):.3f} deg",
            "",
            "Surface features (qz, A^-1)",
            f"  direct:   {surface.get('direct_beam_qz', 0.0):.4f}",
            f"  horizon:  {surface.get('horizon_qz', 0.0):.4f}",
            f"  Yoneda:   {surface.get('yoneda_qz', 0.0):.4f}",
            f"  specular: {surface.get('specular_qz', 0.0):.4f}",
            f"  split dq: {surface.get('critical_q_shift', 0.0):.4f}",
            "",
            "Substrate / beam footprint",
            f"  substrate: {surface.get('substrate_length_mm', 0.0):.1f} x "
            f"{surface.get('substrate_width_mm', 0.0):.1f} mm",
            f"  beam:      {surface.get('beam_height_um', 0.0):.1f} um x "
            f"{surface.get('beam_width_mm', 0.0):.2f} mm",
            f"  footprint: {surface.get('beam_footprint_length_mm', 0.0):.2f} mm",
            f"  spillover: {surface.get('footprint_spillage_fraction', 0.0):.3f}",
            f"  below-horizon T: "
            f"{surface.get('below_horizon_transmission', 0.0):.3f}",
            "",
            "Active operations",
            *[f"  - {op}" for op in active],
        ]
    )


def _artifact_quality_summary(
    artifact_root: Path,
    artifact_samples: list[Any],
) -> dict[str, float | int]:
    quality_rows: list[dict[str, Any]] = []
    for sample in artifact_samples:
        label_path = artifact_root / sample.label_path
        if not label_path.exists():
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        quality = labels.get("quality_assessment")
        if isinstance(quality, dict):
            quality_rows.append(quality)
    if not quality_rows:
        return {
            "sample_count": 0,
            "solvable_count": 0,
            "solvable_fraction": 0.0,
            "median_signal_to_noise": 0.0,
            "median_retrievable_signal_fraction": 0.0,
        }
    import statistics

    snr = [float(row.get("signal_to_noise", 0.0)) for row in quality_rows]
    retrievable = [
        float(row.get("retrievable_signal_fraction", 0.0))
        for row in quality_rows
    ]
    solvable_count = sum(1 for row in quality_rows if row.get("solvable"))
    return {
        "sample_count": len(quality_rows),
        "solvable_count": solvable_count,
        "solvable_fraction": solvable_count / len(quality_rows),
        "median_signal_to_noise": float(statistics.median(snr)),
        "median_retrievable_signal_fraction": float(
            statistics.median(retrievable)
        ),
    }


def _simulation_correction_summary(
    sim_root: Path,
    sim_samples: list[Any],
) -> dict[str, float | int | str]:
    rows: list[dict[str, Any]] = []
    for sample in sim_samples:
        label_path = sim_root / sample.label_path
        if not label_path.exists():
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        rows.append(_missing_wedge_metadata(labels))
    applied = [
        row
        for row in rows
        if bool(row.get("missing_wedge_correction_applied"))
    ]
    masked = [
        float(row.get("missing_wedge_masked_fraction", 0.0)) for row in applied
    ]
    horizons = [
        float(row.get("missing_wedge_horizon_qz", 0.0)) for row in applied
    ]
    import statistics

    return {
        "sample_count": len(rows),
        "applied_count": len(applied),
        "model": (
            str(applied[0].get("missing_wedge_model", "")) if applied else ""
        ),
        "median_masked_fraction": (
            float(statistics.median(masked)) if masked else 0.0
        ),
        "median_horizon_qz": (
            float(statistics.median(horizons)) if horizons else 0.0
        ),
    }


def _artifact_operation_summary(
    artifact_root: Path,
    artifact_samples: list[Any],
) -> dict[str, Any]:
    operation_counts: dict[str, int] = {}
    profiles: dict[str, set[str]] = {}
    surface_samples = 0
    for sample in artifact_samples:
        label_path = artifact_root / sample.label_path
        if not label_path.exists():
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        metadata = labels.get("artifact_metadata", {}) or {}
        operations = [str(item) for item in metadata.get("operations", [])]
        if "surface_scattering" in metadata:
            surface_samples += 1
        profile_id = str(
            metadata.get("profile_id")
            or labels.get("artifact_profile", {}).get("profile_id", "")
            or sample.artifact_profile_id
        )
        if profile_id:
            profiles.setdefault(profile_id, set()).update(operations)
        for operation in operations:
            operation_counts[operation] = (
                operation_counts.get(operation, 0) + 1
            )
    core_surface_ops = (
        "direct_beam_specular",
        "yoneda_band",
        "critical_angle_peak_splitting",
        "substrate_horizon_shadow",
        "footprint_spillage_peak_broadening",
    )
    active_core = [
        operation
        for operation in core_surface_ops
        if operation in operation_counts
    ]
    return {
        "surface_sample_count": surface_samples,
        "operation_counts": operation_counts,
        "profiles": {key: sorted(value) for key, value in profiles.items()},
        "active_core_surface_operations": active_core,
    }


def _write_report(
    report_path: Path,
    *,
    output_root: Path,
    library_root: Path,
    sim_root: Path,
    artifact_root: Path,
    guesses_root: Path,
    clean_preview: Path,
    artifact_preview: Path,
    surface_preview: Path,
    missing_wedge_preview: Path,
    metrics_path: Path,
    hkl_extent: int,
) -> None:
    catalog = yaml.safe_load(
        (library_root / "hybrid3_structure_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    structures = catalog.get("structures", [])
    sim_samples = read_jsonl_manifest(sim_root / "manifest.jsonl")
    artifact_samples = read_jsonl_manifest(
        artifact_root / "artifact_manifest.jsonl"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    first_labels = {}
    if sim_samples:
        first_label_path = sim_root / sim_samples[0].label_path
        if first_label_path.exists():
            first_labels = json.loads(
                first_label_path.read_text(encoding="utf-8")
            )
    detector = first_labels.get("condition", {}).get("detector", {})
    qxy_range = tuple(
        detector.get("qxy_range", (-DEFAULT_Q_MAX, DEFAULT_Q_MAX))
    )
    qz_range = tuple(detector.get("qz_range", (0.0, DEFAULT_Q_MAX)))
    qxy_text = f"[{qxy_range[0]:.1f}, {qxy_range[1]:.1f}]"
    qz_text = f"[{qz_range[0]:.1f}, {qz_range[1]:.1f}]"
    q_max = max(abs(qxy_range[0]), abs(qxy_range[1]), abs(qz_range[1]))
    quality = _artifact_quality_summary(artifact_root, artifact_samples)
    correction = _simulation_correction_summary(sim_root, sim_samples)
    artifact_ops = _artifact_operation_summary(artifact_root, artifact_samples)
    core_surface_text = ", ".join(
        f"`{item}`" for item in artifact_ops["active_core_surface_operations"]
    )
    if not core_surface_text:
        core_surface_text = "`none`"
    structure_rows = "\n".join(
        _structure_table_row(item) for item in structures
    )
    report_path.write_text(
        f"""# HybriD3 Local Training Scaffold Test

Date: 2026-05-30

PDF version: [hybrid3-local-training-scaffold-test.pdf](hybrid3-local-training-scaffold-test.pdf)

## Summary

This local smoke study pulled ten similar HybriD3 atomic-structure records,
cataloged their database and crystallographic metadata, generated a compact
GIWAXS simulation set for 2D fibril-textured samples, applied detector-image
artifacts, trained the baseline vector ranker, evaluated top-k recovery, and
exported ranked structure-file guesses.

| Quantity | Value |
| --- | ---: |
| Structures pulled | {len(structures)} |
| Clean simulation samples | {len(sim_samples)} |
| Artifact-augmented samples | {len(artifact_samples)} |
| Missing-wedge corrected clean samples | {correction["applied_count"]} / {correction["sample_count"]} |
| Median missing-wedge masked fraction | {correction["median_masked_fraction"]:.3f} |
| Surface-artifact augmented samples | {artifact_ops["surface_sample_count"]} |
| Ranker candidates | {metrics.get("candidate_count", 0)} |
| Evaluated artifact images | {metrics.get("evaluated", 0)} |
| Solvable augmented fraction | {quality["solvable_fraction"]:.3f} |
| Median artifact SNR | {quality["median_signal_to_noise"]:.2f} |
| Top-1 accuracy | {metrics.get("top1_accuracy", 0.0):.3f} |
| Top-5 accuracy | {metrics.get("top5_accuracy", 0.0):.3f} |

## Pulled Structure Cohort

Selection targeted visible HybriD3 atomic-structure datasets with 2D `PbI4`
lead-iodide inorganic sublattices. This gives a chemically coherent local test
set for fibril-textured GIWAXS recognition while keeping the run small.

| Dataset | Structure id | Inorganic | Organic | File | Sites |
| ---: | --- | --- | --- | --- | ---: |
{structure_rows}

## Simulation Protocol

Clean GIWAXS images were generated on a deliberately lower-information
reciprocal-space grid with `qxy = {qxy_text} A^-1`, `qz = {qz_text} A^-1`,
and `160 x 128` local smoke-test pixels. The default cap is now biased toward
2-3 A^-1 rather than 4 A^-1 so the solver is tested against detector windows
that resemble practical GIWAXS measurements with less high-q information. The
texture model was `fiber_gaussian` with `theta_x = 90 deg`,
`theta_y = 0/90 deg`, `sigma_theta = 0.035`, `sigma_phi = 0.28`,
`sigma_r = 0.04`, q-dependent broadening terms of `0.25` in-plane and `0.15`
out-of-plane, and `hkl_extent = {hkl_extent}`. The hkl extent is selected as the
maximum recommended reciprocal-lattice search across the pulled structures for
the requested q-range, so long-axis 2D structures such as `hybrid3_2788`,
`hybrid3_2787`, and `hybrid3_2786` still populate the detector plane out to
`q <= {q_max:.1f} A^-1`.

The local forward model now includes a first-order flat-detector solid-angle
response derived from the Ewald-sphere scattering angle
`q = 4 pi sin(theta) / lambda`. This keeps the edge of the detector realistic
while preserving the existing labeled `(hkl)` peak table contract. The same
incident-angle geometry is also applied as a pyFAI-style fiber/GI missing-wedge
correction in qIP/qOOP space, so inaccessible bins below the sample horizon are
zeroed and excluded from the labeled `(hkl)` peak table. This run applied the
`{correction["model"]}` missing-wedge model to {correction["applied_count"]} of
{correction["sample_count"]} clean samples, with a median horizon of
`qz = {correction["median_horizon_qz"]:.4f} A^-1` and a median masked fraction
of {correction["median_masked_fraction"]:.3f}. Every generated clean label
stores a `simulation_metadata` block with the missing-wedge model, incident
angle, wavelength, horizon, and masked fraction used for the example.

Detector artifacts were generated from the scaffold artifact profiles:
`clean`, `default`, `harsh_detector`, and `pilatus1m_reference`. Diffuse
scattering is now generated as broad rings centered on q-values inferred from
the simulated Bragg signal, instead of arbitrary vertical detector streaks.
The profiles also include Poisson noise, Gaussian read noise, q-dependent
background, direct/specular beam artifacts, Yoneda bands, substrate horizon
shadowing, critical-angle peak splitting, beamstop shadow, flat-field variation,
hot/dead pixels, dead-pixel clusters, saturation, and detector-module masks
scaled from randomized common detector footprints including PILATUS 1M, EIGER2,
and a continuous PerkinElmer-style flat panel. The direct-beam, Yoneda,
horizon, and critical-angle operators use the same incident angle and wavelength
stored in the detector geometry. The critical angle is estimated from the parsed
structure electron density unless a profile overrides it. The substrate horizon
model computes beam-footprint spillover from substrate dimensions, beam
width/height, and incident angle, then uses that spillover to tune the horizon
shadow, below-horizon leakage, and added qz peak broadening. The active
surface-scattering operations in this example are {core_surface_text}. Each
artifacted label now includes an `artifact_assessment` block that turns direct
beam, specular reflection, Yoneda bands, substrate horizon/footprint spillage,
critical-angle peak splitting, beamstop shadows, and detector masks into compact
q-space or pixel-space regions. The feedback evaluator and exported structure
guesses rasterize those regions into artifact-aware ranking weights and blend
them with the clean image-overlap score, so non-Bragg aberrations are recognized
without discarding the Bragg-rich regions needed for peak assessment/indexing
tests.

The mathematical surface-artifact contract and artifact-aware training model
are documented in
[Surface Artifacts And Training Model](surface-artifacts-training-model.md).

The augmentation quality gate records `quality_assessment` for every label. In
this smoke run, {quality["solvable_count"]} of {quality["sample_count"]} artifacted
samples are marked solvable, with a median signal-to-noise of
{quality["median_signal_to_noise"]:.2f} and a median retrievable-signal fraction of
{quality["median_retrievable_signal_fraction"]:.3f}. The harsh profile is kept as
a controlled stress-test tail, but its noise, saturation, horizon
spillover, direct-beam, and diffuse-ring strengths are bounded so it remains a
recoverable indexing problem rather than an unrealistic failure case.

## Simulation Examples

![Clean GIWAXS examples](../assets/reports/hybrid3_local_scaffold/{clean_preview.name})

![Missing-wedge correction diagnostic](../assets/reports/hybrid3_local_scaffold/{missing_wedge_preview.name})

![Artifact-augmented examples](../assets/reports/hybrid3_local_scaffold/{artifact_preview.name})

![Surface-scattering diagnostic](../assets/reports/hybrid3_local_scaffold/{surface_preview.name})

## Output Locations

| Output | Path |
| --- | --- |
| Run root | `{output_root}` |
| HybriD3 catalog | `{library_root / "hybrid3_structure_catalog.yaml"}` |
| Ingest manifest | `{library_root / "hybrid3_ingest_manifest.jsonl"}` |
| Clean simulation manifest | `{sim_root / "manifest.jsonl"}` |
| Artifact manifest | `{artifact_root / "artifact_manifest.jsonl"}` |
| Ranker model | `{output_root / "model" / "vector_ranker.json"}` |
| Feedback metrics | `{metrics_path}` |
| Exported guesses | `{guesses_root}` |

## Testing Protocol

1. Verify HybriD3 REST access and select ten visible 2D `PbI4` atomic-structure
   datasets.
2. Download structure-like files from HybriD3 dataset pages and JSmol media
   endpoints.
3. Convert supported structure variants to simulator-readable CIF/POSCAR files.
4. Write an enriched EWALD structure catalog with API metadata and file-derived
   crystallographic metadata.
5. Generate clean GIWAXS simulations using the 2D fibril-texture sweep and
   verify the `simulation_metadata.missing_wedge_correction_applied` labels.
6. Apply deterministic detector and surface-scattering artifacts, then verify
   `artifact_metadata.operations`, `artifact_assessment.regions`, and
   `quality_assessment`.
7. Build the baseline vector-ranker checkpoint from clean simulations.
8. Evaluate artifact images against the ranker and export top-k structure-file
   guesses for downstream structure-analysis testing.

## Detector And Literature Basis

- The PILATUS 1M reference mask follows published modular-detector behavior:
  the original 18-module detector has a large pixel array, module gaps, and
  measurable non-responding pixels that must be masked or treated during data
  reduction.
- The pyFAI detector database provides practical detector definitions and masks
  for many common X-ray detectors, including PILATUS, EIGER, PerkinElmer,
  Rayonix, Lambda, Jungfrau, and other beamline detector families.
- The EIGER2 detector profiles use the DECTRIS-published 75 um pixel size and
  inter-module gap patterns for EIGER2 X detector geometries.
- The GIWAXS protocol follows the perovskite-oriented guidance in
  Steele et al., "How to GIWAXS: Grazing Incidence Wide Angle X-Ray Scattering
  Applied to Metal Halide Perovskite Thin Films," Advanced Energy Materials
  2023, 13, 2300760, including attention to q-range, grazing-incidence
  geometry, detector mapping, and orientation-sensitive interpretation.

References:

- Broennimann et al., "The PILATUS 1M detector,"
  <https://journals.iucr.org/s/issues/2006/02/00/gf0003/>
- pyFAI detector distortion and detector-definition documentation,
  <https://pyfai.readthedocs.io/en/latest/usage/tutorial/Detector/Distortion/Distortion.html>
- DECTRIS EIGER2 X/XE detector specifications,
  <https://www.dectris.com/en/detectors/x-ray-detectors/eiger2/eiger2-for-synchrotrons/eiger2-x/>
- Steele et al., Advanced Energy Materials 2023, DOI 10.1002/aenm.202300760,
  <https://doi.org/10.1002/aenm.202300760>

## Interpretation

This is a scaffold validation, not a final scientific model benchmark. The
baseline vector ranker is intentionally simple: it tests whether the data
contracts, labels, manifests, image generation, artifact augmentation, and
structure-guess export are connected. The next technical step is to replace the
baseline ranker with a learned peak-detection/indexing/retrieval model while
keeping the same manifest and reporting contracts.
""",
        encoding="utf-8",
    )


def _structure_table_row(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    components = metadata.get("components") or {}
    structure_file = metadata.get("structure_file") or {}
    sites = (
        structure_file.get("atom_sites", {}).get("site_count")
        or structure_file.get("site_count")
        or ""
    )
    return (
        f"| {metadata.get('dataset_id', '')} | `{item.get('structure_id', '')}` "
        f"| {components.get('inorganic_formula', '')} "
        f"| {components.get('organic_formula', '')} "
        f"| `{Path(item.get('path', '')).name}` | {sites} |"
    )


def _run(args: list[str], *, cwd: Path) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
