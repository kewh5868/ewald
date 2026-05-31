"""Headless structure-solution benchmark harness.

This module wires together the existing EWALD simulation, peak detection,
lattice indexing, CIF generation, and project I/O primitives.  The benchmark
uses reference CIFs only to generate synthetic measurements and validate the
result afterward.  The solving path receives only the allowed material labels.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from ewald.analysis.structure import (
    DEFAULT_PHASE_TAG,
    MOLECULE_POSE_RESTRAINTS,
    PHASE_REJECTED,
    REFERENCE_MOLECULES,
    CandidateSearchConfig,
    StructurePeak,
    generate_ranked_cif_records,
    guess_lattice_candidates,
)
from ewald.crystallography.cif import compare_cif_atom_coordinates
from ewald.data.models import (
    PEAK_POINT_KIND_COMMITTED,
    STRUCTURE_ANALYSIS_KEY,
    STRUCTURE_ANALYSIS_PEAKS_KEY,
    ImageCorrectionState,
    ProjectState,
)
from ewald.io.project import save_project
from ewald.processing.peak_detection import (
    LocalMaxPeakFinderConfig,
    find_local_maxima_peaks,
)
from ewald.simulation.giwaxs import (
    GIWAXSSimulationParameters,
    PEAK_TABLE_ATTR,
    calculate_giwaxs_peak_rows,
    compare_giwaxs_images,
    save_giwaxs_comparison_plot,
    simulate_giwaxs_image,
)

DEFAULT_SEED = 20260518
DEFAULT_OUTPUT_DIR = Path("example/projects/structure_benchmark")
HALIDE_ELEMENTS = {"I", "Br", "Cl", "F"}
METAL_CATION_ELEMENTS = {"Pb", "Sn", "Ge"}
ORGANIC_HEAVY_ELEMENTS = {"C", "N", "O", "S"}
DONOR_MOLECULES = {
    label
    for label, restraints in MOLECULE_POSE_RESTRAINTS.items()
    if restraints.get("pose_role") == "hydrogen_bond_donor"
}
ACCEPTOR_MOLECULES = {
    label
    for label, restraints in MOLECULE_POSE_RESTRAINTS.items()
    if restraints.get("pose_role") == "hydrogen_bond_acceptor"
}


@dataclass(slots=True)
class BenchmarkStructureSpec:
    """One benchmark structure and the permitted solve constraints."""

    cif_path: Path
    inorganic_atoms: tuple[str, ...]
    organic_molecules: tuple[str, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        self.cif_path = Path(self.cif_path)
        self.inorganic_atoms = tuple(self.inorganic_atoms)
        self.organic_molecules = tuple(self.organic_molecules)

    @property
    def structure_id(self) -> str:
        return _slug(self.label or self.cif_path.stem)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label or self.cif_path.stem,
            "cif_path": str(self.cif_path),
            "inorganic_atoms": list(self.inorganic_atoms),
            "organic_molecules": list(self.organic_molecules),
        }


@dataclass(slots=True)
class BenchmarkRunConfig:
    """Configuration for synthetic benchmark generation and solving."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = DEFAULT_SEED
    simulations_per_structure: int = 1
    hkl_extent: int = 7
    detector_shape: tuple[int, int] = (224, 320)
    qxy_range: tuple[float, float] = (-4.5, 4.5)
    qz_range: tuple[float, float] = (0.0, 4.5)
    sigma_theta: float = 0.035
    sigma_phi: float = 0.35
    sigma_r: float = 0.028
    fiber_tilt_center_deg: float = 90.0
    fiber_tilt_jitter_deg: float = 8.0
    background_level: float = 12.0
    poisson_scale: float = 55000.0
    gaussian_noise_sigma: float = 2.0
    hot_pixel_fraction: float = 0.00012
    peak_threshold_percentile: float = 99.65
    peak_adaptive_threshold: bool = True
    peak_adaptive_floor_percentile: float = 93.0
    peak_min_snr: float = 3.0
    peak_max_peaks: int = 160
    peak_min_distance_px: int = 4
    peak_deduplicate_tolerance: float = 0.035
    candidate_hkl_max: int = 7
    candidate_q_tolerance: float = 0.065
    candidate_relative_tolerance: float = 0.04
    candidate_lattice_min: float = 2.5
    candidate_lattice_max: float = 60.0
    candidate_grid_points: int = 14
    candidate_max_candidates: int = 12
    cif_records_per_candidate: int = 2
    max_generated_cifs_to_compare: int = 8
    comparison_theta_x_offsets: tuple[float, ...] = (-8.0, 0.0, 8.0)
    comparison_theta_y_values: tuple[float, ...] = (
        0.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
    )
    comparison_plot_count: int = 3
    staged_refinement: bool = True
    refinement_coarse_detector_shape: tuple[int, int] = (64, 96)
    refinement_coarse_hkl_extent: int = 4
    refinement_fractional_step: float = 0.08
    refinement_scaffold_variant_count: int = 0
    bragg_intensity_weight: float = 0.35
    bragg_intensity_tolerance: float = 0.08
    bragg_intensity_max_peaks: int = 80
    synthetic_lattice_disorder_fraction: float = 0.0
    synthetic_peak_dropout_fraction: float = 0.0
    synthetic_detector_gap_qxy_width: float = 0.0
    synthetic_detector_gap_qz_width: float = 0.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.detector_shape = (
            int(self.detector_shape[0]),
            int(self.detector_shape[1]),
        )
        self.qxy_range = (float(self.qxy_range[0]), float(self.qxy_range[1]))
        self.qz_range = (float(self.qz_range[0]), float(self.qz_range[1]))
        self.comparison_theta_x_offsets = tuple(
            float(value) for value in self.comparison_theta_x_offsets
        )
        self.comparison_theta_y_values = tuple(
            float(value) for value in self.comparison_theta_y_values
        )
        self.refinement_coarse_detector_shape = (
            int(self.refinement_coarse_detector_shape[0]),
            int(self.refinement_coarse_detector_shape[1]),
        )
        self.refinement_scaffold_variant_count = max(
            0,
            int(self.refinement_scaffold_variant_count),
        )
        self.bragg_intensity_weight = max(
            0.0, float(self.bragg_intensity_weight)
        )
        self.bragg_intensity_tolerance = max(
            1.0e-9, float(self.bragg_intensity_tolerance)
        )
        self.bragg_intensity_max_peaks = max(
            1, int(self.bragg_intensity_max_peaks)
        )
        self.synthetic_lattice_disorder_fraction = max(
            0.0, float(self.synthetic_lattice_disorder_fraction)
        )
        self.synthetic_peak_dropout_fraction = min(
            0.95, max(0.0, float(self.synthetic_peak_dropout_fraction))
        )
        self.synthetic_detector_gap_qxy_width = max(
            0.0, float(self.synthetic_detector_gap_qxy_width)
        )
        self.synthetic_detector_gap_qz_width = max(
            0.0, float(self.synthetic_detector_gap_qz_width)
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["detector_shape"] = list(self.detector_shape)
        payload["qxy_range"] = list(self.qxy_range)
        payload["qz_range"] = list(self.qz_range)
        payload["comparison_theta_x_offsets"] = list(
            self.comparison_theta_x_offsets
        )
        payload["comparison_theta_y_values"] = list(
            self.comparison_theta_y_values
        )
        payload["refinement_coarse_detector_shape"] = list(
            self.refinement_coarse_detector_shape
        )
        return payload


@dataclass(slots=True)
class BenchmarkRunResult:
    """Summary of one benchmark run."""

    run_id: str
    output_dir: Path
    filesets: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "filesets": self.filesets,
        }


def default_structure_specs(
    structures_dir: str | Path = "example/structures",
) -> list[BenchmarkStructureSpec]:
    """Return the provided lead-halide benchmark structures."""

    root = Path(structures_dir)
    entries = [
        (
            "(MA)2(DMF)2Pb2I6_2017Tarasov.cif",
            ("Pb", "I"),
            ("MA", "DMF"),
        ),
        (
            "(MA)2(DMF)2Pb3I8_2017Tarasov.cif",
            ("Pb", "I"),
            ("MA", "DMF"),
        ),
        (
            "(MA)3(DMF)PbI5_2017Tarasov.cif",
            ("Pb", "I"),
            ("MA", "DMF"),
        ),
        ("2hPbI2.cif", ("Pb", "I"), ()),
        (
            "BA_PbI3_1DMF_Dahlman_2019.cif",
            ("Pb", "I"),
            ("BA", "DMF"),
        ),
        (
            "FA2Pb3I8-4DMF-2022-Petrov.cif",
            ("Pb", "I"),
            ("FA", "DMF"),
        ),
        (
            "FA2PbBr4_DMSO-2022-Petrov.cif",
            ("Pb", "Br"),
            ("FA", "DMSO"),
        ),
        (
            "FA5Pb2I9-0_5DMSO-2022-Petrov.cif",
            ("Pb", "I"),
            ("FA", "DMSO"),
        ),
        (
            "FAPbI3-2DMF-2022-Petrov.cif",
            ("Pb", "I"),
            ("FA", "DMF"),
        ),
        (
            "MA2_PbI3I8_2DMSO_Nanfeng_2016.cif",
            ("Pb", "I"),
            ("MA", "DMSO"),
        ),
        (
            "MA3_PbI5_1DMSO_Nanfeng_2016.cif",
            ("Pb", "I"),
            ("MA", "DMSO"),
        ),
        (
            "PbI2_2DMSO_Nanfeng_2016.cif",
            ("Pb", "I"),
            ("DMSO",),
        ),
        (
            "PbI2_DMF_Nanfeng_2016.cif",
            ("Pb", "I"),
            ("DMF",),
        ),
        (
            "PbI2_DMSO_Nanfeng_2016.cif",
            ("Pb", "I"),
            ("DMSO",),
        ),
        (
            "PbI2_NMP_2019_NanfengZheng.cif",
            ("Pb", "I"),
            ("MA", "NMP"),
        ),
    ]
    return [
        BenchmarkStructureSpec(
            cif_path=root / filename,
            inorganic_atoms=atoms,
            organic_molecules=molecules,
        )
        for filename, atoms, molecules in entries
    ]


def load_structure_specs(path: str | Path) -> list[BenchmarkStructureSpec]:
    """Load benchmark structure specs from JSON.

    The file may either be a list of entries or a mapping with a
    ``"structures"`` list.  Each entry accepts ``cif_path`` or ``path``,
    ``inorganic_atoms`` or ``inorganic``, and ``organic_molecules`` or
    ``organic``.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = (
        payload.get("structures", payload)
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(entries, list):
        raise ValueError("Benchmark structure manifest must contain a list.")
    specs = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                "Each benchmark structure entry must be a mapping."
            )
        cif_path = entry.get("cif_path") or entry.get("path")
        if not cif_path:
            raise ValueError("Benchmark structure entries require cif_path.")
        specs.append(
            BenchmarkStructureSpec(
                cif_path=Path(str(cif_path)),
                inorganic_atoms=tuple(
                    str(item)
                    for item in entry.get(
                        "inorganic_atoms",
                        entry.get("inorganic", ()),
                    )
                ),
                organic_molecules=tuple(
                    str(item)
                    for item in entry.get(
                        "organic_molecules",
                        entry.get("organic", ()),
                    )
                ),
                label=entry.get("label"),
            )
        )
    return specs


def run_structure_benchmark(
    specs: Iterable[BenchmarkStructureSpec],
    config: BenchmarkRunConfig | None = None,
) -> BenchmarkRunResult:
    """Generate and solve synthetic benchmark filesets."""

    cfg = config or BenchmarkRunConfig()
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_root = cfg.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    logbook = run_root / "LOGBOOK.md"
    rng = np.random.default_rng(cfg.seed)
    normalized_specs = [spec for spec in specs if spec.cif_path.exists()]
    _write_json(
        run_root / "benchmark_manifest.json",
        {
            "run_id": run_id,
            "created_at": _timestamp(),
            "config": cfg.as_dict(),
            "structures": [spec.as_dict() for spec in normalized_specs],
        },
    )
    _append_logbook(
        logbook,
        [
            f"# EWALD Structure Benchmark {run_id}",
            "",
            f"- Created: {_timestamp()}",
            f"- Seed: {cfg.seed}",
            f"- Structure count: {len(normalized_specs)}",
            (
                "- Solver input policy: inorganic atom labels and organic "
                "molecule labels only; reference CIF cell, stoichiometry, and "
                "coordinates are used only for simulation and validation."
            ),
            "",
        ],
    )

    tasks = [
        (spec, simulation_index)
        for spec in normalized_specs
        for simulation_index in range(
            1, max(1, cfg.simulations_per_structure) + 1
        )
    ]
    rng.shuffle(tasks)
    result = BenchmarkRunResult(run_id=run_id, output_dir=run_root)
    for order_index, (spec, simulation_index) in enumerate(tasks, start=1):
        fileset = _run_one_fileset(
            spec,
            simulation_index,
            order_index,
            run_root,
            logbook,
            rng,
            cfg,
        )
        result.filesets.append(fileset)
        _write_json(run_root / "summary.json", result.as_dict())

    _append_logbook(
        logbook,
        [
            "",
            "## Run Summary",
            "",
            _markdown_summary_table(result.filesets),
            "",
        ],
    )
    return result


def _run_one_fileset(
    spec: BenchmarkStructureSpec,
    simulation_index: int,
    order_index: int,
    run_root: Path,
    logbook: Path,
    rng: np.random.Generator,
    cfg: BenchmarkRunConfig,
) -> dict[str, Any]:
    fileset_id = f"{spec.structure_id}_sim{simulation_index:02d}"
    fileset_dir = run_root / spec.structure_id / f"sim_{simulation_index:02d}"
    fileset_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = fileset_dir / "simulations"
    qspace_dir = fileset_dir / "qspace"
    plots_dir = fileset_dir / "plots"
    rankings_dir = fileset_dir / "rankings"
    generated_dir = fileset_dir / "generated_structures"
    best_dir = fileset_dir / "best_fit_generated_structures"
    for directory in (
        clean_dir,
        qspace_dir,
        plots_dir,
        rankings_dir,
        generated_dir,
        best_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    params = _random_fiber_parameters(rng, cfg)
    simulation_cif_path = _write_lattice_disordered_cif(
        spec.cif_path,
        clean_dir / f"{fileset_id}_simulation_lattice.cif",
        rng,
        cfg,
    )
    clean = simulate_giwaxs_image(simulation_cif_path, params)
    clean_nc = clean_dir / f"{fileset_id}_clean.nc"
    clean.to_netcdf(clean_nc)
    truth_peak_path = clean_dir / f"{fileset_id}_truth_peaks.json"
    truth_peaks = _write_truth_peak_table(truth_peak_path, clean, cfg)
    noisy = _mock_experimental_qspace(clean, rng, cfg)
    tiff_path = fileset_dir / f"{fileset_id}_mock_experiment.tiff"
    qspace_path = qspace_dir / f"{fileset_id}_mock_qspace.npz"
    _write_mock_tiff(tiff_path, noisy, spec, params)
    _write_qspace_npz(qspace_path, noisy, spec, params)

    project = ProjectState(name=f"Benchmark {fileset_id}")
    data_file = project.add_data_file(
        tiff_path,
        benchmark_fileset_id=fileset_id,
        reference_cif=str(spec.cif_path),
        synthetic=True,
    )
    project.processed_products[data_file.data_id] = str(qspace_path)
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=str(data_file.data_id),
            confirmed=True,
            metadata={
                "benchmark_fileset_id": fileset_id,
                "qspace_product": str(qspace_path),
            },
        )
    )
    project.simulations[f"{fileset_id}_reference"] = {
        "simulation_id": f"{fileset_id}_reference",
        "simulation_mode": "benchmark_reference_clean",
        "data_id": data_file.data_id,
        "structure_name": spec.label or spec.cif_path.stem,
        "structure_path": str(simulation_cif_path),
        "cif_path": str(simulation_cif_path),
        "dataset_uri": str(clean_nc),
        "parameters": params.as_dict(),
        "metadata": {
            "created_at": _timestamp(),
            "peak_count": clean.attrs.get("peak_count", 0),
            "synthetic_measurement": str(tiff_path),
            "reference_validation_cif": str(spec.cif_path),
            "truth_peak_table": str(truth_peak_path),
            "lattice_disorder_fraction": (
                cfg.synthetic_lattice_disorder_fraction
            ),
            "peak_dropout_fraction": cfg.synthetic_peak_dropout_fraction,
            "detector_gap_qxy_width": cfg.synthetic_detector_gap_qxy_width,
            "detector_gap_qz_width": cfg.synthetic_detector_gap_qz_width,
        },
    }

    peaks = _detect_structure_peaks(noisy, cfg)
    peak_recovery = _peak_detection_recovery_metrics(peaks, truth_peaks, cfg)
    peak_plot = _save_peak_detection_plot(
        noisy,
        peaks,
        plots_dir / "peak_detection.png",
        title=fileset_id,
    )
    _store_peak_records(project, str(data_file.data_id), peaks)
    candidates = guess_lattice_candidates(
        peaks,
        _candidate_search_config(cfg),
    )
    candidate_records = [candidate.as_dict() for candidate in candidates]
    _write_json(rankings_dir / "lattice_candidates.json", candidate_records)

    generated_records = _generate_candidate_cifs(spec, candidates, cfg)
    generated_records = _materialize_generated_cifs(
        generated_records,
        generated_dir,
        data_file_id=str(data_file.data_id),
        fileset_id=fileset_id,
    )
    project.reference_cifs.setdefault("generated", {}).update(
        {str(record["cif_id"]): dict(record) for record in generated_records}
    )
    for record in generated_records:
        project.structures[str(record["cif_id"])] = dict(record)

    comparisons = _rank_generated_cifs(
        noisy,
        generated_records,
        params,
        cfg,
        rankings_dir,
        plots_dir,
        peak_intensity_peaks=peaks,
    )
    _attach_refined_cif_records(project, comparisons)
    validation = _validate_best_generated_structure(
        comparisons,
        spec.cif_path,
        best_dir,
        run_root / "best_fit_generated_structures" / fileset_id,
    )
    project.analysis_results.setdefault("benchmark", {})[fileset_id] = {
        "fileset_id": fileset_id,
        "solve_order": order_index,
        "constraints": _solve_constraints(spec),
        "peak_count": len(peaks),
        "truth_peak_count": len(truth_peaks),
        "peak_recovery": peak_recovery,
        "lattice_candidates": candidate_records,
        "generated_cif_rankings": comparisons,
        "validation": validation,
    }
    project_path = save_project(project, fileset_dir / f"{fileset_id}.ewld")
    fileset_summary = {
        "fileset_id": fileset_id,
        "structure": spec.as_dict(),
        "solve_order": order_index,
        "simulation_index": simulation_index,
        "mock_tiff": str(tiff_path),
        "mock_qspace": str(qspace_path),
        "clean_simulation": str(clean_nc),
        "simulation_cif": str(simulation_cif_path),
        "truth_peak_table": str(truth_peak_path),
        "peak_detection_plot": str(peak_plot),
        "project": str(project_path),
        "readable_project": str(project_path.with_suffix(".ewald.json")),
        "peak_count": len(peaks),
        "truth_peak_count": len(truth_peaks),
        "peak_recovery": peak_recovery,
        "top_candidate": candidate_records[0] if candidate_records else None,
        "best_generated_cif": validation.get("best_generated_cif"),
        "validation": validation,
    }
    _write_json(rankings_dir / "validation.json", validation)
    _append_fileset_logbook(
        logbook,
        fileset_summary,
        params,
        peaks,
        candidate_records,
        comparisons,
    )
    return fileset_summary


def _random_fiber_parameters(
    rng: np.random.Generator,
    cfg: BenchmarkRunConfig,
) -> GIWAXSSimulationParameters:
    theta_x = cfg.fiber_tilt_center_deg + float(
        rng.uniform(-cfg.fiber_tilt_jitter_deg, cfg.fiber_tilt_jitter_deg)
    )
    theta_y = float(rng.uniform(0.0, 360.0))
    return GIWAXSSimulationParameters(
        sigma_theta=cfg.sigma_theta,
        sigma_phi=cfg.sigma_phi,
        sigma_r=cfg.sigma_r,
        hkl_extent=cfg.hkl_extent,
        theta_x_deg=theta_x,
        theta_y_deg=theta_y,
        qxy_min=cfg.qxy_range[0],
        qxy_max=cfg.qxy_range[1],
        qz_min=cfg.qz_range[0],
        qz_max=cfg.qz_range[1],
        resolution_z=cfg.detector_shape[0],
        resolution_x=cfg.detector_shape[1],
    )


def _mock_experimental_qspace(
    clean: xr.DataArray,
    rng: np.random.Generator,
    cfg: BenchmarkRunConfig,
) -> xr.DataArray:
    values = np.nan_to_num(np.asarray(clean.values, dtype=float), nan=0.0)
    positive = values[values > 0.0]
    if positive.size:
        scale = float(np.nanpercentile(positive, 99.7))
        scale = (
            scale
            if np.isfinite(scale) and scale > 0.0
            else float(np.nanmax(positive))
        )
    else:
        scale = 1.0
    normalized = np.clip(values / max(scale, 1.0e-12), 0.0, None)
    y_grid, x_grid = np.mgrid[: values.shape[0], : values.shape[1]]
    gradient = 1.0 + 0.04 * x_grid / max(values.shape[1] - 1, 1)
    gradient += 0.03 * y_grid / max(values.shape[0] - 1, 1)
    expectation = cfg.background_level * gradient
    expectation += normalized * cfg.poisson_scale
    sampled = rng.poisson(np.clip(expectation, 0.0, 1.0e7)).astype(float)
    if cfg.gaussian_noise_sigma > 0.0:
        sampled += rng.normal(0.0, cfg.gaussian_noise_sigma, sampled.shape)
    hot_count = int(round(sampled.size * max(cfg.hot_pixel_fraction, 0.0)))
    if hot_count > 0:
        flat = sampled.ravel()
        indices = rng.choice(flat.size, size=hot_count, replace=False)
        flat[indices] += rng.uniform(
            cfg.poisson_scale * 0.05,
            cfg.poisson_scale * 0.25,
            size=hot_count,
        )
    sampled = _apply_synthetic_peak_dropout(sampled, clean, rng, cfg)
    sampled = _apply_synthetic_detector_gaps(sampled, clean, cfg)
    return xr.DataArray(
        np.clip(sampled, 0.0, None),
        dims=clean.dims,
        coords=clean.coords,
        name="mock_experimental_intensity",
        attrs={
            **dict(clean.attrs),
            "benchmark_noise_model": "poisson_background_gaussian_hot_pixels",
            "mock_background_level": cfg.background_level,
            "mock_poisson_scale": cfg.poisson_scale,
            "mock_gaussian_noise_sigma": cfg.gaussian_noise_sigma,
            "synthetic_peak_dropout_fraction": (
                cfg.synthetic_peak_dropout_fraction
            ),
            "synthetic_detector_gap_qxy_width": (
                cfg.synthetic_detector_gap_qxy_width
            ),
            "synthetic_detector_gap_qz_width": cfg.synthetic_detector_gap_qz_width,
        },
    )


def _write_lattice_disordered_cif(
    source_path: Path,
    output_path: Path,
    rng: np.random.Generator,
    cfg: BenchmarkRunConfig,
) -> Path:
    fraction = float(cfg.synthetic_lattice_disorder_fraction)
    if fraction <= 0.0:
        return source_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    multipliers = {
        "_cell_length_a": 1.0 + float(rng.normal(0.0, fraction)),
        "_cell_length_b": 1.0 + float(rng.normal(0.0, fraction)),
        "_cell_length_c": 1.0 + float(rng.normal(0.0, fraction)),
    }
    lines = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        key = stripped.split()[0] if stripped else ""
        if key in multipliers:
            parts = stripped.split()
            try:
                value = _cif_number(parts[1])
            except (IndexError, ValueError):
                lines.append(line)
                continue
            lines.append(f"{key} {value * multipliers[key]:.6f}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _write_truth_peak_table(
    path: Path,
    clean: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> list[dict[str, Any]]:
    peaks = _truth_peak_rows(clean, cfg)
    _write_json(
        path,
        {
            "source": "simulated_peak_table",
            "q_tolerance": _truth_peak_tolerance(cfg),
            "peak_count": len(peaks),
            "peaks": peaks,
        },
    )
    return peaks


def _truth_peak_rows(
    clean: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> list[dict[str, Any]]:
    try:
        rows = json.loads(str(clean.attrs.get(PEAK_TABLE_ATTR) or "[]"))
    except json.JSONDecodeError:
        rows = []
    tolerance = _truth_peak_tolerance(cfg)
    deduped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            qxy = abs(float(row["qxy"]))
            qz = float(row["qz"])
            amplitude = float(row.get("amplitude", row.get("intensity", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        key = (int(round(qxy / tolerance)), int(round(qz / tolerance)))
        current = deduped.get(key)
        if current is None or amplitude > float(current.get("amplitude", 0.0)):
            deduped[key] = {
                "h": int(row.get("h", 0)),
                "k": int(row.get("k", 0)),
                "l": int(row.get("l", 0)),
                "qxy": qxy,
                "qz": qz,
                "intensity": float(row.get("intensity", amplitude)),
                "amplitude": amplitude,
            }
    return sorted(
        deduped.values(),
        key=lambda item: float(item.get("amplitude", 0.0)),
        reverse=True,
    )


def _apply_synthetic_peak_dropout(
    sampled: np.ndarray,
    clean: xr.DataArray,
    rng: np.random.Generator,
    cfg: BenchmarkRunConfig,
) -> np.ndarray:
    dropout = float(cfg.synthetic_peak_dropout_fraction)
    if dropout <= 0.0:
        return sampled
    truth = _truth_peak_rows(clean, cfg)
    if not truth:
        return sampled
    count = int(round(len(truth) * dropout))
    if count <= 0:
        return sampled
    selected = rng.choice(
        len(truth), size=min(count, len(truth)), replace=False
    )
    result = np.array(sampled, copy=True)
    qxy_axis = np.asarray(clean.coords["qxy"].values, dtype=float)
    qz_axis = np.asarray(clean.coords["qz"].values, dtype=float)
    qxy_width = max(cfg.sigma_r * 3.0, _axis_step(qxy_axis) * 2.0)
    qz_width = max(cfg.sigma_theta * 3.0, _axis_step(qz_axis) * 2.0)
    for index in selected:
        peak = truth[int(index)]
        qxy_mask = np.abs(qxy_axis - float(peak["qxy"])) <= qxy_width
        qxy_mask |= np.abs(qxy_axis + float(peak["qxy"])) <= qxy_width
        qz_mask = np.abs(qz_axis - float(peak["qz"])) <= qz_width
        if qxy_mask.any() and qz_mask.any():
            result[np.ix_(qz_mask, qxy_mask)] *= 0.04
    return result


def _apply_synthetic_detector_gaps(
    sampled: np.ndarray,
    clean: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> np.ndarray:
    result = np.array(sampled, copy=True)
    if cfg.synthetic_detector_gap_qxy_width > 0.0:
        qxy_axis = np.asarray(clean.coords["qxy"].values, dtype=float)
        mask = np.abs(qxy_axis) <= cfg.synthetic_detector_gap_qxy_width / 2.0
        if mask.any():
            result[:, mask] = 0.0
    if cfg.synthetic_detector_gap_qz_width > 0.0:
        qz_axis = np.asarray(clean.coords["qz"].values, dtype=float)
        center = float(np.nanmedian(qz_axis))
        mask = (
            np.abs(qz_axis - center)
            <= cfg.synthetic_detector_gap_qz_width / 2.0
        )
        if mask.any():
            result[mask, :] = 0.0
    return result


def _peak_detection_recovery_metrics(
    detected: list[StructurePeak],
    truth: list[dict[str, Any]],
    cfg: BenchmarkRunConfig,
) -> dict[str, Any]:
    if not truth:
        return {"truth_peak_count": 0, "detected_peak_count": len(detected)}
    tolerance = max(_truth_peak_tolerance(cfg), cfg.peak_deduplicate_tolerance)
    matched_truth: set[int] = set()
    stray_detected = 0
    for peak in detected:
        best_index = None
        best_distance = math.inf
        for index, truth_peak in enumerate(truth):
            distance = math.hypot(
                abs(float(peak.qxy)) - float(truth_peak["qxy"]),
                float(peak.qz) - float(truth_peak["qz"]),
            )
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= tolerance:
            matched_truth.add(best_index)
        else:
            stray_detected += 1
    return {
        "truth_peak_count": len(truth),
        "detected_peak_count": len(detected),
        "matched_truth_peak_count": len(matched_truth),
        "missed_truth_peak_count": max(0, len(truth) - len(matched_truth)),
        "stray_detected_peak_count": stray_detected,
        "recall": len(matched_truth) / max(len(truth), 1),
        "precision": (
            (len(detected) - stray_detected) / max(len(detected), 1)
            if detected
            else 0.0
        ),
        "match_tolerance": tolerance,
    }


def _truth_peak_tolerance(cfg: BenchmarkRunConfig) -> float:
    return max(cfg.sigma_r * 2.0, cfg.sigma_theta * 2.0, 0.025)


def _axis_step(axis: np.ndarray) -> float:
    if axis.size < 2:
        return 0.0
    return float(np.nanmedian(np.abs(np.diff(axis))))


def _cif_number(value: str) -> float:
    token = str(value).strip().split("(")[0]
    return float(token)


def _write_mock_tiff(
    path: Path,
    image: xr.DataArray,
    spec: BenchmarkStructureSpec,
    params: GIWAXSSimulationParameters,
) -> None:
    import tifffile

    values = np.asarray(image.values, dtype=float)
    finite = values[np.isfinite(values)]
    upper = float(np.nanpercentile(finite, 99.95)) if finite.size else 1.0
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0
    scaled = np.clip(values / upper * 65535.0, 0.0, 65535.0).astype(np.uint16)
    metadata = {
        "benchmark": "ewald_structure_benchmark",
        "reference_cif": str(spec.cif_path),
        "solve_constraints": _solve_constraints(spec),
        "simulation_parameters": params.as_dict(),
    }
    tifffile.imwrite(
        path, scaled, description=json.dumps(metadata, sort_keys=True)
    )


def _write_qspace_npz(
    path: Path,
    image: xr.DataArray,
    spec: BenchmarkStructureSpec,
    params: GIWAXSSimulationParameters,
) -> None:
    np.savez(
        path,
        intensity=np.asarray(image.values, dtype=float),
        q_ip=np.asarray(image.coords["qxy"].values, dtype=float),
        q_oop=np.asarray(image.coords["qz"].values, dtype=float),
        metadata_json=json.dumps(
            {
                "benchmark": "ewald_structure_benchmark",
                "reference_cif": str(spec.cif_path),
                "solve_constraints": _solve_constraints(spec),
                "simulation_parameters": params.as_dict(),
            },
            sort_keys=True,
        ),
    )


def _detect_structure_peaks(
    image: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> list[StructurePeak]:
    raw = find_local_maxima_peaks(
        image.values,
        x_axis=image.coords["qxy"].values,
        y_axis=image.coords["qz"].values,
        config=LocalMaxPeakFinderConfig(
            threshold_percentile=cfg.peak_threshold_percentile,
            adaptive_threshold=cfg.peak_adaptive_threshold,
            adaptive_floor_percentile=cfg.peak_adaptive_floor_percentile,
            min_snr=cfg.peak_min_snr,
            max_peaks=cfg.peak_max_peaks,
            min_distance_px=cfg.peak_min_distance_px,
            neighborhood_radius_px=1,
        ),
    )
    deduped: dict[tuple[int, int], Any] = {}
    tolerance = max(float(cfg.peak_deduplicate_tolerance), 1.0e-9)
    for peak in raw:
        if peak.y < 0.0:
            continue
        key = (
            int(round(abs(peak.x) / tolerance)),
            int(round(peak.y / tolerance)),
        )
        current = deduped.get(key)
        if current is None or peak.intensity > current.intensity:
            deduped[key] = peak
    ordered = sorted(
        deduped.values(),
        key=lambda item: float(item.intensity),
        reverse=True,
    )[: cfg.peak_max_peaks]
    return [
        StructurePeak(
            peak_id=f"peak_{index:03d}",
            label=f"Peak {index:03d}",
            qxy=abs(float(peak.x)),
            qz=float(peak.y),
            source="benchmark_peak_detection",
            phase_tag=DEFAULT_PHASE_TAG,
            fit_quality=peak.score,
            metadata={
                "detected_intensity": peak.intensity,
                "background": peak.background,
                "noise": peak.noise,
                "snr": peak.snr,
                "prominence": peak.prominence,
            },
        )
        for index, peak in enumerate(ordered, start=1)
    ]


def _bragg_peak_intensity_match_for_image(
    peaks: list[StructurePeak] | None,
    simulated: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> dict[str, Any]:
    if not peaks:
        return _empty_bragg_intensity_match("no detected peaks")
    try:
        rows = json.loads(str(simulated.attrs.get(PEAK_TABLE_ATTR) or "[]"))
    except json.JSONDecodeError:
        rows = []
    return _score_bragg_peak_intensity_match(
        peaks,
        rows,
        tolerance=cfg.bragg_intensity_tolerance,
        max_peaks=cfg.bragg_intensity_max_peaks,
    )


def _score_bragg_peak_intensity_match(
    peaks: Iterable[StructurePeak],
    simulated_peak_rows: Iterable[dict[str, Any]],
    *,
    tolerance: float,
    max_peaks: int,
) -> dict[str, Any]:
    """Compare relative observed and simulated Bragg peak intensities."""

    observed = [
        (peak, _observed_peak_intensity(peak))
        for peak in peaks
        if peak.include
        and peak.phase_tag != PHASE_REJECTED
        and np.isfinite(float(peak.q_magnitude))
    ]
    observed = [
        (peak, intensity)
        for peak, intensity in observed
        if np.isfinite(intensity) and intensity > 0.0
    ]
    observed.sort(key=lambda item: item[1], reverse=True)
    observed = observed[: max(1, int(max_peaks))]
    if len(observed) < 2:
        return _empty_bragg_intensity_match(
            "fewer than two observed peaks with positive intensity"
        )

    rows = [
        row for row in simulated_peak_rows if _simulated_peak_intensity(row) > 0.0
    ]
    if not rows:
        return {
            **_empty_bragg_intensity_match(
                "no simulated Bragg peaks with positive intensity"
            ),
            "compared_peak_count": len(observed),
            "intensity_match_penalty": 2.0,
        }

    sim_qxy = np.asarray(
        [abs(float(row.get("qxy", 0.0))) for row in rows],
        dtype=float,
    )
    sim_qz = np.asarray(
        [float(row.get("qz", 0.0)) for row in rows],
        dtype=float,
    )
    sim_intensity = np.asarray(
        [_simulated_peak_intensity(row) for row in rows],
        dtype=float,
    )
    tolerance = max(float(tolerance), 1.0e-9)
    sigma = max(tolerance * 0.5, 1.0e-9)
    observed_values: list[float] = []
    predicted_values: list[float] = []
    peak_records: list[dict[str, Any]] = []
    for peak, observed_intensity in observed:
        distances = np.hypot(
            sim_qxy - abs(float(peak.qxy)),
            sim_qz - float(peak.qz),
        )
        mask = distances <= tolerance
        predicted = 0.0
        best_hkl = None
        best_distance = None
        if np.any(mask):
            weights = np.exp(-0.5 * (distances[mask] / sigma) ** 2)
            predicted = float(np.sum(sim_intensity[mask] * weights))
            best_local = int(np.argmax(sim_intensity[mask] * weights))
            masked_indices = np.flatnonzero(mask)
            best_index = int(masked_indices[best_local])
            best_row = rows[best_index]
            best_hkl = [
                int(best_row.get("h", 0)),
                int(best_row.get("k", 0)),
                int(best_row.get("l", 0)),
            ]
            best_distance = float(distances[best_index])
        observed_values.append(float(observed_intensity))
        predicted_values.append(float(predicted))
        if len(peak_records) < 20:
            peak_records.append(
                {
                    "peak_id": peak.peak_id,
                    "label": peak.label,
                    "qxy": float(peak.qxy),
                    "qz": float(peak.qz),
                    "observed_intensity": float(observed_intensity),
                    "predicted_intensity": float(predicted),
                    "best_hkl": best_hkl,
                    "best_distance": best_distance,
                }
            )

    observed_array = np.asarray(observed_values, dtype=float)
    predicted_array = np.asarray(predicted_values, dtype=float)
    matched = predicted_array > 0.0
    observed_norm = observed_array / max(float(np.sum(observed_array)), 1.0e-12)
    if float(np.sum(predicted_array)) > 0.0:
        predicted_norm = predicted_array / float(np.sum(predicted_array))
    else:
        predicted_norm = np.zeros_like(predicted_array)
    relative_l1 = 0.5 * float(np.sum(np.abs(observed_norm - predicted_norm)))
    eps = 1.0e-12
    log_observed = np.log(observed_norm + eps)
    log_predicted = np.log(predicted_norm + eps)
    log_rmse = float(np.sqrt(np.mean((log_observed - log_predicted) ** 2)))
    log_mae = float(np.mean(np.abs(log_observed - log_predicted)))
    correlation = _safe_correlation(log_observed, log_predicted)
    matched_fraction = float(np.count_nonzero(matched)) / max(
        predicted_array.size,
        1,
    )
    correlation_penalty = 1.0 - max(correlation, 0.0)
    penalty = min(
        2.0,
        relative_l1
        + 0.15 * min(log_rmse, 4.0) / 4.0
        + 0.25 * correlation_penalty
        + 0.30 * (1.0 - matched_fraction),
    )
    return {
        "status": "computed",
        "observed_intensity_model": (
            "ROI integrated intensity when available; detected peak intensity "
            "fallback"
        ),
        "simulated_intensity_model": (
            "texture-weighted calculated Bragg amplitudes summed within "
            "q tolerance"
        ),
        "compared_peak_count": int(predicted_array.size),
        "matched_peak_count": int(np.count_nonzero(matched)),
        "matched_peak_fraction": matched_fraction,
        "tolerance": tolerance,
        "relative_l1": relative_l1,
        "log_intensity_rmse": log_rmse,
        "log_intensity_mae": log_mae,
        "log_intensity_correlation": correlation,
        "intensity_match_penalty": penalty,
        "peaks": peak_records,
    }


def _empty_bragg_intensity_match(reason: str) -> dict[str, Any]:
    return {
        "status": "not_computed",
        "reason": reason,
        "compared_peak_count": 0,
        "matched_peak_count": 0,
        "matched_peak_fraction": 0.0,
        "relative_l1": None,
        "log_intensity_rmse": None,
        "log_intensity_mae": None,
        "log_intensity_correlation": None,
        "intensity_match_penalty": 0.0,
        "peaks": [],
    }


def _bragg_intensity_metric_summary(
    match: dict[str, Any],
    weighted_penalty: float,
) -> dict[str, Any]:
    return {
        "bragg_intensity_weighted_penalty": float(weighted_penalty),
        "bragg_intensity_match_penalty": match.get(
            "intensity_match_penalty"
        ),
        "bragg_intensity_relative_l1": match.get("relative_l1"),
        "bragg_intensity_log_rmse": match.get("log_intensity_rmse"),
        "bragg_intensity_correlation": match.get(
            "log_intensity_correlation"
        ),
        "bragg_intensity_matched_fraction": match.get(
            "matched_peak_fraction"
        ),
    }


def _observed_peak_intensity(peak: StructurePeak) -> float:
    metadata = dict(peak.metadata or {})
    candidates: list[Any] = []
    nested = metadata.get("integrated_intensity")
    if isinstance(nested, dict):
        candidates.extend(
            [
                nested.get("integrated_intensity"),
                nested.get("area_scaled_integrated_intensity"),
            ]
        )
    candidates.extend(
        [
            metadata.get("net_integrated_intensity"),
            metadata.get("background_subtracted_integrated_intensity"),
            metadata.get("integrated_intensity"),
            metadata.get("area_scaled_integrated_intensity"),
            metadata.get("roi_integrated_intensity"),
            metadata.get("detected_integrated_intensity"),
            metadata.get("detected_intensity"),
            metadata.get("prominence"),
            peak.fit_quality,
        ]
    )
    for value in candidates:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0.0:
            return parsed
    return 0.0


def _simulated_peak_intensity(row: dict[str, Any]) -> float:
    for key in ("amplitude", "intensity"):
        try:
            parsed = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0.0:
            return parsed
    return 0.0


def _safe_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if float(np.nanstd(x)) <= 1.0e-12:
        return 0.0
    if float(np.nanstd(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _store_peak_records(
    project: ProjectState,
    data_id: str,
    peaks: list[StructurePeak],
) -> None:
    project.peak_sets[data_id] = [
        {
            "peak_id": peak.peak_id,
            "label": peak.label,
            "qxy": peak.qxy,
            "qz": peak.qz,
            "source": peak.source,
            "point_kind": PEAK_POINT_KIND_COMMITTED,
            "phase_tag": peak.phase_tag,
            "include": peak.include,
            "metadata": dict(peak.metadata),
        }
        for peak in peaks
    ]
    analysis = project.analysis_results.setdefault(STRUCTURE_ANALYSIS_KEY, {})
    analysis[data_id] = {
        STRUCTURE_ANALYSIS_PEAKS_KEY: [peak.as_dict() for peak in peaks],
        "candidates": [],
        "families": [],
        "wyckoff": {},
    }


def _save_peak_detection_plot(
    image: xr.DataArray,
    peaks: list[StructurePeak],
    output_path: Path,
    *,
    title: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(image.values, dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite, 99.7)) if finite.size else None
    extent = (
        float(np.nanmin(image.coords["qxy"].values)),
        float(np.nanmax(image.coords["qxy"].values)),
        float(np.nanmin(image.coords["qz"].values)),
        float(np.nanmax(image.coords["qz"].values)),
    )
    figure, axis = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax if vmax and np.isfinite(vmax) and vmax > 0.0 else None,
    )
    if peaks:
        axis.scatter(
            [peak.qxy for peak in peaks],
            [peak.qz for peak in peaks],
            s=16,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
        )
    axis.set_title(f"Peak detection: {title}")
    axis.set_xlabel(r"$q_{xy}$ ($\AA^{-1}$)")
    axis.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _candidate_search_config(cfg: BenchmarkRunConfig) -> CandidateSearchConfig:
    return CandidateSearchConfig(
        hkl_max=cfg.candidate_hkl_max,
        q_tolerance=cfg.candidate_q_tolerance,
        relative_tolerance=cfg.candidate_relative_tolerance,
        lattice_min=cfg.candidate_lattice_min,
        lattice_max=cfg.candidate_lattice_max,
        grid_points=cfg.candidate_grid_points,
        max_candidates=cfg.candidate_max_candidates,
        phase_tag=DEFAULT_PHASE_TAG,
        enable_projected_axis_search=True,
    )


def _generate_candidate_cifs(
    spec: BenchmarkStructureSpec,
    candidates: list[Any],
    cfg: BenchmarkRunConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    molecules = _molecule_records(spec.organic_molecules)
    for candidate in candidates:
        records.extend(
            generate_ranked_cif_records(
                candidate,
                atoms=spec.inorganic_atoms,
                molecules=molecules,
                stoichiometry="",
                limit=max(1, cfg.cif_records_per_candidate),
                allow_explicit_templates=False,
            )
        )
        if len(records) >= max(1, cfg.max_generated_cifs_to_compare):
            break
    return sorted(
        records, key=lambda item: float(item.get("score", math.inf))
    )[: max(1, cfg.max_generated_cifs_to_compare)]


def _molecule_records(labels: Iterable[str]) -> list[dict[str, Any]]:
    molecules = []
    for label in labels:
        key = str(label).strip()
        if not key or key.lower() == "none":
            continue
        metadata = dict(REFERENCE_MOLECULES.get(key, {}))
        metadata.setdefault("formula", "")
        metadata.setdefault("name", key)
        metadata.setdefault("source", "benchmark-constraint")
        metadata["label"] = key
        molecules.append(metadata)
    return molecules


def _materialize_generated_cifs(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    data_file_id: str,
    fileset_id: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for record in records:
        payload = dict(record)
        cif_id = _slug(
            str(payload.get("cif_id") or f"generated_{len(materialized) + 1}")
        )
        path = output_dir / f"{cif_id}.cif"
        path.write_text(str(payload.get("cif_text") or ""), encoding="utf-8")
        payload.update(
            {
                "cif_id": cif_id,
                "data_id": data_file_id,
                "benchmark_fileset_id": fileset_id,
                "path": str(path),
                "local_path": str(path),
                "structure_path": str(path),
            }
        )
        materialized.append(payload)
    return materialized


def _rank_generated_cifs(
    target: xr.DataArray,
    records: list[dict[str, Any]],
    target_params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
    rankings_dir: Path,
    plots_dir: Path,
    *,
    peak_intensity_peaks: list[StructurePeak] | None = None,
) -> list[dict[str, Any]]:
    rankings = []
    parameter_grid = _comparison_parameter_grid(target_params, cfg)
    for record in records:
        path = Path(str(record.get("path") or record.get("structure_path")))
        if not path.exists():
            continue
        stages: dict[str, Any] = {}
        scoring_path = path
        if cfg.staged_refinement:
            refined = _staged_refinement_path(
                target,
                record,
                path,
                target_params,
                cfg,
                rankings_dir / "staged_refinement" / str(record.get("cif_id")),
            )
            scoring_path = Path(str(refined["path"]))
            stages = dict(refined.get("stages", {}))
        density = _cif_density_g_cm3(scoring_path)
        density_penalty = _density_plausibility_penalty(scoring_path, density)
        chemistry_metrics = _cif_physical_chemistry_metrics(scoring_path)
        physical_penalty = float(
            chemistry_metrics.get("physical_penalty", 0.0) or 0.0
        )
        charge_penalty = float(
            record.get("charge_balance_penalty", 0.0) or 0.0
        )
        lattice_prior_penalty = float(
            record.get("lattice_prior_penalty", 0.0) or 0.0
        )
        best: dict[str, Any] | None = None
        best_comparison = None
        for params in parameter_grid:
            try:
                simulated = simulate_giwaxs_image(scoring_path, params)
                comparison = compare_giwaxs_images(
                    target,
                    simulated,
                    target_label="Mock experiment",
                    simulated_label=str(
                        record.get("cif_id", scoring_path.stem)
                    ),
                )
                bragg_intensity_match = _bragg_peak_intensity_match_for_image(
                    peak_intensity_peaks,
                    simulated,
                    cfg,
                )
                bragg_intensity_penalty = (
                    cfg.bragg_intensity_weight
                    * float(
                        bragg_intensity_match.get(
                            "intensity_match_penalty",
                            0.0,
                        )
                    )
                )
                comparison.metrics.update(
                    _bragg_intensity_metric_summary(
                        bragg_intensity_match,
                        bragg_intensity_penalty,
                    )
                )
            except Exception as exc:
                if best is None:
                    best = {
                        "generated_cif_id": record.get("cif_id"),
                        "path": str(scoring_path),
                        "source_path": str(path),
                        "error": str(exc),
                        "metrics": {},
                        "density_g_cm3": density,
                        "density_penalty": density_penalty,
                        "chemistry_metrics": chemistry_metrics,
                        "physical_penalty": physical_penalty,
                        "charge_penalty": charge_penalty,
                        "lattice_prior_penalty": lattice_prior_penalty,
                        "bragg_intensity_match": {},
                        "bragg_intensity_penalty": 0.0,
                        "parameters": params.as_dict(),
                        "stages": stages,
                    }
                continue
            current = {
                "generated_cif_id": record.get("cif_id"),
                "path": str(scoring_path),
                "source_path": str(path),
                "source_rank": record.get("rank"),
                "source_score": record.get("score"),
                "metrics": dict(comparison.metrics),
                "density_g_cm3": density,
                "density_penalty": density_penalty,
                "chemistry_metrics": chemistry_metrics,
                "physical_penalty": physical_penalty,
                "charge_penalty": charge_penalty,
                "lattice_prior_penalty": lattice_prior_penalty,
                "bragg_intensity_match": bragg_intensity_match,
                "bragg_intensity_penalty": bragg_intensity_penalty,
                "parameters": params.as_dict(),
                "stages": stages,
            }
            if best is None or _comparison_sort_key(
                current
            ) < _comparison_sort_key(best):
                best = current
                best_comparison = comparison
        if best is not None:
            best["_comparison"] = best_comparison
            rankings.append(best)
    rankings.sort(key=_comparison_sort_key)
    serializable = []
    for rank, item in enumerate(rankings, start=1):
        comparison = item.pop("_comparison", None)
        item["fit_rank"] = rank
        serializable.append(dict(item))
        if comparison is not None and rank <= cfg.comparison_plot_count:
            plot_path = plots_dir / f"comparison_rank_{rank:02d}.png"
            save_giwaxs_comparison_plot(
                comparison,
                plot_path,
                title=f"Generated CIF rank {rank}",
            )
            item["comparison_plot"] = str(plot_path)
            serializable[-1]["comparison_plot"] = str(plot_path)
    _write_json(rankings_dir / "generated_cif_rankings.json", serializable)
    return serializable


def _attach_refined_cif_records(
    project: ProjectState,
    comparisons: list[dict[str, Any]],
) -> None:
    generated = project.reference_cifs.setdefault("generated", {})
    if not isinstance(generated, dict):
        return
    for item in comparisons:
        cif_id = str(item.get("generated_cif_id") or "")
        if not cif_id or cif_id not in generated:
            continue
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            continue
        record = generated[cif_id]
        if not isinstance(record, dict):
            continue
        refined_text = path.read_text(encoding="utf-8")
        original_path = record.get("path")
        record["source_candidate_path"] = original_path
        record["path"] = str(path)
        record["local_path"] = str(path)
        record["structure_path"] = str(path)
        record["cif_text"] = refined_text
        record["refinement_stages"] = item.get("stages", {})
        record["fit_metrics"] = item.get("metrics", {})
        record["fit_rank"] = item.get("fit_rank")
        project.structures[cif_id] = dict(record)


def _staged_refinement_path(
    target: xr.DataArray,
    record: dict[str, Any],
    full_path: Path,
    target_params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inorganic_elements = [
        str(element) for element in record.get("atoms", []) if element
    ]
    molecule_labels = [
        str(item.get("label") or item.get("name") or "")
        for item in record.get("molecules", [])
        if str(item.get("label") or item.get("name") or "")
    ]
    coarse_params = _coarse_refinement_parameters(target_params, cfg)
    scaffold_path = output_dir / f"{full_path.stem}_inorganic_scaffold.cif"
    _write_filtered_cif(
        full_path,
        scaffold_path,
        keep_elements=set(inorganic_elements),
        organic_note=", ".join(molecule_labels) or None,
    )
    scaffold_eval = _evaluate_cif_path(
        target,
        scaffold_path,
        [coarse_params],
        label=f"{record.get('cif_id')}: scaffold",
    )

    inorganic_shifted_path = scaffold_path
    inorganic_shift_eval = scaffold_eval
    secondary_elements = set(inorganic_elements[1:])
    if secondary_elements:
        inorganic_shifted_path, inorganic_shift_eval = _best_shifted_cif(
            target,
            scaffold_path,
            secondary_elements,
            coarse_params,
            cfg,
            output_dir,
            stage_name="inorganic_secondary_shift",
        )

    full_inorganic_refined = (
        output_dir / f"{full_path.stem}_inorganic_refined.cif"
    )
    if secondary_elements and inorganic_shifted_path != scaffold_path:
        _write_shifted_cif(
            full_path,
            full_inorganic_refined,
            element_selector=secondary_elements,
            fractional_shift=tuple(
                inorganic_shift_eval.get("fractional_shift", (0.0, 0.0, 0.0))
            ),
        )
    else:
        full_inorganic_refined.write_text(
            full_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    physicalized_path = output_dir / f"{full_path.stem}_chemical_geometry.cif"
    _write_physicalized_cif(full_inorganic_refined, physicalized_path)
    physical_eval = _evaluate_cif_path(
        target,
        physicalized_path,
        [coarse_params],
        label=f"{record.get('cif_id')}: chemical geometry",
    )
    unphysical_eval = _evaluate_cif_path(
        target,
        full_inorganic_refined,
        [coarse_params],
        label=f"{record.get('cif_id')}: pre-geometry",
    )
    base_sources: list[tuple[Path, dict[str, Any], str]] = [
        (physicalized_path, physical_eval, "chemical_geometry"),
        (full_inorganic_refined, unphysical_eval, "pre_geometry"),
    ]
    perovskite_sources = (
        _perovskite_scaffold_refinement_sources(
            target,
            full_inorganic_refined,
            coarse_params,
            cfg,
            output_dir,
            label=f"{record.get('cif_id')}: perovskite scaffold",
        )
        if cfg.refinement_scaffold_variant_count > 0
        else []
    )
    base_sources.extend(perovskite_sources)
    base_sources = sorted(
        base_sources,
        key=lambda item: _comparison_sort_key(item[1]),
    )[:3]
    molecular_candidates: list[tuple[Path, dict[str, Any], str]] = []
    if molecule_labels:
        prefixes = {_label_token(label) for label in molecule_labels}
        for source_path, _, source_label in base_sources:
            shifted_path, shifted_eval = _best_shifted_cif(
                target,
                source_path,
                prefixes,
                coarse_params,
                cfg,
                output_dir,
                stage_name=f"{source_label}_molecule_shift",
                selector_mode="label_prefix",
            )
            molecular_candidates.append(
                (shifted_path, shifted_eval, source_label)
            )
    else:
        molecular_candidates.extend(base_sources)

    molecule_refined_path, molecule_eval, molecule_stage_name = min(
        molecular_candidates,
        key=lambda item: _comparison_sort_key(item[1]),
    )

    return {
        "path": str(molecule_refined_path),
        "stages": {
            "inorganic_scaffold": scaffold_eval,
            "inorganic_secondary_refinement": inorganic_shift_eval,
            "chemical_geometry_refinement": physical_eval,
            "pre_geometry_refinement": unphysical_eval,
            "perovskite_scaffold_refinement": [
                evaluation for _, evaluation, _ in perovskite_sources
            ],
            "selected_molecular_refinement_source": molecule_stage_name,
            "molecular_body_refinement": molecule_eval,
        },
    }


def _coarse_refinement_parameters(
    target_params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
) -> GIWAXSSimulationParameters:
    return GIWAXSSimulationParameters(
        sigma_theta=target_params.sigma_theta,
        sigma_phi=target_params.sigma_phi,
        sigma_r=target_params.sigma_r,
        hkl_extent=min(
            cfg.hkl_extent, max(1, cfg.refinement_coarse_hkl_extent)
        ),
        theta_x_deg=target_params.theta_x_deg,
        theta_y_deg=target_params.theta_y_deg,
        qxy_min=cfg.qxy_range[0],
        qxy_max=cfg.qxy_range[1],
        qz_min=cfg.qz_range[0],
        qz_max=cfg.qz_range[1],
        resolution_z=cfg.refinement_coarse_detector_shape[0],
        resolution_x=cfg.refinement_coarse_detector_shape[1],
    )


def _best_shifted_cif(
    target: xr.DataArray,
    source_path: Path,
    selector: set[str],
    params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
    output_dir: Path,
    *,
    stage_name: str,
    selector_mode: str = "element",
) -> tuple[Path, dict[str, Any]]:
    best_path = source_path
    best_eval: dict[str, Any] | None = None
    shifts = _fractional_shift_grid(cfg.refinement_fractional_step)
    for index, shift in enumerate(shifts):
        if shift == (0.0, 0.0, 0.0):
            candidate_path = source_path
        else:
            candidate_path = (
                output_dir / f"{source_path.stem}_{stage_name}_{index:02d}.cif"
            )
            if selector_mode == "label_prefix":
                _write_shifted_cif(
                    source_path,
                    candidate_path,
                    label_prefix_selector=selector,
                    fractional_shift=shift,
                )
            else:
                _write_shifted_cif(
                    source_path,
                    candidate_path,
                    element_selector=selector,
                    fractional_shift=shift,
                )
        evaluation = _evaluate_cif_path(
            target,
            candidate_path,
            [params],
            label=f"{source_path.stem}: {stage_name}",
        )
        evaluation["fractional_shift"] = list(shift)
        if best_eval is None or _comparison_sort_key(
            evaluation
        ) < _comparison_sort_key(best_eval):
            best_eval = evaluation
            best_path = candidate_path
    return best_path, best_eval or {}


def _prefer_chemical_geometry(
    physical_eval: dict[str, Any],
    unphysical_eval: dict[str, Any],
) -> bool:
    if _comparison_sort_key(physical_eval) <= _comparison_sort_key(
        unphysical_eval
    ):
        return True
    physical_metrics = physical_eval.get("metrics", {})
    unphysical_metrics = unphysical_eval.get("metrics", {})
    physical_focus = float(physical_metrics.get("peak_focus_score", math.inf))
    unphysical_focus = float(
        unphysical_metrics.get("peak_focus_score", math.inf)
    )
    physical_penalty = float(physical_eval.get("physical_penalty", 0.0) or 0.0)
    unphysical_penalty = float(
        unphysical_eval.get("physical_penalty", 0.0) or 0.0
    )
    return (
        unphysical_penalty - physical_penalty >= 0.04
        and physical_focus - unphysical_focus <= 0.30
    )


def _fractional_shift_grid(step: float) -> list[tuple[float, float, float]]:
    value = float(step)
    shifts = [
        (0.0, 0.0, 0.0),
        (value, 0.0, 0.0),
        (-value, 0.0, 0.0),
        (0.0, value, 0.0),
        (0.0, -value, 0.0),
        (0.0, 0.0, value),
        (0.0, 0.0, -value),
    ]
    return list(dict.fromkeys(shifts))


def _evaluate_cif_path(
    target: xr.DataArray,
    path: Path,
    parameter_grid: list[GIWAXSSimulationParameters],
    *,
    label: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for params in parameter_grid:
        try:
            simulated = simulate_giwaxs_image(path, params)
            comparison = compare_giwaxs_images(
                target,
                simulated,
                target_label="Mock experiment",
                simulated_label=label,
            )
            current = {
                "path": str(path),
                "metrics": dict(comparison.metrics),
                **_path_chemistry_rank_fields(path),
                "parameters": params.as_dict(),
            }
        except Exception as exc:
            current = {
                "path": str(path),
                "metrics": {},
                **_path_chemistry_rank_fields(path),
                "parameters": params.as_dict(),
                "error": str(exc),
            }
        if best is None or _comparison_sort_key(
            current
        ) < _comparison_sort_key(best):
            best = current
    return best or {"path": str(path), "metrics": {}}


def _write_filtered_cif(
    source_path: Path,
    output_path: Path,
    *,
    keep_elements: set[str],
    organic_note: str | None = None,
) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = []
    composition: dict[str, float] = {}
    in_atom_rows = False
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            filtered.append(line)
            continue
        if (
            not in_atom_rows
            or not stripped
            or stripped.startswith(("_", "#", "loop_"))
        ):
            filtered.append(line)
            continue
        parts = stripped.split()
        if len(parts) >= 6 and parts[1] in keep_elements:
            filtered.append(line)
            composition[parts[1]] = composition.get(parts[1], 0.0) + 1.0
    filtered = [
        (
            f"_chemical_formula_sum '{_formula_sum(composition)}'"
            if line.strip().startswith("_chemical_formula_sum")
            else line
        )
        for line in filtered
    ]
    filtered = _annotate_filtered_inorganic_scaffold(
        filtered,
        organic_note=organic_note,
    )
    output_path.write_text(
        "\n".join(filtered).rstrip() + "\n", encoding="utf-8"
    )


def _annotate_filtered_inorganic_scaffold(
    lines: list[str],
    *,
    organic_note: str | None,
) -> list[str]:
    note = _filtered_scaffold_organic_note(lines, organic_note)
    annotated: list[str] = []
    inserted = False
    for line in lines:
        annotated.append(line)
        if line.startswith("data_") and not inserted:
            annotated.append("# structure variant: inorganic scaffold only")
            annotated.append(f"# organic molecules to add: {note}")
            inserted = True
    for index, line in enumerate(annotated):
        if line.startswith("# molecular species:"):
            annotated[index] = (
                "# molecular species: inorganic scaffold only; planned "
                f"organics: {note}"
            )
            break
    return annotated


def _filtered_scaffold_organic_note(
    lines: list[str],
    organic_note: str | None,
) -> str:
    note = str(organic_note or "").strip()
    if note:
        return note
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("# molecular species:"):
            continue
        parsed = stripped.split(":", 1)[1].strip()
        if parsed and parsed.lower() not in {"none", "unspecified"}:
            return parsed
    return "none recorded"


def _write_physicalized_cif(source_path: Path, output_path: Path) -> None:
    rows, lattice = _read_simple_cif_sites(source_path)
    if not rows or lattice is None:
        output_path.write_text(
            source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return
    adjusted = {
        int(row["row_index"]): np.asarray(row["frac"], dtype=float)
        for row in rows
    }
    _repair_cation_sublattice(rows, lattice, adjusted)
    _repair_halide_coordination(rows, lattice, adjusted)
    _move_molecules_to_voids(rows, lattice, adjusted)
    _relax_halides_for_organic_contacts(rows, lattice, adjusted)
    lines = source_path.read_text(encoding="utf-8").splitlines()
    for row in rows:
        row_index = int(row["row_index"])
        if row_index < 0 or row_index >= len(lines):
            continue
        parts = lines[row_index].split()
        if len(parts) < 6:
            continue
        frac = adjusted[row_index] % 1.0
        parts[2:5] = [f"{value:.6f}" for value in frac]
        lines[row_index] = " ".join(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def _perovskite_scaffold_refinement_sources(
    target: xr.DataArray,
    source_path: Path,
    params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
    output_dir: Path,
    *,
    label: str,
) -> list[tuple[Path, dict[str, Any], str]]:
    rows, lattice = _read_simple_cif_sites(source_path)
    variants = _perovskite_cation_coordinate_variants(rows, lattice)
    if not variants:
        return []
    sources: list[tuple[Path, dict[str, Any], str]] = []
    variant_count = max(0, int(cfg.refinement_scaffold_variant_count))
    for index, cation_positions in enumerate(
        variants[:variant_count], start=1
    ):
        raw_path = (
            output_dir
            / f"{source_path.stem}_perovskite_cation_motif_{index:02d}.cif"
        )
        physicalized_path = (
            output_dir
            / f"{source_path.stem}_perovskite_scaffold_{index:02d}.cif"
        )
        _write_cation_scaffold_cif(source_path, raw_path, cation_positions)
        _write_physicalized_cif(raw_path, physicalized_path)
        evaluation = _evaluate_cif_path(
            target,
            physicalized_path,
            [params],
            label=f"{label} {index}",
        )
        sources.append(
            (
                physicalized_path,
                evaluation,
                f"perovskite_scaffold_{index:02d}",
            )
        )
    return sources


def _perovskite_cation_coordinate_variants(
    rows: list[dict[str, Any]],
    lattice: np.ndarray | None,
) -> list[list[np.ndarray]]:
    if lattice is None:
        return []
    cations = [
        row for row in rows if str(row["element"]) in {"Pb", "Sn", "Ge"}
    ]
    halides = [
        row for row in rows if str(row["element"]) in {"I", "Br", "Cl", "F"}
    ]
    if not cations or not halides:
        return []
    count = len(cations)
    offsets = [
        (0.68, 0.61),
        (0.72, 0.58),
        (0.64, 0.66),
        (0.76, 0.54),
    ]
    variants: list[list[np.ndarray]] = []
    if count == 6:
        for x_frac, y_frac in offsets:
            variants.append(
                [
                    np.asarray((0.0, 0.0, 0.0), dtype=float),
                    np.asarray((0.5, 0.5, 0.5), dtype=float),
                    np.asarray((x_frac, y_frac, 0.0), dtype=float),
                    np.asarray((-x_frac, -y_frac, 0.0), dtype=float),
                    np.asarray(
                        (-x_frac + 0.5, y_frac + 0.5, 0.5), dtype=float
                    ),
                    np.asarray(
                        (x_frac + 0.5, -y_frac + 0.5, 0.5), dtype=float
                    ),
                ]
            )
    elif count == 4:
        for x_frac, y_frac in offsets:
            variants.append(
                [
                    np.asarray((x_frac, y_frac, 0.0), dtype=float),
                    np.asarray((-x_frac, -y_frac, 0.0), dtype=float),
                    np.asarray(
                        (-x_frac + 0.5, y_frac + 0.5, 0.5), dtype=float
                    ),
                    np.asarray(
                        (x_frac + 0.5, -y_frac + 0.5, 0.5), dtype=float
                    ),
                ]
            )
    elif count == 3:
        for x_frac, y_frac in offsets:
            variants.append(
                [
                    np.asarray((0.0, 0.0, 0.0), dtype=float),
                    np.asarray((0.5, 0.5, 0.5), dtype=float),
                    np.asarray((x_frac, y_frac, 0.0), dtype=float),
                ]
            )
    return [[coord % 1.0 for coord in variant] for variant in variants]


def _write_cation_scaffold_cif(
    source_path: Path,
    output_path: Path,
    cation_positions: list[np.ndarray],
) -> None:
    rows, _ = _read_simple_cif_sites(source_path)
    cation_rows = [
        row for row in rows if str(row["element"]) in {"Pb", "Sn", "Ge"}
    ]
    if len(cation_rows) != len(cation_positions):
        output_path.write_text(
            source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return
    adjusted = {
        int(row["row_index"]): np.asarray(position, dtype=float) % 1.0
        for row, position in zip(cation_rows, cation_positions, strict=True)
    }
    _write_adjusted_fractional_cif(source_path, output_path, adjusted)


def _write_adjusted_fractional_cif(
    source_path: Path,
    output_path: Path,
    adjusted: dict[int, np.ndarray],
) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    for row_index, frac in adjusted.items():
        if row_index < 0 or row_index >= len(lines):
            continue
        parts = lines[row_index].split()
        if len(parts) < 6:
            continue
        wrapped = np.asarray(frac, dtype=float) % 1.0
        parts[2:5] = [f"{value:.6f}" for value in wrapped]
        lines[row_index] = " ".join(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def _repair_halide_coordination(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
    adjusted: dict[int, np.ndarray],
) -> None:
    cations = [
        row for row in rows if str(row["element"]) in {"Pb", "Sn", "Ge"}
    ]
    halides = [
        row for row in rows if str(row["element"]) in {"I", "Br", "Cl", "F"}
    ]
    if not cations or not halides:
        return
    candidates = _halide_site_candidates(cations, lattice, adjusted)
    if not candidates:
        return
    cation_centers = [
        np.asarray(
            adjusted.get(int(cation["row_index"]), cation["frac"]),
            dtype=float,
        )
        for cation in cations
    ]
    cation_windows = [
        _coordination_distance_window(str(cation["element"]))
        for cation in cations
    ]
    cation_neighbor_counts = [0 for _ in cations]
    assigned_halides: list[np.ndarray] = []
    used: set[int] = set()
    for halide in sorted(halides, key=lambda item: str(item["label"])):
        current = np.asarray(halide["frac"], dtype=float)
        best_index = None
        best_score = math.inf
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            score = 0.15 * _pbc_distance(current, candidate, lattice)
            coordinated_cations = []
            nearest_cation = math.inf
            for cation_index, center in enumerate(cation_centers):
                distance = _pbc_distance(candidate, center, lattice)
                nearest_cation = min(nearest_cation, distance)
                lower, upper = cation_windows[cation_index]
                if lower <= distance <= upper:
                    coordinated_cations.append(cation_index)
            if nearest_cation < 2.50:
                score += 100.0 + 25.0 * (2.50 - nearest_cation)
            if not coordinated_cations:
                score += 20.0
            else:
                deficit = sum(
                    max(0, 4 - cation_neighbor_counts[cation_index])
                    for cation_index in coordinated_cations
                )
                oversupplied = sum(
                    max(0, cation_neighbor_counts[cation_index] - 5)
                    for cation_index in coordinated_cations
                )
                score -= 2.0 * deficit
                score += 1.5 * oversupplied
            nearest_halide = min(
                (
                    _pbc_distance(candidate, other, lattice)
                    for other in assigned_halides
                ),
                default=math.inf,
            )
            if nearest_halide < 2.55:
                score += 50.0 + 15.0 * (2.55 - nearest_halide)
            if score < best_score:
                best_index = index
                best_score = score
        if best_index is None:
            continue
        used.add(best_index)
        assigned = candidates[best_index] % 1.0
        adjusted[int(halide["row_index"])] = assigned
        assigned_halides.append(assigned)
        for cation_index, center in enumerate(cation_centers):
            lower, upper = cation_windows[cation_index]
            if lower <= _pbc_distance(assigned, center, lattice) <= upper:
                cation_neighbor_counts[cation_index] += 1
    _improve_halide_coordination_assignment(
        halides,
        candidates,
        cation_centers,
        cation_windows,
        adjusted,
        lattice,
    )


def _improve_halide_coordination_assignment(
    halides: list[dict[str, Any]],
    candidates: list[np.ndarray],
    cation_centers: list[np.ndarray],
    cation_windows: list[tuple[float, float]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> None:
    if not halides or not candidates:
        return
    row_indices = [int(row["row_index"]) for row in halides]
    assignments = [
        np.asarray(adjusted.get(index, halide["frac"]), dtype=float) % 1.0
        for index, halide in zip(row_indices, halides, strict=True)
    ]
    current_candidates = list(candidates)
    for assignment in assignments:
        current_candidates.append(assignment)
    current_candidates = _dedupe_fractional_points(current_candidates)
    candidate_keys = [
        tuple(int(round(value * 1000.0)) for value in candidate % 1.0)
        for candidate in current_candidates
    ]
    used = {
        tuple(int(round(value * 1000.0)) for value in assignment % 1.0)
        for assignment in assignments
    }
    best_score = _halide_assignment_score(
        assignments,
        [np.asarray(row["frac"], dtype=float) for row in halides],
        cation_centers,
        cation_windows,
        lattice,
    )
    for _ in range(3):
        improved = False
        for halide_index, current in enumerate(list(assignments)):
            current_key = tuple(int(round(value * 1000.0)) for value in current)
            used.discard(current_key)
            local_best = current
            local_score = best_score
            for candidate, key in zip(
                current_candidates,
                candidate_keys,
                strict=True,
            ):
                if key in used:
                    continue
                trial = list(assignments)
                trial[halide_index] = candidate % 1.0
                score = _halide_assignment_score(
                    trial,
                    [np.asarray(row["frac"], dtype=float) for row in halides],
                    cation_centers,
                    cation_windows,
                    lattice,
                )
                if score + 1.0e-9 < local_score:
                    local_score = score
                    local_best = candidate % 1.0
            assignments[halide_index] = local_best
            used.add(
                tuple(int(round(value * 1000.0)) for value in local_best)
            )
            if local_score + 1.0e-9 < best_score:
                best_score = local_score
                improved = True
        if not improved:
            break
    for index, assignment in zip(row_indices, assignments, strict=True):
        adjusted[index] = assignment % 1.0


def _halide_assignment_score(
    assignments: list[np.ndarray],
    original_positions: list[np.ndarray],
    cation_centers: list[np.ndarray],
    cation_windows: list[tuple[float, float]],
    lattice: np.ndarray,
) -> float:
    score = 0.0
    target_min = 4
    for cation_index, center in enumerate(cation_centers):
        lower, upper = cation_windows[cation_index]
        distances = sorted(
            _pbc_distance(assignment, center, lattice)
            for assignment in assignments
        )
        near = [distance for distance in distances if lower <= distance <= upper]
        deficit = max(0, target_min - len(near))
        score += 5.0 * deficit * deficit
        if distances:
            nearest = distances[0]
            if nearest < lower - 0.20:
                score += 15.0 * (lower - 0.20 - nearest)
            mean_four = float(np.mean(distances[: min(4, len(distances))]))
            if mean_four > upper + 0.35:
                score += 3.0 * (mean_four - upper - 0.35)
    for left_index, left in enumerate(assignments):
        coordinated = 0
        nearest = math.inf
        for center, (lower, upper) in zip(
            cation_centers,
            cation_windows,
            strict=True,
        ):
            distance = _pbc_distance(left, center, lattice)
            nearest = min(nearest, distance)
            if lower <= distance <= upper:
                coordinated += 1
        if coordinated == 0:
            score += 12.0 if nearest < math.inf else 20.0
        for right in assignments[left_index + 1 :]:
            distance = _pbc_distance(left, right, lattice)
            if distance < 2.55:
                score += 30.0 * (2.55 - distance)
    for original, assignment in zip(
        original_positions,
        assignments,
        strict=True,
    ):
        score += 0.08 * _pbc_distance(original, assignment, lattice)
    return score


def _repair_cation_sublattice(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
    adjusted: dict[int, np.ndarray],
) -> None:
    cations = [
        row for row in rows if str(row["element"]) in {"Pb", "Sn", "Ge"}
    ]
    if not cations:
        return
    motif_sites = _cation_sublattice_site_candidates(len(cations))
    if len(motif_sites) < len(cations):
        return
    selected_sites = _assign_fractional_sites(cations, motif_sites, lattice)
    for row, site in zip(cations, selected_sites):
        adjusted[int(row["row_index"])] = np.asarray(site, dtype=float) % 1.0


def _cation_sublattice_site_candidates(count: int) -> list[np.ndarray]:
    if count <= 0:
        return []
    layered_offset = 0.22
    tilt_offset = 0.10
    motif = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5 + layered_offset, 0.5 + tilt_offset, 0.0),
        (0.5 - layered_offset, 0.5 - tilt_offset, 0.0),
        (0.5 + layered_offset, tilt_offset, 0.5),
        (0.5 - layered_offset, 1.0 - tilt_offset, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.25, 0.25, 0.0),
        (0.75, 0.75, 0.5),
        (0.25, 0.75, 0.5),
        (0.75, 0.25, 0.0),
    ]
    if count == 1:
        motif = [(0.5, 0.5, 0.5), *motif]
    elif count == 2:
        motif = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), *motif]
    return _dedupe_fractional_points(
        [np.asarray(point, dtype=float) for point in motif]
    )[: max(count, 1)]


def _assign_fractional_sites(
    rows: list[dict[str, Any]],
    candidate_sites: list[np.ndarray],
    lattice: np.ndarray,
) -> list[np.ndarray]:
    count = len(rows)
    sites = [np.asarray(site, dtype=float) % 1.0 for site in candidate_sites]
    if count == 0:
        return []
    if len(sites) < count:
        return [np.asarray(row["frac"], dtype=float) for row in rows]
    if count <= 8:
        from itertools import permutations

        best_assignment = sites[:count]
        best_score = math.inf
        for indices in permutations(range(len(sites)), count):
            score = 0.0
            for row, site_index in zip(rows, indices):
                score += _pbc_distance(row["frac"], sites[site_index], lattice)
            if score < best_score:
                best_score = score
                best_assignment = [sites[index] for index in indices]
        return best_assignment
    used: set[int] = set()
    assignment = []
    for row in rows:
        best_index = None
        best_distance = math.inf
        for index, site in enumerate(sites):
            if index in used:
                continue
            distance = _pbc_distance(row["frac"], site, lattice)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            assignment.append(np.asarray(row["frac"], dtype=float))
            continue
        used.add(best_index)
        assignment.append(sites[best_index])
    return assignment


def _halide_site_candidates(
    cations: list[dict[str, Any]],
    lattice: np.ndarray,
    adjusted: dict[int, np.ndarray] | None = None,
) -> list[np.ndarray]:
    adjusted = adjusted or {}
    inv_lattice = np.linalg.inv(lattice)
    cart_axes = [lattice[index] for index in range(3)]
    directions = []
    for axis in cart_axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            continue
        unit = axis / norm
        directions.extend([unit, -unit])
    candidates: list[np.ndarray] = []
    for cation in cations:
        distance = 3.05 if str(cation["element"]) == "Pb" else 2.85
        lower, upper = _coordination_distance_window(str(cation["element"]))
        center = np.asarray(
            adjusted.get(int(cation["row_index"]), cation["frac"]),
            dtype=float,
        )
        for direction in directions:
            frac_offset = (direction * distance) @ inv_lattice
            candidate = (center + frac_offset) % 1.0
            actual_distance = _pbc_distance(center, candidate, lattice)
            if lower <= actual_distance <= upper:
                candidates.append(candidate)
    for left_index, left in enumerate(cations):
        for right in cations[left_index + 1 :]:
            delta = np.asarray(
                adjusted.get(int(right["row_index"]), right["frac"]),
                dtype=float,
            ) - np.asarray(
                adjusted.get(int(left["row_index"]), left["frac"]),
                dtype=float,
            )
            delta -= np.round(delta)
            distance = float(np.linalg.norm(delta @ lattice))
            if 3.3 <= distance <= 7.2:
                left_center = np.asarray(
                    adjusted.get(int(left["row_index"]), left["frac"]),
                    dtype=float,
                )
                midpoint = (left_center + delta / 2.0) % 1.0
                left_lower, left_upper = _coordination_distance_window(
                    str(left["element"])
                )
                right_lower, right_upper = _coordination_distance_window(
                    str(right["element"])
                )
                left_distance = _pbc_distance(left_center, midpoint, lattice)
                right_center = np.asarray(
                    adjusted.get(int(right["row_index"]), right["frac"]),
                    dtype=float,
                )
                right_distance = _pbc_distance(right_center, midpoint, lattice)
                if (
                    left_lower <= left_distance <= left_upper
                    and right_lower <= right_distance <= right_upper
                ):
                    candidates.append(midpoint)
    return _dedupe_fractional_points(candidates)


def _move_molecules_to_voids(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
    adjusted: dict[int, np.ndarray],
) -> None:
    bodies: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        token = _molecular_body_token(str(row["label"]))
        if token:
            bodies.setdefault(token, []).append(row)
    if not bodies:
        return
    inorganic_rows = [
        row for row in rows if not _molecular_body_token(str(row["label"]))
    ]
    occupied_centers: list[np.ndarray] = []
    occupied_bodies: list[tuple[np.ndarray, list[str], str]] = []
    for body_token, body_rows in sorted(
        bodies.items(),
        key=lambda item: _molecule_placement_sort_key(item[0]),
    ):
        row_indices = [int(row["row_index"]) for row in body_rows]
        coords = np.asarray([adjusted[index] for index in row_indices])
        elements = [str(row["element"]) for row in body_rows]
        species = _molecular_species_token(body_token)
        center = _fractional_center(coords)
        best_coords = coords
        best_center = center
        best_score = -math.inf
        candidate_centers = _ranked_molecule_candidate_centers(
            coords,
            elements,
            _molecule_candidate_centers(
                center,
                inorganic_rows,
                occupied_centers,
                adjusted,
            ),
            inorganic_rows,
            occupied_centers,
            occupied_bodies,
            adjusted,
            lattice,
        )
        for candidate_center in candidate_centers:
            for candidate_coords in _rigid_body_pose_candidates(
                coords,
                elements,
                species,
                candidate_center,
                inorganic_rows,
                occupied_bodies,
                adjusted,
                lattice,
            ):
                score = _molecule_pose_score(
                    candidate_coords,
                    elements,
                    species,
                    candidate_center,
                    inorganic_rows,
                    occupied_centers,
                    occupied_bodies,
                    adjusted,
                    lattice,
                )
                if score > best_score:
                    best_score = score
                    best_center = candidate_center
                    best_coords = candidate_coords
        for index, coord in zip(row_indices, best_coords, strict=True):
            adjusted[index] = np.asarray(coord, dtype=float) % 1.0
        occupied_centers.append(best_center % 1.0)
        occupied_bodies.append((best_coords % 1.0, elements, species))


def _molecule_placement_sort_key(body_token: str) -> tuple[int, str]:
    species = _molecular_species_token(body_token)
    if species in DONOR_MOLECULES:
        return (0, body_token)
    if species in ACCEPTOR_MOLECULES:
        return (1, body_token)
    return (2, body_token)


def _molecule_candidate_centers(
    current_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_centers: list[np.ndarray],
    adjusted: dict[int, np.ndarray],
) -> list[np.ndarray]:
    grid = [
        (x, y, z)
        for x in (0.125, 0.375, 0.625, 0.875)
        for y in (0.125, 0.375, 0.625, 0.875)
        for z in (0.125, 0.375, 0.625, 0.875)
    ]
    high_symmetry = [
        (x, y, z)
        for x in (0.0, 0.25, 0.5, 0.75)
        for y in (0.0, 0.25, 0.5, 0.75)
        for z in (0.0, 0.25, 0.5, 0.75)
        if (x, y, z).count(0.0) + (x, y, z).count(0.5) >= 2
    ]
    centers = [np.asarray(current_center, dtype=float), *map(np.asarray, grid)]
    centers.extend(np.asarray(point, dtype=float) for point in high_symmetry)
    for center in occupied_centers:
        center = np.asarray(center, dtype=float)
        centers.extend(
            [
                (-center) % 1.0,
                (center + 0.5) % 1.0,
                np.asarray((1.0 - center[0], center[1], center[2])),
                np.asarray((center[0], 1.0 - center[1], center[2])),
                np.asarray((center[0], center[1], 1.0 - center[2])),
            ]
        )
    cations = [
        np.asarray(adjusted[int(row["row_index"])], dtype=float)
        for row in inorganic_rows
        if str(row["element"]) in METAL_CATION_ELEMENTS
        and int(row["row_index"]) in adjusted
    ]
    for left_index, left in enumerate(cations):
        for right in cations[left_index + 1 :]:
            delta = np.asarray(right, dtype=float) - np.asarray(
                left, dtype=float
            )
            delta -= np.round(delta)
            centers.append((left + 0.5 * delta) % 1.0)
    return _dedupe_fractional_points(
        [np.asarray(center) % 1.0 for center in centers]
    )


def _ranked_molecule_candidate_centers(
    coords: np.ndarray,
    elements: list[str],
    centers: list[np.ndarray],
    inorganic_rows: list[dict[str, Any]],
    occupied_centers: list[np.ndarray],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
    *,
    limit: int = 36,
) -> list[np.ndarray]:
    current_center = _fractional_center(coords)
    ranked = []
    for center in centers:
        shifted = (coords + (np.asarray(center) - current_center)) % 1.0
        score = _molecule_center_steric_score(
            shifted,
            elements,
            np.asarray(center, dtype=float),
            inorganic_rows,
            occupied_centers,
            occupied_bodies,
            adjusted,
            lattice,
        )
        ranked.append((score, np.asarray(center, dtype=float) % 1.0))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [center for _, center in ranked[: max(1, int(limit))]]


def _molecule_center_steric_score(
    coords: np.ndarray,
    elements: list[str],
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_centers: list[np.ndarray],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    min_distance = math.inf
    clash_penalty = 0.0
    for coord, element in zip(coords, elements, strict=True):
        for inorganic in inorganic_rows:
            distance = _pbc_distance(
                coord,
                adjusted[int(inorganic["row_index"])],
                lattice,
            )
            minimum = _minimum_pair_distance(
                str(element),
                str(inorganic["element"]),
                True,
                False,
            )
            min_distance = min(min_distance, distance)
            if distance < minimum:
                clash_penalty += (minimum - distance) / max(minimum, 1.0e-9)
    min_body = math.inf
    for center in occupied_centers:
        min_body = min(min_body, _pbc_distance(candidate_center, center, lattice))
    for occupied_coords, occupied_elements, _ in occupied_bodies:
        for left_coord, left_element in zip(coords, elements, strict=True):
            for right_coord, right_element in zip(
                occupied_coords,
                occupied_elements,
                strict=True,
            ):
                distance = _pbc_distance(left_coord, right_coord, lattice)
                minimum = _minimum_pair_distance(
                    str(left_element),
                    str(right_element),
                    True,
                    True,
                )
                if distance < minimum:
                    clash_penalty += 0.75 * (minimum - distance) / max(
                        minimum,
                        1.0e-9,
                    )
    if not np.isfinite(min_distance):
        min_distance = 3.0
    if not np.isfinite(min_body):
        min_body = 4.0
    return (
        min(min_distance, 3.0)
        + 0.15 * min(min_body, 4.0)
        - 3.5 * clash_penalty
        - 5.0 * _body_unit_cell_boundary_penalty(coords)
    )


def _rigid_body_pose_candidates(
    coords: np.ndarray,
    elements: list[str],
    species: str,
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> list[np.ndarray]:
    source_center = _fractional_center(coords)
    cart_offsets = _reference_body_cart_offsets(
        species,
        elements,
    )
    if cart_offsets is None:
        cart_offsets = _body_cart_offsets(coords, source_center, lattice)
    matrices = _molecule_rotation_matrices(
        species,
        cart_offsets,
        elements,
        candidate_center,
        inorganic_rows,
        occupied_bodies,
        adjusted,
        lattice,
    )
    inv_lattice = np.linalg.inv(lattice)
    candidates = []
    for matrix in matrices:
        rotated_offsets = cart_offsets @ matrix.T
        frac_offsets = rotated_offsets @ inv_lattice
        candidates.append((np.asarray(candidate_center) + frac_offsets) % 1.0)
    return _dedupe_body_poses(candidates)


def _reference_body_cart_offsets(
    species: str,
    elements: list[str],
) -> np.ndarray | None:
    template = REFERENCE_MOLECULES.get(species, {}).get("atoms", ())
    if len(template) != len(elements):
        return None
    template_elements = [str(atom[0]) for atom in template]
    if template_elements != [str(element) for element in elements]:
        return None
    template_coords = np.asarray(
        [
            [float(atom[1]), float(atom[2]), float(atom[3])]
            for atom in template
        ],
        dtype=float,
    )
    return template_coords - np.mean(template_coords, axis=0)


def _molecule_rotation_matrices(
    species: str,
    cart_offsets: np.ndarray,
    elements: list[str],
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> list[np.ndarray]:
    matrices = [np.eye(3)]
    for axis in np.eye(3):
        for angle in (math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
            matrices.append(_axis_angle_rotation_matrix(axis, angle))
    role_vector = _molecule_role_vector(species, cart_offsets, elements)
    target_vector = _molecule_target_vector(
        species,
        candidate_center,
        inorganic_rows,
        occupied_bodies,
        adjusted,
        lattice,
    )
    if role_vector is not None and target_vector is not None:
        base = _rotation_matrix_between(role_vector, target_vector)
        target_unit = _unit_vector(target_vector)
        if target_unit is not None:
            for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
                matrices.append(
                    _axis_angle_rotation_matrix(target_unit, angle) @ base
                )
    return _dedupe_rotation_matrices(matrices)


def _molecule_role_vector(
    species: str,
    cart_offsets: np.ndarray,
    elements: list[str],
) -> np.ndarray | None:
    heavy_indices = [
        index for index, element in enumerate(elements) if element != "H"
    ]
    if not heavy_indices:
        return None
    heavy_center = np.mean(cart_offsets[heavy_indices], axis=0)
    preferred = ("N",) if species in DONOR_MOLECULES else ("O", "S", "N")
    selected = [
        index
        for index, element in enumerate(elements)
        if element in preferred and element != "H"
    ]
    if not selected:
        selected = heavy_indices[:1]
    vector = np.mean(cart_offsets[selected], axis=0) - heavy_center
    return vector if np.linalg.norm(vector) > 1.0e-9 else None


def _molecule_target_vector(
    species: str,
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> np.ndarray | None:
    targets: list[np.ndarray] = []
    if species in DONOR_MOLECULES:
        targets.extend(
            np.asarray(adjusted[int(row["row_index"])], dtype=float)
            for row in inorganic_rows
            if str(row["element"]) in HALIDE_ELEMENTS
            and int(row["row_index"]) in adjusted
        )
        for (
            occupied_coords,
            occupied_elements,
            occupied_species,
        ) in occupied_bodies:
            if occupied_species not in ACCEPTOR_MOLECULES:
                continue
            targets.extend(
                coord
                for coord, element in zip(
                    occupied_coords,
                    occupied_elements,
                    strict=True,
                )
                if element in {"O", "N", "S"}
            )
    elif species in ACCEPTOR_MOLECULES:
        for (
            occupied_coords,
            occupied_elements,
            occupied_species,
        ) in occupied_bodies:
            if occupied_species not in DONOR_MOLECULES:
                continue
            targets.extend(
                coord
                for coord, element in zip(
                    occupied_coords,
                    occupied_elements,
                    strict=True,
                )
                if element == "H"
            )
    if not targets:
        return None
    best_target = min(
        targets,
        key=lambda target: _pbc_distance(candidate_center, target, lattice),
    )
    delta = np.asarray(best_target, dtype=float) - np.asarray(
        candidate_center,
        dtype=float,
    )
    delta -= np.round(delta)
    cart = delta @ lattice
    return cart if np.linalg.norm(cart) > 1.0e-9 else None


def _molecule_pose_score(
    coords: np.ndarray,
    elements: list[str],
    species: str,
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_centers: list[np.ndarray],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    min_inorganic = math.inf
    inorganic_clash_penalty = 0.0
    for coord, element in zip(coords, elements, strict=True):
        for inorganic in inorganic_rows:
            distance = _pbc_distance(
                coord,
                adjusted[int(inorganic["row_index"])],
                lattice,
            )
            min_inorganic = min(min_inorganic, distance)
            minimum = _minimum_pair_distance(
                str(element),
                str(inorganic["element"]),
                True,
                False,
            )
            if distance < minimum:
                weight = 1.5 if element != "H" else 0.8
                inorganic_clash_penalty += weight * (
                    (minimum - distance) / max(minimum, 1.0e-9)
                )
    min_body = math.inf
    for center in occupied_centers:
        min_body = min(
            min_body,
            _pbc_distance(candidate_center, center, lattice),
        )
    min_body_atom = math.inf
    body_clash_penalty = 0.0
    for occupied_coords, occupied_elements, _ in occupied_bodies:
        for left_coord, left_element in zip(coords, elements, strict=True):
            for right_coord, right_element in zip(
                occupied_coords,
                occupied_elements,
                strict=True,
            ):
                distance = _pbc_distance(left_coord, right_coord, lattice)
                minimum = _minimum_pair_distance(
                    str(left_element),
                    str(right_element),
                    True,
                    True,
                )
                min_body_atom = min(min_body_atom, distance)
                if distance < minimum:
                    heavy_pair = "H" not in {left_element, right_element}
                    weight = 1.4 if heavy_pair else 0.8
                    body_clash_penalty += weight * (
                        (minimum - distance) / max(minimum, 1.0e-9)
                    )
    if not np.isfinite(min_inorganic):
        min_inorganic = 0.0
    if not np.isfinite(min_body):
        min_body = 6.0
    if not np.isfinite(min_body_atom):
        min_body_atom = 4.0
    hbond_score = _pose_hydrogen_bond_score(
        coords,
        elements,
        species,
        inorganic_rows,
        occupied_bodies,
        adjusted,
        lattice,
    )
    internal_penalty = _body_internal_restraint_penalty(
        coords,
        elements,
        species,
        lattice,
    )
    boundary_penalty = _body_unit_cell_boundary_penalty(coords)
    return (
        0.9 * min(min_inorganic, 3.0)
        + 0.18 * min(min_body, 6.0)
        + 0.32 * min(min_body_atom, 4.0)
        + 1.1 * hbond_score
        - 2.6 * inorganic_clash_penalty
        - 2.2 * body_clash_penalty
        - 4.0 * internal_penalty
        - 5.0 * boundary_penalty
    )


def _body_internal_restraint_penalty(
    coords: np.ndarray,
    elements: list[str],
    species: str,
    lattice: np.ndarray,
) -> float:
    template = REFERENCE_MOLECULES.get(species, {}).get("atoms", ())
    if len(template) != len(coords):
        return 0.0
    template_coords = np.asarray(
        [
            [float(atom[1]), float(atom[2]), float(atom[3])]
            for atom in template
        ],
        dtype=float,
    )
    penalty = 0.0
    for left_index in range(len(coords)):
        for right_index in range(left_index + 1, len(coords)):
            target = float(
                np.linalg.norm(
                    template_coords[left_index] - template_coords[right_index]
                )
            )
            if target <= 0.1:
                continue
            observed = _pbc_distance(
                coords[left_index],
                coords[right_index],
                lattice,
            )
            tolerance = (
                0.18
                if "H" not in {elements[left_index], elements[right_index]}
                else 0.28
            )
            deviation = abs(observed - target)
            if deviation > tolerance:
                weight = 1.2 if tolerance == 0.18 else 0.6
                penalty += weight * (deviation - tolerance) / max(target, 1.0)
    return penalty


def _body_unit_cell_boundary_penalty(
    coords: np.ndarray,
    *,
    span_limit: float = 0.72,
) -> float:
    if coords.size == 0:
        return 0.0
    wrapped = np.asarray(coords, dtype=float) % 1.0
    spans = np.max(wrapped, axis=0) - np.min(wrapped, axis=0)
    return float(np.sum(np.clip(spans - span_limit, 0.0, None)))


def _relax_halides_for_organic_contacts(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
    adjusted: dict[int, np.ndarray],
) -> None:
    cations = [
        row for row in rows if str(row["element"]) in METAL_CATION_ELEMENTS
    ]
    halides = [row for row in rows if str(row["element"]) in HALIDE_ELEMENTS]
    organics = [row for row in rows if _is_molecular_site(str(row["label"]))]
    if not cations or not halides or not organics:
        return
    base_candidates = _halide_site_candidates(cations, lattice, adjusted)
    if not base_candidates:
        return
    assigned = [
        np.asarray(adjusted[int(row["row_index"])], dtype=float)
        for row in halides
        if int(row["row_index"]) in adjusted
    ]
    for halide in sorted(halides, key=lambda item: str(item["label"])):
        row_index = int(halide["row_index"])
        current = np.asarray(
            adjusted.get(row_index, halide["frac"]), dtype=float
        )
        candidates = _dedupe_fractional_points([current, *base_candidates])
        current_score = _halide_organic_relaxation_score(
            current,
            halide,
            cations,
            organics,
            assigned,
            adjusted,
            lattice,
        )
        best = current
        best_score = current_score
        for candidate in candidates:
            score = _halide_organic_relaxation_score(
                candidate,
                halide,
                cations,
                organics,
                assigned,
                adjusted,
                lattice,
            )
            if score < best_score:
                best = candidate
                best_score = score
        if best_score + 0.05 < current_score:
            adjusted[row_index] = best % 1.0
            assigned = [
                best if np.allclose(coord, current) else coord
                for coord in assigned
            ]


def _halide_organic_relaxation_score(
    candidate: np.ndarray,
    halide: dict[str, Any],
    cations: list[dict[str, Any]],
    organics: list[dict[str, Any]],
    assigned_halides: list[np.ndarray],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    score = 0.0
    coordinated = 0
    for cation in cations:
        distance = _pbc_distance(
            candidate,
            adjusted[int(cation["row_index"])],
            lattice,
        )
        lower, upper = _coordination_distance_window(str(cation["element"]))
        if lower <= distance <= upper:
            coordinated += 1
        elif distance < lower:
            score += 2.0 * (lower - distance)
        elif distance < upper + 1.0:
            score += 0.4 * (distance - upper)
    if coordinated == 0:
        score += 2.0
    for other in assigned_halides:
        if np.allclose(other, adjusted[int(halide["row_index"])]):
            continue
        distance = _pbc_distance(candidate, other, lattice)
        if distance < 2.45:
            score += 1.4 * (2.45 - distance)
    for organic in organics:
        organic_coord = adjusted[int(organic["row_index"])]
        distance = _pbc_distance(candidate, organic_coord, lattice)
        minimum = _minimum_pair_distance(
            str(halide["element"]),
            str(organic["element"]),
            False,
            True,
        )
        if distance < minimum:
            score += 1.6 * (minimum - distance) / max(minimum, 1.0e-9)
    hbond_bonus = _halide_hydrogen_bond_bonus(
        candidate, organics, adjusted, lattice
    )
    return score - 0.35 * hbond_bonus


def _body_cart_offsets(
    coords: np.ndarray,
    center: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(coords, dtype=float) - np.asarray(center, dtype=float)
    delta -= np.round(delta)
    return delta @ lattice


def _dedupe_body_poses(poses: list[np.ndarray]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for pose in poses:
        wrapped = np.asarray(pose, dtype=float) % 1.0
        key = tuple(
            tuple(int(round(value * 1000.0)) for value in row)
            for row in wrapped
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(wrapped)
    return deduped


def _dedupe_rotation_matrices(matrices: list[np.ndarray]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for matrix in matrices:
        rounded = tuple(int(round(value * 1000.0)) for value in matrix.ravel())
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(matrix)
    return deduped


def _axis_angle_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = _unit_vector(axis)
    if unit is None:
        return np.eye(3)
    x_axis, y_axis, z_axis = unit
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    one_minus = 1.0 - cos_angle
    return np.asarray(
        [
            [
                cos_angle + x_axis * x_axis * one_minus,
                x_axis * y_axis * one_minus - z_axis * sin_angle,
                x_axis * z_axis * one_minus + y_axis * sin_angle,
            ],
            [
                y_axis * x_axis * one_minus + z_axis * sin_angle,
                cos_angle + y_axis * y_axis * one_minus,
                y_axis * z_axis * one_minus - x_axis * sin_angle,
            ],
            [
                z_axis * x_axis * one_minus - y_axis * sin_angle,
                z_axis * y_axis * one_minus + x_axis * sin_angle,
                cos_angle + z_axis * z_axis * one_minus,
            ],
        ],
        dtype=float,
    )


def _rotation_matrix_between(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    source_unit = _unit_vector(source)
    target_unit = _unit_vector(target)
    if source_unit is None or target_unit is None:
        return np.eye(3)
    cross = np.cross(source_unit, target_unit)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm <= 1.0e-9:
        if dot > 0:
            return np.eye(3)
        fallback_axis = np.cross(source_unit, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback_axis) <= 1.0e-9:
            fallback_axis = np.cross(source_unit, np.asarray([0.0, 1.0, 0.0]))
        return _axis_angle_rotation_matrix(fallback_axis, math.pi)
    return _axis_angle_rotation_matrix(cross / norm, math.acos(dot))


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        return None
    return values / norm


def _pose_hydrogen_bond_score(
    coords: np.ndarray,
    elements: list[str],
    species: str,
    inorganic_rows: list[dict[str, Any]],
    occupied_bodies: list[tuple[np.ndarray, list[str], str]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    if species not in DONOR_MOLECULES and species not in ACCEPTOR_MOLECULES:
        return 0.0
    score = 0.0
    if species in DONOR_MOLECULES:
        acceptors = [
            (adjusted[int(row["row_index"])], str(row["element"]))
            for row in inorganic_rows
            if str(row["element"]) in HALIDE_ELEMENTS
            and int(row["row_index"]) in adjusted
        ]
        for (
            occupied_coords,
            occupied_elements,
            occupied_species,
        ) in occupied_bodies:
            if occupied_species not in ACCEPTOR_MOLECULES:
                continue
            acceptors.extend(
                (coord, element)
                for coord, element in zip(
                    occupied_coords,
                    occupied_elements,
                    strict=True,
                )
                if element in {"O", "N", "S"}
            )
        for donor_index, hydrogen_index in _donor_hydrogen_pairs(
            coords, elements, lattice
        ):
            score += _best_hydrogen_bond_score(
                coords[donor_index],
                coords[hydrogen_index],
                acceptors,
                lattice,
            )
    if species in ACCEPTOR_MOLECULES:
        donors = []
        for (
            occupied_coords,
            occupied_elements,
            occupied_species,
        ) in occupied_bodies:
            if occupied_species not in DONOR_MOLECULES:
                continue
            donor_pairs = _donor_hydrogen_pairs(
                occupied_coords,
                occupied_elements,
                lattice,
            )
            donors.extend(
                (occupied_coords[donor_index], occupied_coords[hydrogen_index])
                for donor_index, hydrogen_index in donor_pairs
            )
        acceptor_indices = [
            index
            for index, element in enumerate(elements)
            if element in {"O", "N", "S"}
        ]
        for acceptor_index in acceptor_indices:
            for donor_coord, hydrogen_coord in donors:
                score += _single_hydrogen_bond_score(
                    donor_coord,
                    hydrogen_coord,
                    coords[acceptor_index],
                    lattice,
                )
    if species in DONOR_MOLECULES and score <= 0.0:
        return -0.25
    return min(2.0, score)


def _halide_hydrogen_bond_bonus(
    halide_coord: np.ndarray,
    organics: list[dict[str, Any]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    bodies: dict[str, list[dict[str, Any]]] = {}
    for row in organics:
        body = _molecular_body_token(str(row["label"]))
        if body:
            bodies.setdefault(body, []).append(row)
    bonus = 0.0
    for body, body_rows in bodies.items():
        if _molecular_species_token(body) not in DONOR_MOLECULES:
            continue
        coords = np.asarray(
            [adjusted[int(row["row_index"])] for row in body_rows],
            dtype=float,
        )
        elements = [str(row["element"]) for row in body_rows]
        for donor_index, hydrogen_index in _donor_hydrogen_pairs(
            coords,
            elements,
            lattice,
        ):
            bonus += _single_hydrogen_bond_score(
                coords[donor_index],
                coords[hydrogen_index],
                halide_coord,
                lattice,
            )
    return min(2.0, bonus)


def _best_hydrogen_bond_score(
    donor_coord: np.ndarray,
    hydrogen_coord: np.ndarray,
    acceptors: list[tuple[np.ndarray, str]],
    lattice: np.ndarray,
) -> float:
    return max(
        (
            _single_hydrogen_bond_score(
                donor_coord,
                hydrogen_coord,
                acceptor_coord,
                lattice,
            )
            for acceptor_coord, _ in acceptors
        ),
        default=0.0,
    )


def _single_hydrogen_bond_score(
    donor_coord: np.ndarray,
    hydrogen_coord: np.ndarray,
    acceptor_coord: np.ndarray,
    lattice: np.ndarray,
) -> float:
    h_to_acceptor = _pbc_cartesian_delta(
        hydrogen_coord, acceptor_coord, lattice
    )
    h_to_donor = _pbc_cartesian_delta(hydrogen_coord, donor_coord, lattice)
    distance = float(np.linalg.norm(h_to_acceptor))
    if distance < 1.45 or distance > 3.35:
        return 0.0
    angle = _angle_degrees(h_to_donor, h_to_acceptor)
    if angle < 105.0:
        return 0.0
    distance_score = max(0.0, 1.0 - abs(distance - 2.45) / 1.05)
    angle_score = max(0.0, (angle - 105.0) / 75.0)
    return distance_score * angle_score


def _donor_hydrogen_pairs(
    coords: np.ndarray,
    elements: list[str],
    lattice: np.ndarray,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    donor_indices = [
        index for index, element in enumerate(elements) if element == "N"
    ]
    hydrogen_indices = [
        index for index, element in enumerate(elements) if element == "H"
    ]
    for donor_index in donor_indices:
        for hydrogen_index in hydrogen_indices:
            distance = _pbc_distance(
                coords[donor_index],
                coords[hydrogen_index],
                lattice,
            )
            if 0.75 <= distance <= 1.30:
                pairs.append((donor_index, hydrogen_index))
    return pairs


def _pbc_cartesian_delta(
    left_frac: np.ndarray,
    right_frac: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(right_frac, dtype=float) - np.asarray(
        left_frac, dtype=float
    )
    delta -= np.round(delta)
    return delta @ lattice


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = _unit_vector(left)
    right_unit = _unit_vector(right)
    if left_unit is None or right_unit is None:
        return 0.0
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _write_shifted_cif(
    source_path: Path,
    output_path: Path,
    *,
    fractional_shift: tuple[float, float, float],
    element_selector: set[str] | None = None,
    label_prefix_selector: set[str] | None = None,
) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shifted_lines = []
    in_atom_rows = False
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            shifted_lines.append(line)
            continue
        if (
            not in_atom_rows
            or not stripped
            or stripped.startswith(("_", "#", "loop_"))
        ):
            shifted_lines.append(line)
            continue
        parts = stripped.split()
        if len(parts) < 6 or not _selected_atom_site(
            parts,
            element_selector=element_selector,
            label_prefix_selector=label_prefix_selector,
        ):
            shifted_lines.append(line)
            continue
        try:
            coords = [float(parts[index]) for index in (2, 3, 4)]
        except ValueError:
            shifted_lines.append(line)
            continue
        shifted = [
            (coord + delta) % 1.0
            for coord, delta in zip(coords, fractional_shift)
        ]
        parts[2:5] = [f"{value:.6f}" for value in shifted]
        shifted_lines.append(" ".join(parts))
    output_path.write_text(
        "\n".join(shifted_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def _selected_atom_site(
    parts: list[str],
    *,
    element_selector: set[str] | None,
    label_prefix_selector: set[str] | None,
) -> bool:
    if element_selector is not None and parts[1] in element_selector:
        return True
    if label_prefix_selector is not None:
        token = _label_token(parts[0])
        return any(
            token.startswith(prefix) for prefix in label_prefix_selector
        )
    return False


def _label_token(value: str) -> str:
    return "".join(char.upper() for char in str(value) if char.isalnum())


def _formula_sum(composition: dict[str, float]) -> str:
    pieces = []
    for element in sorted(composition):
        count = composition[element]
        count_text = (
            str(int(round(count)))
            if abs(count - round(count)) <= 1.0e-9
            else f"{count:.4g}"
        )
        pieces.append(f"{element}{count_text}")
    return " ".join(pieces) or "X1"


def _comparison_parameter_grid(
    target_params: GIWAXSSimulationParameters,
    cfg: BenchmarkRunConfig,
) -> list[GIWAXSSimulationParameters]:
    params = []
    for offset in cfg.comparison_theta_x_offsets:
        theta_x = cfg.fiber_tilt_center_deg + offset
        for theta_y in cfg.comparison_theta_y_values:
            params.append(
                GIWAXSSimulationParameters(
                    sigma_theta=target_params.sigma_theta,
                    sigma_phi=target_params.sigma_phi,
                    sigma_r=target_params.sigma_r,
                    hkl_extent=cfg.hkl_extent,
                    theta_x_deg=theta_x,
                    theta_y_deg=theta_y,
                    qxy_min=cfg.qxy_range[0],
                    qxy_max=cfg.qxy_range[1],
                    qz_min=cfg.qz_range[0],
                    qz_max=cfg.qz_range[1],
                    resolution_z=cfg.detector_shape[0],
                    resolution_x=cfg.detector_shape[1],
                )
            )
    return params or [target_params]


def _comparison_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        return (math.inf, math.inf, math.inf)
    return (
        float(metrics.get("peak_focus_score", math.inf))
        + float(item.get("density_penalty", 0.0) or 0.0)
        + float(item.get("physical_penalty", 0.0) or 0.0)
        + float(item.get("charge_penalty", 0.0) or 0.0)
        + float(item.get("lattice_prior_penalty", 0.0) or 0.0)
        + float(item.get("bragg_intensity_penalty", 0.0) or 0.0),
        float(metrics.get("difference_rmse", math.inf)),
        -float(metrics.get("correlation", 0.0)),
    )


def _path_chemistry_rank_fields(path: Path) -> dict[str, Any]:
    density = _cif_density_g_cm3(path)
    chemistry_metrics = _cif_physical_chemistry_metrics(path)
    return {
        "density_g_cm3": density,
        "density_penalty": _density_plausibility_penalty(path, density),
        "chemistry_metrics": chemistry_metrics,
        "physical_penalty": float(
            chemistry_metrics.get("physical_penalty", 0.0) or 0.0
        ),
    }


def _cif_density_g_cm3(path: Path) -> float | None:
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(Structure.from_file(path).density)
    except Exception:
        return None


def _density_plausibility_penalty(
    path: Path,
    density_g_cm3: float | None,
) -> float:
    if density_g_cm3 is None or not np.isfinite(density_g_cm3):
        return 0.0
    if not _looks_like_heavy_halide_cif(path):
        return 0.0
    density = float(density_g_cm3)
    if density < 3.0:
        return min(1.25, (3.0 - density) * 0.55)
    if density > 7.5:
        return min(1.0, (density - 7.5) * 0.20)
    return 0.0


def _looks_like_heavy_halide_cif(path: Path) -> bool:
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elements = {
                str(element)
                for element in Structure.from_file(path).composition.elements
            }
    except Exception:
        return False
    return bool(elements & {"Pb", "Sn"}) and bool(elements & {"I", "Br", "Cl"})


def _cif_physical_chemistry_metrics(path: Path) -> dict[str, Any]:
    rows, lattice = _read_simple_cif_sites(path)
    if not rows or lattice is None:
        return {"physical_penalty": 0.0, "status": "unavailable"}
    clash_count = 0
    clash_penalty = 0.0
    organic_inorganic_clash_count = 0
    cation_halide_close_count = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if _same_molecular_body(left["label"], right["label"]):
                continue
            distance = _pbc_distance(left["frac"], right["frac"], lattice)
            minimum = _minimum_pair_distance(
                str(left["element"]),
                str(right["element"]),
                _is_molecular_site(left["label"]),
                _is_molecular_site(right["label"]),
            )
            if distance < minimum:
                clash_count += 1
                clash_penalty += (minimum - distance) / max(minimum, 1.0e-9)
                if _organic_inorganic_pair(left["label"], right["label"]):
                    organic_inorganic_clash_count += 1
                if _cation_halide_pair(left["element"], right["element"]):
                    cation_halide_close_count += 1
    coordination = _coordination_metrics(rows, lattice)
    coordination_penalty = float(coordination.get("penalty", 0.0))
    organic_restraints = _organic_restraint_metrics(rows, lattice)
    hydrogen_bonding = _hydrogen_bond_metrics(rows, lattice)
    physical_penalty = min(
        2.0,
        coordination_penalty
        + 0.02 * clash_penalty
        + 0.04 * organic_inorganic_clash_count
        + 0.03 * cation_halide_close_count
        + 0.08 * float(organic_restraints.get("penalty", 0.0))
        + 0.05 * float(hydrogen_bonding.get("penalty", 0.0)),
    )
    return {
        "physical_penalty": physical_penalty,
        "clash_count": clash_count,
        "organic_inorganic_clash_count": organic_inorganic_clash_count,
        "cation_halide_close_count": cation_halide_close_count,
        "coordination": coordination,
        "organic_restraints": organic_restraints,
        "hydrogen_bonding": hydrogen_bonding,
        "policy": (
            "hard-sphere contacts, organic-inorganic separation, and "
            "Pb/Sn-halide coordination priors; rigid-body organic "
            "DFIX/DANG/SADI/FLAT-like geometry checks; donor/acceptor "
            "hydrogen-bond orientation; no free-floating inorganic ions"
        ),
    }


def _coordination_metrics(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
) -> dict[str, Any]:
    cations = [
        row for row in rows if str(row["element"]) in METAL_CATION_ELEMENTS
    ]
    halides = [row for row in rows if str(row["element"]) in HALIDE_ELEMENTS]
    if not cations or not halides:
        return {"penalty": 0.0, "cation_count": len(cations)}
    records = []
    halide_records = []
    penalty = 0.0
    free_cation_count = 0
    for cation in cations:
        distances = sorted(
            _pbc_distance(cation["frac"], halide["frac"], lattice)
            for halide in halides
        )
        near = [
            distance
            for distance in distances
            if _coordination_distance_window(str(cation["element"]))[0]
            <= distance
            <= _coordination_distance_window(str(cation["element"]))[1]
        ]
        count = len(near)
        if count == 0:
            free_cation_count += 1
            penalty += 0.25
        if count < 4:
            penalty += 0.08 * (4 - count)
        elif count > 7:
            penalty += 0.025 * (count - 7)
        if distances:
            lower, upper = _coordination_distance_window(
                str(cation["element"])
            )
            nearest = float(distances[0])
            if nearest < lower - 0.25:
                penalty += min(0.16, (lower - 0.25 - nearest) * 0.12)
            mean_six = float(np.mean(distances[: min(6, len(distances))]))
            if mean_six > upper + 0.45:
                penalty += min(0.16, (mean_six - upper - 0.45) * 0.08)
        records.append(
            {
                "label": cation["label"],
                "element": cation["element"],
                "halide_neighbors": count,
                "nearest_halide_distance": distances[0] if distances else None,
                "mean_six_halide_distance": (
                    float(np.mean(distances[: min(6, len(distances))]))
                    if distances
                    else None
                ),
            }
        )
    uncoordinated_halide_count = 0
    for halide in halides:
        distances = []
        for cation in cations:
            distance = _pbc_distance(halide["frac"], cation["frac"], lattice)
            lower, upper = _coordination_distance_window(
                str(cation["element"])
            )
            distances.append((distance, lower, upper))
        nearest = min((item[0] for item in distances), default=math.inf)
        coordinated = [
            distance
            for distance, lower, upper in distances
            if lower - 0.45 <= distance <= upper + 0.45
        ]
        if not coordinated:
            uncoordinated_halide_count += 1
            penalty += 0.10 if nearest < math.inf else 0.20
        if nearest > 4.35:
            penalty += min(0.18, 0.05 * (nearest - 4.35))
        halide_records.append(
            {
                "label": halide["label"],
                "element": halide["element"],
                "coordinated_cation_count": len(coordinated),
                "nearest_cation_distance": (
                    float(nearest) if np.isfinite(nearest) else None
                ),
            }
        )
    return {
        "penalty": min(1.0, penalty),
        "cation_count": len(cations),
        "halide_count": len(halides),
        "free_cation_count": free_cation_count,
        "uncoordinated_halide_count": uncoordinated_halide_count,
        "cation_records": records[:12],
        "halide_records": halide_records[:16],
    }


def _organic_restraint_metrics(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
) -> dict[str, Any]:
    bodies = _molecular_bodies(rows)
    penalty = 0.0
    body_records = []
    for body_token, body_rows in sorted(bodies.items()):
        species = _molecular_species_token(body_token)
        template = REFERENCE_MOLECULES.get(species, {}).get("atoms", ())
        coords = np.asarray([row["frac"] for row in body_rows], dtype=float)
        elements = [str(row["element"]) for row in body_rows]
        max_deviation = 0.0
        pair_count = 0
        if len(template) == len(body_rows):
            template_coords = np.asarray(
                [
                    [float(atom[1]), float(atom[2]), float(atom[3])]
                    for atom in template
                ],
                dtype=float,
            )
            for left_index in range(len(body_rows)):
                for right_index in range(left_index + 1, len(body_rows)):
                    observed = _pbc_distance(
                        coords[left_index],
                        coords[right_index],
                        lattice,
                    )
                    target = float(
                        np.linalg.norm(
                            template_coords[left_index]
                            - template_coords[right_index]
                        )
                    )
                    if target <= 0.1:
                        continue
                    deviation = abs(observed - target)
                    max_deviation = max(max_deviation, deviation)
                    pair_count += 1
                    tolerance = (
                        0.18
                        if "H"
                        not in {
                            elements[left_index],
                            elements[right_index],
                        }
                        else 0.28
                    )
                    if deviation > tolerance:
                        weight = 1.2 if tolerance == 0.18 else 0.6
                        penalty += (
                            weight * (deviation - tolerance) / max(target, 1.0)
                        )
        planarity_rms = _planarity_rms_for_body(
            species, coords, elements, lattice
        )
        if planarity_rms > 0.18:
            penalty += 1.5 * (planarity_rms - 0.18)
        fractional_spans = (
            np.max(coords % 1.0, axis=0) - np.min(coords % 1.0, axis=0)
            if coords.size
            else np.zeros(3, dtype=float)
        )
        boundary_penalty = _body_unit_cell_boundary_penalty(coords)
        if boundary_penalty > 0.0:
            penalty += 1.5 * boundary_penalty
        body_records.append(
            {
                "body": body_token,
                "species": species,
                "pair_restraint_count": pair_count,
                "max_pair_distance_deviation": max_deviation,
                "planarity_rms": planarity_rms,
                "max_fractional_span": float(np.max(fractional_spans)),
                "unit_cell_boundary_penalty": boundary_penalty,
            }
        )
    return {
        "penalty": min(1.0, penalty),
        "body_count": len(bodies),
        "body_records": body_records[:12],
        "policy": "rigid-body DFIX/DANG/SADI and FLAT-style checks",
    }


def _hydrogen_bond_metrics(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
) -> dict[str, Any]:
    bodies = _molecular_bodies(rows)
    acceptors = [
        (row["frac"], str(row["element"]), "")
        for row in rows
        if str(row["element"]) in HALIDE_ELEMENTS
    ]
    for body_token, body_rows in bodies.items():
        species = _molecular_species_token(body_token)
        if species not in ACCEPTOR_MOLECULES:
            continue
        acceptors.extend(
            (row["frac"], str(row["element"]), body_token)
            for row in body_rows
            if str(row["element"]) in {"O", "N", "S"}
        )
    donor_count = 0
    plausible_count = 0
    best_scores = []
    for body_token, body_rows in bodies.items():
        species = _molecular_species_token(body_token)
        if species not in DONOR_MOLECULES:
            continue
        coords = np.asarray([row["frac"] for row in body_rows], dtype=float)
        elements = [str(row["element"]) for row in body_rows]
        pairs = _donor_hydrogen_pairs(coords, elements, lattice)
        if not pairs:
            continue
        donor_count += 1
        best = 0.0
        filtered_acceptors = [
            (coord, element)
            for coord, element, owner in acceptors
            if owner != body_token
        ]
        for donor_index, hydrogen_index in pairs:
            best = max(
                best,
                _best_hydrogen_bond_score(
                    coords[donor_index],
                    coords[hydrogen_index],
                    filtered_acceptors,
                    lattice,
                ),
            )
        best_scores.append(best)
        if best >= 0.20:
            plausible_count += 1
    penalty = 0.0
    if donor_count and acceptors:
        penalty = max(0, donor_count - plausible_count) / donor_count
    return {
        "penalty": float(penalty),
        "donor_body_count": donor_count,
        "plausible_donor_body_count": plausible_count,
        "mean_best_score": float(np.mean(best_scores)) if best_scores else 0.0,
    }


def _planarity_rms_for_body(
    species: str,
    coords: np.ndarray,
    elements: list[str],
    lattice: np.ndarray,
) -> float:
    restraints = MOLECULE_POSE_RESTRAINTS.get(species, {})
    if not restraints.get("planar_heavy_atom_restraint"):
        return 0.0
    heavy_indices = [
        index
        for index, element in enumerate(elements)
        if element in ORGANIC_HEAVY_ELEMENTS
    ]
    if len(heavy_indices) < 4:
        return 0.0
    reference = coords[heavy_indices[0]]
    cart = []
    for index in heavy_indices:
        delta = coords[index] - reference
        delta -= np.round(delta)
        cart.append(delta @ lattice)
    cartesian = np.asarray(cart, dtype=float)
    centered = cartesian - np.mean(cartesian, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances * distances)))


def _dedupe_fractional_points(points: list[np.ndarray]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for point in points:
        wrapped = np.asarray(point, dtype=float) % 1.0
        key = tuple(int(round(value * 1000.0)) for value in wrapped)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(wrapped)
    return deduped


def _fractional_center(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros(3, dtype=float)
    reference = np.asarray(coords[0], dtype=float)
    unwrapped = []
    for coord in coords:
        delta = np.asarray(coord, dtype=float) - reference
        delta -= np.round(delta)
        unwrapped.append(reference + delta)
    return (np.mean(np.asarray(unwrapped), axis=0) % 1.0).astype(float)


def _molecule_void_score(
    coords: np.ndarray,
    elements: list[str],
    candidate_center: np.ndarray,
    inorganic_rows: list[dict[str, Any]],
    occupied_centers: list[np.ndarray],
    occupied_bodies: list[tuple[np.ndarray, list[str]]],
    adjusted: dict[int, np.ndarray],
    lattice: np.ndarray,
) -> float:
    current_center = _fractional_center(coords)
    shifted = (coords + (candidate_center - current_center)) % 1.0
    min_inorganic = math.inf
    for coord in shifted:
        for inorganic in inorganic_rows:
            distance = _pbc_distance(
                coord,
                adjusted[int(inorganic["row_index"])],
                lattice,
            )
            min_inorganic = min(min_inorganic, distance)
    min_body = math.inf
    for center in occupied_centers:
        min_body = min(
            min_body,
            _pbc_distance(candidate_center, center, lattice),
        )
    min_body_atom = math.inf
    body_clash_penalty = 0.0
    for occupied_coords, occupied_elements in occupied_bodies:
        for left_coord, left_element in zip(shifted, elements, strict=True):
            for right_coord, right_element in zip(
                occupied_coords,
                occupied_elements,
                strict=True,
            ):
                distance = _pbc_distance(left_coord, right_coord, lattice)
                minimum = _minimum_pair_distance(
                    str(left_element),
                    str(right_element),
                    True,
                    True,
                )
                min_body_atom = min(min_body_atom, distance)
                if distance < minimum:
                    body_clash_penalty += (minimum - distance) / max(
                        minimum,
                        1.0e-9,
                    )
    if not np.isfinite(min_inorganic):
        min_inorganic = 0.0
    if not np.isfinite(min_body):
        min_body = 6.0
    if not np.isfinite(min_body_atom):
        min_body_atom = 4.0
    return (
        min(min_inorganic, 4.0)
        + 0.2 * min(min_body, 6.0)
        + 0.35 * min(min_body_atom, 4.0)
        - 2.0 * body_clash_penalty
    )


def _read_simple_cif_sites(
    path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], None
    cell: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    in_atom_rows = False
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] in {
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        }:
            try:
                cell[parts[0]] = _cif_number(parts[1])
            except (IndexError, ValueError):
                pass
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or stripped.startswith(("_", "#", "loop_"))
            or len(parts) < 6
        ):
            continue
        try:
            frac = np.asarray([float(parts[index]) for index in (2, 3, 4)])
        except ValueError:
            continue
        rows.append(
            {
                "label": str(parts[0]),
                "element": str(parts[1]),
                "frac": frac,
                "row_index": line_index,
            }
        )
    lattice = _lattice_matrix_from_cell(cell)
    return rows, lattice


def _lattice_matrix_from_cell(cell: dict[str, float]) -> np.ndarray | None:
    try:
        a = float(cell["_cell_length_a"])
        b = float(cell["_cell_length_b"])
        c = float(cell["_cell_length_c"])
        alpha = math.radians(float(cell["_cell_angle_alpha"]))
        beta = math.radians(float(cell["_cell_angle_beta"]))
        gamma = math.radians(float(cell["_cell_angle_gamma"]))
    except KeyError:
        return None
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) <= 1.0e-12:
        return None
    avec = np.asarray([a, 0.0, 0.0])
    bvec = np.asarray([b * math.cos(gamma), b * sin_gamma, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    cvec = np.asarray([cx, cy, math.sqrt(cz_sq)])
    return np.vstack([avec, bvec, cvec])


def _pbc_distance(
    left_frac: np.ndarray,
    right_frac: np.ndarray,
    lattice: np.ndarray,
) -> float:
    delta = np.asarray(left_frac, dtype=float) - np.asarray(
        right_frac, dtype=float
    )
    delta -= np.round(delta)
    cartesian = delta @ lattice
    return float(np.linalg.norm(cartesian))


def _minimum_pair_distance(
    left_element: str,
    right_element: str,
    left_molecular: bool,
    right_molecular: bool,
) -> float:
    if left_molecular != right_molecular:
        if "H" in {left_element, right_element}:
            return 1.55
        return 2.25
    pair = frozenset((left_element, right_element))
    if pair <= {"Pb", "Sn", "Ge"}:
        return 3.2
    if pair <= {"I", "Br", "Cl", "F"}:
        return 2.45
    if _cation_halide_pair(left_element, right_element):
        return 2.25
    if "H" in pair:
        return 0.75
    return 1.1


def _coordination_distance_window(element: str) -> tuple[float, float]:
    if element == "Sn":
        return (2.55, 3.35)
    if element == "Ge":
        return (2.35, 3.15)
    return (2.65, 3.65)


def _cation_halide_pair(left_element: str, right_element: str) -> bool:
    pair = {str(left_element), str(right_element)}
    return bool(pair & {"Pb", "Sn", "Ge"}) and bool(
        pair & {"I", "Br", "Cl", "F"}
    )


def _organic_inorganic_pair(left_label: str, right_label: str) -> bool:
    return _is_molecular_site(left_label) != _is_molecular_site(right_label)


def _same_molecular_body(left_label: str, right_label: str) -> bool:
    left = _molecular_body_token(left_label)
    return bool(left) and left == _molecular_body_token(right_label)


def _is_molecular_site(label: str) -> bool:
    return bool(_molecular_body_token(label))


def _molecular_body_token(label: str) -> str:
    token = _label_token(label)
    for molecule in ("DMF", "DMSO", "NMP", "MA", "FA", "BA"):
        if token.startswith(molecule):
            digits = []
            for char in token[len(molecule) :]:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                remainder = token[len(molecule) + len(digits) :]
                suffix = ""
                if any(char.isdigit() for char in remainder):
                    trailing = []
                    for char in reversed(remainder):
                        if char.isalpha():
                            trailing.append(char)
                        else:
                            break
                    suffix = "".join(reversed(trailing))
                return molecule + "".join(digits) + suffix
    return ""


def _molecular_species_token(body_token: str) -> str:
    token = _label_token(body_token)
    for molecule in ("DMF", "DMSO", "NMP", "MA", "FA", "BA"):
        if token.startswith(molecule):
            return molecule
    return "".join(char for char in token if not char.isdigit())


def _molecular_bodies(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    bodies: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        token = _molecular_body_token(str(row["label"]))
        if token:
            bodies.setdefault(token, []).append(row)
    return bodies


def _validate_best_generated_structure(
    comparisons: list[dict[str, Any]],
    reference_cif: Path,
    local_best_dir: Path,
    global_best_dir: Path,
) -> dict[str, Any]:
    if not comparisons:
        return {"status": "no_generated_cif_rankings"}
    best = comparisons[0]
    best_path = Path(str(best.get("path", "")))
    if not best_path.exists():
        return {
            "status": "best_generated_cif_missing",
            "best_generated_cif": str(best_path),
        }
    local_best_dir.mkdir(parents=True, exist_ok=True)
    global_best_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_best_dir / best_path.name
    global_copy = global_best_dir / best_path.name
    shutil.copy2(best_path, local_copy)
    shutil.copy2(best_path, global_copy)
    try:
        comparison = compare_cif_atom_coordinates(best_path, reference_cif)
        lattice_metrics = _lattice_validation_metrics(comparison)
        composition_metrics = _composition_validation_metrics(comparison)
        status = "validated"
    except Exception as exc:
        comparison = {"error": str(exc)}
        lattice_metrics = {}
        composition_metrics = {}
        status = "validation_error"
    pair_distribution_metrics = _pair_distribution_validation_metrics(
        best_path,
        reference_cif,
    )
    attained_solution = bool(
        status == "validated"
        and float(lattice_metrics.get("sorted_abc_relative_error", math.inf))
        <= 0.08
        and float(
            composition_metrics.get("element_count_relative_error", math.inf)
        )
        <= 0.10
    )
    return {
        "status": status,
        "attained_solution": attained_solution,
        "best_generated_cif": str(best_path),
        "best_local_copy": str(local_copy),
        "best_global_copy": str(global_copy),
        "image_fit_metrics": best.get("metrics", {}),
        "density_g_cm3": best.get("density_g_cm3"),
        "density_penalty": best.get("density_penalty"),
        "chemistry_metrics": best.get("chemistry_metrics", {}),
        "physical_penalty": best.get("physical_penalty"),
        "charge_penalty": best.get("charge_penalty"),
        "lattice_prior_penalty": best.get("lattice_prior_penalty"),
        "cif_comparison": comparison,
        "lattice_metrics": lattice_metrics,
        "composition_metrics": composition_metrics,
        "pair_distribution_metrics": pair_distribution_metrics,
    }


def _lattice_validation_metrics(
    comparison: dict[str, Any],
) -> dict[str, float]:
    generated = comparison.get("generated_summary", {}).get("lattice", {})
    reference = comparison.get("reference_summary", {}).get("lattice", {})
    lengths_generated = [
        float(generated.get(axis, math.nan)) for axis in ("a", "b", "c")
    ]
    lengths_reference = [
        float(reference.get(axis, math.nan)) for axis in ("a", "b", "c")
    ]
    sorted_generated = sorted(lengths_generated)
    sorted_reference = sorted(lengths_reference)
    sorted_abs_sum = float(
        sum(abs(g - r) for g, r in zip(sorted_generated, sorted_reference))
    )
    reference_sum = max(
        float(sum(abs(value) for value in sorted_reference)), 1.0e-12
    )
    angle_abs_sum = float(
        sum(
            abs(
                float(generated.get(axis, math.nan))
                - float(reference.get(axis, math.nan))
            )
            for axis in ("alpha", "beta", "gamma")
        )
    )
    return {
        "sorted_abc_abs_sum": sorted_abs_sum,
        "sorted_abc_relative_error": sorted_abs_sum / reference_sum,
        "angle_abs_sum": angle_abs_sum,
        "abc_abs_sum": float(
            comparison.get("lattice_delta", {}).get("abc_abs_sum", math.nan)
        ),
    }


def _composition_validation_metrics(
    comparison: dict[str, Any],
) -> dict[str, float]:
    by_element = (
        comparison.get("coordinate_match", {}).get("by_element", {})
        if isinstance(comparison.get("coordinate_match"), dict)
        else {}
    )
    reference_total = 0.0
    generated_total = 0.0
    absolute_delta = 0.0
    for payload in by_element.values():
        if not isinstance(payload, dict):
            continue
        reference_count = float(payload.get("reference_count", 0.0) or 0.0)
        generated_count = float(payload.get("generated_count", 0.0) or 0.0)
        reference_total += reference_count
        generated_total += generated_count
        absolute_delta += abs(generated_count - reference_count)
    denominator = max(reference_total, 1.0e-12)
    return {
        "generated_element_count": generated_total,
        "reference_element_count": reference_total,
        "element_count_abs_delta": absolute_delta,
        "element_count_relative_error": absolute_delta / denominator,
    }


def _pair_distribution_validation_metrics(
    generated_cif: Path,
    reference_cif: Path,
    *,
    max_distance: float = 8.0,
    bin_width: float = 0.10,
) -> dict[str, Any]:
    try:
        generated_rows, generated_lattice = _read_pair_distribution_sites(
            generated_cif
        )
        reference_rows, reference_lattice = _read_pair_distribution_sites(
            reference_cif
        )
        if generated_lattice is None or reference_lattice is None:
            return {"status": "missing_lattice"}
        bins = np.arange(0.0, max_distance + bin_width, bin_width)
        generated_profiles = _partial_pair_distribution_profiles(
            generated_rows,
            generated_lattice,
            bins,
        )
        reference_profiles = _partial_pair_distribution_profiles(
            reference_rows,
            reference_lattice,
            bins,
        )
        records = []
        weighted_l1 = 0.0
        weighted_rmse = 0.0
        total_weight = 0.0
        for pair in sorted(
            set(generated_profiles) | set(reference_profiles),
            key=_pair_distribution_sort_key,
        ):
            generated = generated_profiles.get(pair)
            reference = reference_profiles.get(pair)
            if generated is None:
                generated = np.zeros(len(bins) - 1, dtype=float)
            if reference is None:
                reference = np.zeros(len(bins) - 1, dtype=float)
            generated_count = float(np.sum(generated))
            reference_count = float(np.sum(reference))
            generated_norm = generated / max(generated_count, 1.0)
            reference_norm = reference / max(reference_count, 1.0)
            diff = generated_norm - reference_norm
            l1_distance = float(np.sum(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            weight = max(generated_count, reference_count, 1.0)
            weighted_l1 += weight * l1_distance
            weighted_rmse += weight * rmse
            total_weight += weight
            records.append(
                {
                    "pair": pair,
                    "generated_pair_count": generated_count,
                    "reference_pair_count": reference_count,
                    "l1_distance": l1_distance,
                    "rmse": rmse,
                    "generated_peak_distances": _pair_distribution_peaks(
                        generated,
                        bins,
                    ),
                    "reference_peak_distances": _pair_distribution_peaks(
                        reference,
                        bins,
                    ),
                }
            )
        records.sort(
            key=lambda item: (
                float(item["l1_distance"]),
                float(item["rmse"]),
                max(
                    float(item["generated_pair_count"]),
                    float(item["reference_pair_count"]),
                ),
            ),
            reverse=True,
        )
        return {
            "status": "computed",
            "max_distance": max_distance,
            "bin_width": bin_width,
            "weighted_l1_distance": float(weighted_l1 / max(total_weight, 1.0)),
            "weighted_rmse": float(weighted_rmse / max(total_weight, 1.0)),
            "worst_partial_pairs": records[:12],
            "policy": (
                "post-hoc partial pair distribution diagnostic only; "
                "reference PDF is not used by the blind solver"
            ),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _partial_pair_distribution_profiles(
    rows: list[dict[str, Any]],
    lattice: np.ndarray,
    bins: np.ndarray,
) -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            distance = _pbc_distance(left["frac"], right["frac"], lattice)
            if distance <= 0.0 or distance >= float(bins[-1]):
                continue
            pair = _pair_distribution_key(
                str(left["element"]),
                str(right["element"]),
            )
            if pair not in profiles:
                profiles[pair] = np.zeros(len(bins) - 1, dtype=float)
            bin_index = int(np.searchsorted(bins, distance, side="right") - 1)
            if 0 <= bin_index < len(profiles[pair]):
                profiles[pair][bin_index] += 1.0
    return profiles


def _read_pair_distribution_sites(
    path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(str(path))
        rows = [
            {
                "label": f"{site.specie.symbol}{index + 1}",
                "element": str(site.specie.symbol),
                "frac": np.asarray(site.frac_coords, dtype=float),
                "row_index": index,
            }
            for index, site in enumerate(structure.sites)
        ]
        return rows, np.asarray(structure.lattice.matrix, dtype=float)
    except Exception:
        return _read_simple_cif_sites(path)


def _pair_distribution_key(left_element: str, right_element: str) -> str:
    left = _element_sort_token(left_element)
    right = _element_sort_token(right_element)
    if _pair_element_rank(left) <= _pair_element_rank(right):
        return f"{left}-{right}"
    return f"{right}-{left}"


def _pair_distribution_sort_key(pair: str) -> tuple[int, int, str]:
    left, _, right = pair.partition("-")
    return (_pair_element_rank(left), _pair_element_rank(right), pair)


def _pair_element_rank(element: str) -> int:
    order = {
        "Pb": 0,
        "Sn": 1,
        "Ge": 2,
        "I": 3,
        "Br": 4,
        "Cl": 5,
        "F": 6,
        "O": 7,
        "N": 8,
        "S": 9,
        "C": 10,
        "H": 11,
    }
    return order.get(_element_sort_token(element), 100)


def _element_sort_token(element: str) -> str:
    letters = "".join(char for char in str(element) if char.isalpha())
    if not letters:
        return str(element)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"


def _pair_distribution_peaks(
    counts: np.ndarray,
    bins: np.ndarray,
    *,
    limit: int = 5,
) -> list[dict[str, float]]:
    if counts.size == 0 or float(np.sum(counts)) <= 0.0:
        return []
    centers = (bins[:-1] + bins[1:]) / 2.0
    indices = np.argsort(counts)[::-1]
    peaks = []
    for index in indices[:limit]:
        if counts[index] <= 0.0:
            continue
        peaks.append(
            {
                "distance": float(centers[index]),
                "count": float(counts[index]),
            }
        )
    return peaks


def _solve_constraints(spec: BenchmarkStructureSpec) -> dict[str, Any]:
    return {
        "inorganic_atoms": list(spec.inorganic_atoms),
        "organic_molecules": list(spec.organic_molecules),
        "molecules": _molecule_records(spec.organic_molecules),
        "excluded_from_solver": [
            "reference_stoichiometry",
            "reference_lattice_constants",
            "reference_fractional_coordinates",
            "reference_space_group",
        ],
    }


def _append_fileset_logbook(
    logbook: Path,
    summary: dict[str, Any],
    params: GIWAXSSimulationParameters,
    peaks: list[StructurePeak],
    candidates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    top = candidates[0] if candidates else {}
    validation = summary.get("validation", {})
    lattice_metrics = validation.get("lattice_metrics", {})
    composition_metrics = validation.get("composition_metrics", {})
    recovery = summary.get("peak_recovery", {})
    lines = [
        "",
        f"## {summary['fileset_id']}",
        "",
        f"- Solve order: {summary['solve_order']}",
        f"- Mock TIFF: `{summary['mock_tiff']}`",
        f"- Project: `{summary['project']}`",
        f"- Peak detection plot: `{summary['peak_detection_plot']}`",
        (
            f"- Orientation: theta_x={params.theta_x_deg:.3f} deg, "
            f"theta_y={params.theta_y_deg:.3f} deg"
        ),
        f"- Peak detection: {len(peaks)} deduplicated q-space peaks",
        (
            "- Peak recovery: "
            f"{recovery.get('matched_truth_peak_count', 'n/a')}/"
            f"{recovery.get('truth_peak_count', 'n/a')} truth peaks, "
            f"precision={float(recovery.get('precision', math.nan)):.3g}, "
            f"recall={float(recovery.get('recall', math.nan)):.3g}"
        ),
        (
            "- Top lattice candidate: "
            + (
                f"{top.get('crystal_system')} a={top.get('a'):.4g} "
                f"b={top.get('b'):.4g} c={top.get('c'):.4g} "
                f"score={top.get('score'):.4g}"
                if top
                else "none"
            )
        ),
        (
            "- Best generated CIF: "
            f"`{validation.get('best_generated_cif', 'none')}`"
        ),
        (
            "- Lattice validation: "
            f"sorted abc relative error="
            f"{lattice_metrics.get('sorted_abc_relative_error', math.nan):.4g}, "
            f"angle abs sum={lattice_metrics.get('angle_abs_sum', math.nan):.4g}"
        ),
        (
            "- Composition validation: "
            "element count relative error="
            f"{composition_metrics.get('element_count_relative_error', math.nan):.4g}"
        ),
        "",
        "Top detected peaks:",
        "",
        "| peak | qxy | qz | intensity | snr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for peak in peaks[:10]:
        lines.append(
            "| {label} | {qxy:.4f} | {qz:.4f} | {intensity:.4g} | {snr:.4g} |".format(
                label=peak.label,
                qxy=peak.qxy,
                qz=peak.qz,
                intensity=float(
                    peak.metadata.get("detected_intensity") or 0.0
                ),
                snr=float(peak.metadata.get("snr") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "Generated CIF ranking:",
            "",
            _markdown_generated_table(comparisons),
        ]
    )
    _append_logbook(logbook, lines)


def _markdown_generated_table(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return "No generated CIF rankings."
    lines = [
        "| rank | cif | peak focus | rmse | corr |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for item in comparisons[:8]:
        metrics = item.get("metrics", {})
        lines.append(
            "| {rank} | `{cif}` | {focus:.4g} | {rmse:.4g} | {corr:.4g} |".format(
                rank=item.get("fit_rank", ""),
                cif=item.get("generated_cif_id", ""),
                focus=float(metrics.get("peak_focus_score", math.nan)),
                rmse=float(metrics.get("difference_rmse", math.nan)),
                corr=float(metrics.get("correlation", math.nan)),
            )
        )
    return "\n".join(lines)


def _markdown_summary_table(filesets: list[dict[str, Any]]) -> str:
    if not filesets:
        return "No filesets were run."
    lines = [
        "| fileset | peaks | best CIF | sorted abc rel err | project |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for item in filesets:
        metrics = item.get("validation", {}).get("lattice_metrics", {})
        lines.append(
            "| `{fileset}` | {peaks} | `{best}` | {rel:.4g} | `{project}` |".format(
                fileset=item.get("fileset_id", ""),
                peaks=item.get("peak_count", 0),
                best=Path(str(item.get("best_generated_cif", "none"))).name,
                rel=float(metrics.get("sorted_abc_relative_error", math.nan)),
                project=item.get("project", ""),
            )
        )
    return "\n".join(lines)


def _append_logbook(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip())
        handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_safe(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum():
            chars.append(char.lower())
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "benchmark"
